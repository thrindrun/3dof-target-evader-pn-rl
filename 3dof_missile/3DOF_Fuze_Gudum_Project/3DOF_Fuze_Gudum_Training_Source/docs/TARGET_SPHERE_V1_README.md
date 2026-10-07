# Target Sphere V1 — hedef merkezli 5 km fırlatma küresi

Bu sürümde amaç şudur:

- Hedef, küçük jitter dışında aynı başlangıç bölgesinde ve aynı global düz uçuş
  yönünde kalır.
- Füze, hedef merkezli 5 km yarıçaplı kürenin yüzeyinde örneklenir.
- Füze başlangıçta doğrudan hedefe bakar.
- Hedefin heading'i fırlatma noktasına göre çevrilmez. Bu nedenle hedefin füze
  açısından önden, arkadan ve yandan görünüşleri küre boyunca kendiliğinden
  oluşur.
- Mevcut `legacy`, `rotated_nominal` ve `polar` modları korunur. Yeni modun adı
  `target_sphere`dır.

Matematiksel olarak hedefin konumu `T`, örneklenen birim küre vektörü `u` ve
başlangıç menzili `R` ise:

```text
missile_position = T + R * u
missile_direction = normalize(T - missile_position)
```

Örneğin hedefin yatay konumu `(400, 0)` ve `R=5000 m` ise bu iki fırlatma da
aynı kabuğun üzerindedir:

```text
u = (1.0, 0.0)  -> füze = (5400,    0)  -> uzaklık 5000 m
u = (0.6, 0.8)  -> füze = (3400, 4000)  -> uzaklık 5000 m
```

Test dosyası bu iki bağıl örneği sayısal olarak ayrıca doğrular.

## `scale=0.5` neyi değiştirir?

`--randomization-scale 0.5`, 5 km küreyi küçültmez ve füze noktalarını küçük
bir buluta sıkıştırmaz. Şunların mevcut jitter sınırlarını yarıya indirir:

- hedef merkez konumu,
- hedef ve füze hızı,
- hedef heading ve flight-path açısı,
- legacy modundaki diğer küçük başlangıç sapmaları.

Kürenin boyutu ve kapsadığı yönler yalnız şu argümanlarla belirlenir:

- `--range-min-m`, `--range-max-m`
- `--launch-azimuth-min-deg`, `--launch-azimuth-max-deg`
- `--launch-elevation-min-deg`, `--launch-elevation-max-deg`

## NED ve yer düzlemi notu

Simülasyon `NED = North, East, Down` kullanır; irtifa `h=-D`dir. 3.3 km
irtifadaki bir hedefin çevresindeki 5 km tam kürenin alt bölümü fiziksel yer
düzleminin altında kalır. Noktaları elemek tam küre hedefini bozacağı için
`target_sphere` varsayılan olarak hedef merkezini yalnız irtifada, bütün küre en
az 100 m yer payıyla sığacak kadar yükseltir. N/E konumu ve bağıl geometri
değişmez. Uygulanan yükseltme her episode CSV'sindeki
`target_sphere_altitude_shift_m` alanında kaydedilir.

Yalnız yatay çember (`elevation=0`) testinde bu yükseltmeye gerek kalmaz.

## Kurulum

PowerShell'i proje kökünde açın:

```powershell
cd C:\Projects\missile_guidance
Expand-Archive .\source_target_sphere_v1_ready.zip -DestinationPath . -Force
python -m py_compile .\air_to_air_3dof_ppo.py .\training\train_ppo.py .\tests\evaluate_ppo_geometry.py
python .\tests\check_geometry_v1.py
python .\tests\check_target_sphere_v1.py
```

Son satırın beklenen özeti:

```text
Target Sphere V1 kontrolleri başarılı.
İstenen iki yatay 5 km örneği           : doğru
Doldurulan 30 derece azimut sektörü     : 12
Doldurulan eşit-alan elevation bandı    : 6
```

## 1. Önce mevcut modeli yatay tam çemberde ölçün

Bu testte hedef küçük değişir; füze hedefin yatay çevresindeki bütün 360 derece
fırlatma noktalarından gelir ve hedefe bakar:

```powershell
python .\tests\evaluate_ppo_geometry.py --model .\models\obs_v2_random50_9m_1m\best_model.zip --episodes 500 --seed 320000 --geometry-mode target_sphere --range-min-m 5000 --range-max-m 5000 --launch-azimuth-min-deg -180 --launch-azimuth-max-deg 180 --launch-elevation-min-deg 0 --launch-elevation-max-deg 0 --aim-azimuth-error-deg 0 --aim-elevation-error-deg 0 --randomization-scale 0.5 --passed-target-arm-progress-m 100 --sector-count 12 --elevation-band-count 1 --sector-threshold-m 100 --thresholds 500 200 100 50 25 10
```

## 2. Ardından mevcut modeli tam 3B kürede ölçün

```powershell
python .\tests\evaluate_ppo_geometry.py --model .\models\obs_v2_random50_9m_1m\best_model.zip --episodes 500 --seed 330000 --geometry-mode target_sphere --range-min-m 5000 --range-max-m 5000 --launch-azimuth-min-deg -180 --launch-azimuth-max-deg 180 --launch-elevation-min-deg -90 --launch-elevation-max-deg 90 --aim-azimuth-error-deg 0 --aim-elevation-error-deg 0 --sphere-ground-clearance-m 100 --randomization-scale 0.5 --passed-target-arm-progress-m 100 --sector-count 12 --elevation-band-count 6 --sector-threshold-m 100 --thresholds 500 200 100 50 25 10
```

Tam küre örneklemesi `sin(elevation)` uzayında uniform yapılır. Bu, kutupları
gereğinden fazla seçmeden küre yüzeyinde eşit alanlara eşit olasılık verir.
Çıktı hem 12 azimut sektörünü hem 6 eşit-alan elevation bandını ayrı raporlar.

## Tek episode ayrıntılı 3B yörünge grafiği

Bu test, seçilen bir seed için füze ve hedef yollarını büyük bir 3B grafikte;
menzil, irtifa ve PPO ivme komutunu ise ayrı zaman grafiklerinde gösterir.
Başlangıçtaki 5 km görüş hattı ile en yakın yaklaşma doğrusu da işaretlenir:

```powershell
python .\tests\plot_ppo_target_sphere.py --model .\models\obs_v2_random50_9m_1m\best_model.zip --radius 10 --seed 330000 --range-m 5000 --launch-azimuth-min-deg -180 --launch-azimuth-max-deg 180 --launch-elevation-min-deg -90 --launch-elevation-max-deg 90 --aim-azimuth-error-deg 0 --aim-elevation-error-deg 0 --sphere-ground-clearance-m 100 --randomization-scale 0.5 --passed-target-arm-progress-m 100
```

Grafik ekranda açılır ve ayrıca şu klasöre PNG olarak kaydedilir:

```text
results\trajectory\
```

Aynı episode'un zaman, menzil, komut ve NED konum geçmişi aynı klasörde NPZ
olarak saklanır. Başka bir geometri görmek için yalnız `--seed` değerini
değiştirin. Örneğin `330001`, `330002` ve `330003` birbirinden farklı küre
noktaları üretir.

## Eğitim notu

Doğrudan 9–10 m vurma yarıçapında tam küre eğitimine geçmek ağır bir dağılım
değişimidir. Önce yukarıdaki iki test sonucunu kaydedin. Sonuca göre güvenli
curriculum sırası şudur:

1. tam yatay çember, geniş vurma yarıçapı;
2. elevation `-20..20`;
3. elevation `-45..45`;
4. tam küre `-90..90`;
5. bundan sonra vurma yarıçapını aşamalı küçültme.

Bu süreçte observation sürümü değişmez: model yine `v2 (23 girdi)` kullanır.

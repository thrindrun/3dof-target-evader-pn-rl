# Fresh-Start Predictive-CPA V6

Bu sürüm, eski dar-geometri checkpoint'ini kullanmadan 23 girdili PPO ağını
doğrudan tam yatay 5 km çemberinde sıfırdan eğitmek içindir.

## Neler değişti?

- `--fresh-start`: yanlışlıkla `--source` verilmesini engelleyen açık sıfırdan
  başlama doğrulaması.
- `--predictive-cpa-reward`: mevcut bağıl konum ve hız sabit kalırsa oluşacak
  tahminî en-yakın-geçiş mesafesindeki iyileşmeye erken reward verir.
- Gerçek minimum menzile dayalı kesintisiz Reward V4 aynen korunur.
- `--eval-sector-count 12`: her doğrulamada 12 azimut sektörüne eşit episode
  ayırır ve en kötü sektör ortalamasını checkpoint seçiminde kullanır.
- Predictive CPA kapalıyken V5/V4/V3 sayısal davranışı korunur.

Predictive CPA bir güdüm komutu veya PN değildir. Yalnızca PPO'nun kilometrelerce
uzaktayken yaptığı erken önleme düzeltmesinin iyi mi kötü mü olduğunu reward'a
aktarır. PPO iki normal-ivme komutunu üretmeye devam eder.

## 1. Kurulum

Zip dosyasını proje köküne koyun ve PowerShell'de çalıştırın:

```powershell
cd C:\Projects\missile_guidance
Expand-Archive .\source_fresh_start_cpa_v6_ready.zip -DestinationPath . -Force
```

Bu paket model klasörlerini içermez ve mevcut modelleri silmez.

## 2. Hızlı kontroller

```powershell
python -m py_compile .\air_to_air_3dof_ppo.py .\training\train_ppo.py .\tests\evaluate_ppo_geometry.py .\tests\check_fresh_start_cpa_v6.py
python .\tests\check_geometry_v1.py
python .\tests\check_target_sphere_v1.py
python .\tests\check_closest_approach_v4.py
python .\tests\check_priority_replay_v5.py
python .\tests\check_fresh_start_cpa_v6.py
```

Son satırlar arasında şunlar görünmelidir:

```text
Fresh-Start Predictive-CPA V6 kontrolleri başarılı.
Reward kapalıyken eski dinamik        : korunuyor
12 sektörlü doğrulama episode sayısı  : eşit
```

## 3. İlk sıfırdan eğitim: 1000 m

Bu ilk koşu eski model yüklemez. Tam çember, yatay launch, düz hedef ve
`randomization-scale=0.5` kullanır. Ara doğrulama maliyetini azaltmak için her
50 bin adımda 120 episode çalıştırılır; 12 sektörün her biri 10 episode alır.

```powershell
python .\training\train_ppo.py `
  --stage fresh_cpa_v6_1000m `
  --radius 1000 `
  --steps 200000 `
  --fresh-start `
  --n-envs 12 `
  --seed 260912 `
  --randomize `
  --randomization-scale 0.5 `
  --learning-rate 0.0003 `
  --action-std 0.35 `
  --eval-episodes 120 `
  --eval-seed 700000 `
  --eval-frequency 50000 `
  --eval-sector-count 12 `
  --observation-version v2 `
  --geometry-mode target_sphere `
  --range-min-m 5000 `
  --range-max-m 5000 `
  --launch-azimuth-min-deg -180 `
  --launch-azimuth-max-deg 180 `
  --launch-elevation-min-deg 0 `
  --launch-elevation-max-deg 0 `
  --aim-azimuth-error-deg 0 `
  --aim-elevation-error-deg 0 `
  --sphere-ground-clearance-m 100 `
  --passed-target-arm-progress-m 100 `
  --closest-approach-training `
  --predictive-cpa-reward `
  --predictive-cpa-weight 1.0 `
  --predictive-cpa-scale-m 100 `
  --predictive-cpa-clip-m 250 `
  --predictive-cpa-horizon-s 35
```

Başlangıç başlığında özellikle şunları kontrol edin:

```text
Başlangıç modu       : SIFIRDAN — yeni actor ve critic
Kaynak model         : YOK — sıfırdan eğitim
Observation          : v2 (23 girdi)
Başlangıç PPO adımı  : 0
Priority Replay V5   : False
Paralel ortam sayısı : 12
Rollout örnek sayısı : 12288
Dengeli azimut sektörü: 12
Closest-approach V4  : True
Predictive CPA V6    : True
```

`--source`, `--reset-critic` ve priority-window argümanlarını bu ilk komuta
eklemeyin.

Doğrulamada yazılan 12 sektör başarı değeri soldan sağa şu aralıklardır:

```text
[-180,-150), [-150,-120), ... , [120,150), [150,180)
```

## 4. İlk stage sonrası bağımsız test

Eğitim bitince `final_model.zip` yerine `best_model.zip` test edilmelidir:

```powershell
python .\tests\evaluate_ppo_geometry.py `
  --model .\models\fresh_cpa_v6_1000m\best_model.zip `
  --episodes 500 `
  --seed 710000 `
  --geometry-mode target_sphere `
  --range-min-m 5000 `
  --range-max-m 5000 `
  --launch-azimuth-min-deg -180 `
  --launch-azimuth-max-deg 180 `
  --launch-elevation-min-deg 0 `
  --launch-elevation-max-deg 0 `
  --aim-azimuth-error-deg 0 `
  --aim-elevation-error-deg 0 `
  --sphere-ground-clearance-m 100 `
  --randomization-scale 0.5 `
  --passed-target-arm-progress-m 100 `
  --sector-count 12 `
  --elevation-band-count 1 `
  --sector-threshold-m 1000 `
  --thresholds 2000 1500 1000 750 500 300 200 150 100 75 50 25 10
```

Bu sonuç görülmeden 500 m stage'ine geçmeyin. İlk kararda yalnız toplam başarıya
değil, 12 sektörün ortalama/P90 değerlerine ve eğitim sırasında yazılan "en kötü
sektör ortalaması"na bakılmalıdır.

## Planlanan curriculum

İlk sonuç olumluysa aynı reward sürümü korunarak yaklaşık şu sıra izlenecektir:

```text
1000 m -> 500 m -> 300 m -> 150 m -> 100 m -> 75 m
```

Yalnız ilk stage sıfırdan başlar. Sonraki stage'ler bir öncekinin
`best_model.zip` dosyasından devam eder; artık Reward V6 değişmediği için critic
yeniden sıfırlanmaz.

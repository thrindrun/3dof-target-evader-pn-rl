# Closest-Approach V4 — 75 m tam çember deneyi

Bu paket, mevcut Target Sphere V1 altyapısına isteğe bağlı bir eğitim modu
ekler. Observation ve actor mimarisi değişmez: model yine `v2 (23 girdi)`
kullanır ve mevcut `best_model.zip` doğrudan kaynak alınabilir.

## Neden V4?

Eski V3 eğitiminde episode, curriculum yarıçapına ilk girildiği anda biterdi.
Örneğin 100 m aşamasında politika 99 m gördüğünde artık 20 m, 5 m veya gerçek
en yakın geçişi gözlemlemiyordu. Sonraki aşamayı 75 m yapmak bu nedenle hem yeni
bir terminal dağılımı oluşturuyor hem de 74.9 m ile 75.1 m arasında büyük bir
reward sıçraması yaratıyordu.

`--closest-approach-training` açıldığında:

- 75 m yalnız başarı/curriculum eşiği olur;
- episode eşikte kesilmez, gerçek en yakın geçiş tamamlanana kadar sürer;
- terminal reward minimum menzile göre kesintisizdir;
- 75 m'nin içinde de daha küçük minimum menzil daha fazla reward alır;
- doğrulama callback'i başarıyı `minimum_range <= radius` üzerinden ölçer;
- küçük başarı oynamalarının çok kötü P90 kuyruğunu gizlemesi zorlaştırılır.

Bayrak verilmezse eski V3 davranışı aynen korunur. Bu nedenle eski komutlar ve
modeller geriye dönük uyumludur.

Varsayılan katsayılarla 75 m aşamasındaki ham terminal doğruluk terimi:

| Minimum menzil | Terminal doğruluk terimi |
|---:|---:|
| 0 m | +50 |
| 37.5 m | +25 |
| 75 m | 0 |
| 112.5 m | -25 |
| 150 m | -50 |
| 225 m ve üstü | -100 |

## Paketin kurulumu

ZIP'i proje klasörüne indirin. VS Code PowerShell terminalinde:

```powershell
cd C:\Projects\missile_guidance
Expand-Archive .\source_closest_approach_v4_ready.zip -DestinationPath . -Force
python -m py_compile .\air_to_air_3dof_ppo.py .\training\train_ppo.py .\tests\evaluate_ppo_geometry.py .\tests\check_closest_approach_v4.py
python .\tests\check_geometry_v1.py
python .\tests\check_target_sphere_v1.py
python .\tests\check_closest_approach_v4.py
```

Son kontrolün başında şu satır görülmelidir:

```text
Closest-Approach V4 kontrolleri başarılı.
```

Paket model, sonuç, checkpoint veya log klasörlerini içermez; mevcut eğitim
ürünlerine dokunmaz. `Expand-Archive -Force` yalnız paketteki kaynak/test
dosyalarını günceller.

## İlk deney: 75 m, 300 bin adım

İlk çalışmayı 3 milyon adımla başlatmayın. Aşağıdaki 300 bin adım, yeni reward
tasarımının gerçekten doğru yöne gittiğini ucuz biçimde sınar. Kaynak olarak
75 m denemesini değil, son güvenilir 100 m şampiyonunu kullanır.

```powershell
python .\training\train_ppo.py --stage target_sphere_ring75_closest_v4_300k --radius 75 --steps 300000 --source .\models\target_sphere_ring100_consolidate_3m_v1\best_model.zip --n-envs 8 --seed 120912 --randomize --randomization-scale 0.5 --learning-rate 0.000003 --action-std 0.025 --reset-critic --eval-episodes 300 --eval-seed 600000 --eval-frequency 20000 --observation-version v2 --geometry-mode target_sphere --range-min-m 5000 --range-max-m 5000 --launch-azimuth-min-deg -180 --launch-azimuth-max-deg 180 --launch-elevation-min-deg 0 --launch-elevation-max-deg 0 --aim-azimuth-error-deg 0 --aim-elevation-error-deg 0 --sphere-ground-clearance-m 100 --passed-target-arm-progress-m 100 --closest-approach-training
```

Başlangıç özetinde özellikle şunları kontrol edin:

```text
Observation          : v2 (23 girdi)
Critic sıfırlandı    : True
Geometri modu        : target_sphere
Paralel ortam sayısı : 8
Closest-approach V4  : True
Doğrulama seedleri   : 600000-600299
```

`--reset-critic` yalnız eski reward'a kalibre olmuş value/critic dalını ve
optimizer geçmişini sıfırlar. Füze komutunu üreten actor ağırlıkları korunur.

## Eğitimden sonra bağımsız 500-episode testi

Bu test eğitim doğrulamasından farklı seedleri kullanır ve gerçek minimum
menzili ölçebilmek için episode'u mikroskobik hit eşiğiyle en yakın geçişe kadar
sürdürür:

```powershell
python .\tests\evaluate_ppo_geometry.py --model .\models\target_sphere_ring75_closest_v4_300k\best_model.zip --episodes 500 --seed 320000 --geometry-mode target_sphere --range-min-m 5000 --range-max-m 5000 --launch-azimuth-min-deg -180 --launch-azimuth-max-deg 180 --launch-elevation-min-deg 0 --launch-elevation-max-deg 0 --aim-azimuth-error-deg 0 --aim-elevation-error-deg 0 --sphere-ground-clearance-m 100 --randomization-scale 0.5 --passed-target-arm-progress-m 100 --sector-count 12 --elevation-band-count 1 --sector-threshold-m 75 --thresholds 1000 750 500 300 200 150 100 75 50 25 10
```

## Karar ölçütü

Aynı `320000-320499` test grubunda kaynak 100 m modelinin referansı:

- 75 m başarı: `%67.40`
- ortalama minimum menzil: `61.711 m`
- P90 minimum menzil: `116.126 m`

300 bin adım sonunda önce şu üç değeri karşılaştırın. Yeni modelin 75 m
başarısı gerilememeli; ortalama ve P90'ın en az biri belirgin biçimde düşmeli.
Sektör tablosunda daha önce zor olan arka/yan sektörlerin yüzlerce metreye
kaçmaması da gerekir. Sonuç bu yöndeyse aynı V4 tasarımıyla daha uzun bir
konsolidasyon aşamasına geçilir. Değilse milyonlarca adım eklemek yerine reward
ve terminal telemetrisi yeniden incelenir.

`final_model.zip` yerine daima doğrulama tarafından seçilen
`best_model.zip` test edilmelidir.

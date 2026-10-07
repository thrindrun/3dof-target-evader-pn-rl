# Priority Replay V5 — zor crossing sektörleri

Bu paket Closest-Approach V4'ü korur ve yalnız eğitim başlangıç geometrisi
örneklemesine isteğe bağlı bir karışım ekler. Observation değişmez:
`v2 (23 girdi)`. Mevcut V4 modelinin actor ve critic ağırlıkları doğrudan
kullanılabilir.

## Neyi çözüyor?

Uniform tam çember testinde başarısızlıkların çoğu şu dört 30 derecelik
crossing sektöründe toplandı:

```text
[-150, -120), [-120, -90), [120, 150), [150, 180)
```

V5 komutunda bunlar iki bitişik pencere olarak tanımlanır:

```text
[-150, -90), [120, 180)
```

Her eğitim episode'u için:

- `%60` olasılıkla füze bu iki priority pencerenin birleşiminden seçilir;
- `%40` olasılıkla füze bütün `-180..180°` çemberden uniform seçilir.

Priority alanı tam çemberin üçte birini kapladığı için uniform replay'in de bir
kısmı bu alanlara düşer. Böylece toplam eğitim örneklerinin beklenen `%73.33`ü
zor bölgelerde, `%26.67`si diğer bölgelerde olur. Bu ikinci kısım kolay
sektörlerin unutulmasını engelleyen replay'dir.

Doğrulama dağılımı eğitim dağılımından bilerek ayrıdır. Kod doğrulama ortamında
priority oranını otomatik olarak sıfırlar ve bütün `-180..180°` çemberi uniform
ölçer. Dolayısıyla doğrulama başarı oranı zor örneklerin yapay biçimde fazla ya
da az sayılmasından etkilenmez.

## Kurulum ve kontroller

ZIP'i `C:\Projects\missile_guidance` klasörüne indirin ve VS Code PowerShell
terminalinde çalıştırın:

```powershell
cd C:\Projects\missile_guidance
Expand-Archive .\source_priority_replay_v5_ready.zip -DestinationPath . -Force
python -m py_compile .\air_to_air_3dof_ppo.py .\training\train_ppo.py .\tests\check_priority_replay_v5.py
python .\tests\check_geometry_v1.py
python .\tests\check_target_sphere_v1.py
python .\tests\check_closest_approach_v4.py
python .\tests\check_priority_replay_v5.py
```

Son kontrolün beklenen ana çıktıları yaklaşık şöyledir:

```text
Priority Replay V5 kontrolleri başarılı.
Priority dal gerçekleşme oranı        : %60 civarı
Zor bölgede toplam örnek oranı        : %73.33 civarı
Uniform replay ile dolan sektör sayısı: 12
Doğrulama priority oranı              : %0.00
```

Paket herhangi bir model, checkpoint, sonuç veya log içermez ve bu klasörlere
dokunmaz. Bayraklar verilmezse V5 örneklemesi kapalıdır; önceki V4/V3 komutları
eski davranışlarıyla çalışır.

## İlk kontrollü V5 eğitimi: 400 bin adım

Kaynak olarak yeni V4 75 m `best_model.zip` kullanılır. Reward hâlâ aynı V4
reward'u olduğu için critic yeniden sıfırlanmaz. Yalnız action std çok küçük bir
artışla `0.03` yapılır.

```powershell
python .\training\train_ppo.py --stage target_sphere_ring75_priority_v5_400k --radius 75 --steps 400000 --source .\models\target_sphere_ring75_closest_v4_300k\best_model.zip --n-envs 8 --seed 130912 --randomize --randomization-scale 0.5 --learning-rate 0.000003 --action-std 0.03 --eval-episodes 300 --eval-seed 620000 --eval-frequency 20000 --observation-version v2 --geometry-mode target_sphere --range-min-m 5000 --range-max-m 5000 --launch-azimuth-min-deg -180 --launch-azimuth-max-deg 180 --launch-elevation-min-deg 0 --launch-elevation-max-deg 0 --aim-azimuth-error-deg 0 --aim-elevation-error-deg 0 --sphere-ground-clearance-m 100 --passed-target-arm-progress-m 100 --closest-approach-training --priority-azimuth-probability 0.60 --priority-azimuth-window -150 -90 --priority-azimuth-window 120 180
```

Başlangıç tablosunda şunları doğrulayın:

```text
Kaynak model         : ...target_sphere_ring75_closest_v4_300k\best_model.zip
Critic sıfırlandı    : False
Priority Replay V5   : True
Priority pencereler  : [-150, -90), [120, 180)
Priority dal olasılığı: %60.00
Uniform replay oranı : %40.00
Zor bölgede beklenen : %73.33
Doğrulama dağılımı   : uniform ana launch aralığı
Closest-approach V4  : True
Doğrulama seedleri   : 620000-620299
```

## Eğitimden sonraki karşılaştırmalı test

Daima `best_model.zip` kullanın:

```powershell
python .\tests\evaluate_ppo_geometry.py --model .\models\target_sphere_ring75_priority_v5_400k\best_model.zip --episodes 500 --seed 320000 --geometry-mode target_sphere --range-min-m 5000 --range-max-m 5000 --launch-azimuth-min-deg -180 --launch-azimuth-max-deg 180 --launch-elevation-min-deg 0 --launch-elevation-max-deg 0 --aim-azimuth-error-deg 0 --aim-elevation-error-deg 0 --sphere-ground-clearance-m 100 --randomization-scale 0.5 --passed-target-arm-progress-m 100 --sector-count 12 --elevation-band-count 1 --sector-threshold-m 75 --thresholds 1000 750 500 300 200 150 100 75 50 25 10
```

V4 kaynak modelinin aynı seed grubundaki referansı:

- 75 m başarı: `%68.00`
- ortalama minimum menzil: `60.350 m`
- medyan minimum menzil: `53.676 m`
- P90 minimum menzil: `116.372 m`

Bu aşamanın asıl amacı yalnız medyanı küçültmek değildir. P90'ın düşmesi ve
önceden `%0` olan `[-120,-90)` ile `[120,150)` sektörlerinden en az birinin hit
üretmeye başlaması gerekir. Tam çember toplam başarısı ve kolay sektörler de
gerilememelidir. Bunlar gerçekleşmeden eğitimi milyonlarca adım uzatmayın.

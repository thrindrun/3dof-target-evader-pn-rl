# Target Sphere V1 — 3B yörünge testi eklentisi

Bu küçük eklenti mevcut `source_target_sphere_v1_ready.zip` kurulumuna yalnızca
şu dosyayı ekler:

```text
tests\plot_ppo_target_sphere.py
```

## Kurulum

ZIP dosyasını `C:\Projects\missile_guidance` klasörüne indirin ve proje
terminalinde çalıştırın:

```powershell
cd C:\Projects\missile_guidance
Expand-Archive .\target_sphere_trajectory_test_v1.zip -DestinationPath . -Force
python -m py_compile .\tests\plot_ppo_target_sphere.py
```

## Tam küreden tek örnek ve 3B yol grafiği

```powershell
python .\tests\plot_ppo_target_sphere.py --model .\models\obs_v2_random50_9m_1m\best_model.zip --radius 10 --seed 330000 --range-m 5000 --launch-azimuth-min-deg -180 --launch-azimuth-max-deg 180 --launch-elevation-min-deg -90 --launch-elevation-max-deg 90 --aim-azimuth-error-deg 0 --aim-elevation-error-deg 0 --sphere-ground-clearance-m 100 --randomization-scale 0.5 --passed-target-arm-progress-m 100
```

Grafik ekranda açılır. PNG ve NPZ dosyaları `results\trajectory` klasörüne
kaydedilir. Başka başlangıç noktaları görmek için yalnız `--seed` değerini
örneğin `330001`, `330002` veya `330003` yapın.

## Yalnız yatay çemberden tek örnek

Füze ve hedef başlangıçta aynı irtifada olsun istenirse komuttaki elevation
sınırları şöyle değiştirilir:

```text
--launch-elevation-min-deg 0 --launch-elevation-max-deg 0
```

Grafiğin 3B panelinde füze/hedef yolları, başlangıç ve bitiş noktaları,
başlangıçtaki 5 km görüş hattı ve en yakın yaklaşma bağlantısı gösterilir.
Sağdaki paneller menzil, iki aracın irtifası ve PPO ivme komutudur.

## Aynı hedefe iki farklı füze yolu

Mevcut ortam tek füze + tek hedef olduğu için bu görselleştirme aynı hedef
başlangıç durumu ve aynı seed ile iki bağımsız PPO episode'u çalıştırır. İki
füze yolu tek 3B grafikte karşılaştırılır:

```powershell
python .\tests\plot_ppo_two_missiles.py --model .\models\obs_v2_random50_9m_1m\best_model.zip --radius 10 --seed 330000 --range-m 5000 --missile-1-azimuth-deg 0 --missile-1-elevation-deg -25 --missile-2-azimuth-deg 120 --missile-2-elevation-deg 35 --randomization-scale 0.5 --sphere-ground-clearance-m 100 --passed-target-arm-progress-m 100
```

İki füzenin de hedefle aynı irtifadan başladığı eski yatay örneği çizmek için:

```powershell
python .\tests\plot_ppo_two_missiles.py --model .\models\obs_v2_random50_9m_1m\best_model.zip --radius 10 --seed 330000 --range-m 5000 --missile-1-azimuth-deg 0 --missile-1-elevation-deg 0 --missile-2-azimuth-deg 53.130102 --missile-2-elevation-deg 0 --randomization-scale 0.5 --sphere-ground-clearance-m 100 --passed-target-arm-progress-m 100
```

İkinci komutta hedef merkezine göre füze başlangıç offsetleri yaklaşık
`(5000, 0, 0)` ve `(3000, 4000, 0)` metredir. Terminalde yazan
`Ortak hedef yol farkı maks.` değerinin sıfıra çok yakın olması, iki grafikteki
hedef senaryosunun aynı olduğunu doğrular.

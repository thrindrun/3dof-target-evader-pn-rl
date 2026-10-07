$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

# Bu komut, training_metadata.json kaydindaki son hard-pocket asamasini yeniden
# kurar. Birebir calismasi icin onceki curriculum checkpoint'i su konumda
# bulunmalidir:
# models/fresh_cpa_v6_el30_100m_1m/best_model.zip
# Bu onceki checkpoint bu teslim paketinde bulunmadigindan komut varsayilan
# durumda bilincli olarak baslamaz ve acik bir FileNotFoundError verir.
& .\.venv\Scripts\python.exe .\train.py `
  --stage fresh_cpa_v6_hardpocket_7p5m_300k_reproduction `
  --radius 7.5 `
  --steps 300000 `
  --source .\models\fresh_cpa_v6_el30_100m_1m\best_model.zip `
  --learning-rate 0.00001 `
  --action-std 0.05 `
  --reset-critic `
  --observation-version v2 `
  --seed 42 `
  --randomize `
  --randomization-scale 0.5 `
  --geometry-mode target_sphere `
  --range-min-m 5000 `
  --range-max-m 5000 `
  --launch-azimuth-min-deg -180 `
  --launch-azimuth-max-deg 180 `
  --launch-elevation-min-deg -60 `
  --launch-elevation-max-deg -40 `
  --priority-azimuth-window -180 -150 `
  --priority-azimuth-probability 0.7 `
  --aim-azimuth-error-deg 0 `
  --aim-elevation-error-deg 0 `
  --sphere-ground-clearance-m 100 `
  --sphere-auto-raise-target `
  --legacy-replay-probability 0 `
  --passed-target-arm-progress-m 100 `
  --closest-approach-training `
  --predictive-cpa-reward `
  --predictive-cpa-weight 1 `
  --predictive-cpa-scale-m 100 `
  --predictive-cpa-clip-m 250 `
  --predictive-cpa-horizon-s 35 `
  --n-envs 12 `
  --eval-episodes 600 `
  --eval-seed 810000 `
  --eval-frequency 50000 `
  --eval-sector-count 12


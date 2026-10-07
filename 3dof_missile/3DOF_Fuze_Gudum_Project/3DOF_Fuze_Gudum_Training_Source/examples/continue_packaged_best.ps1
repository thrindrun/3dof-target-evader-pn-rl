$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

# Teslim edilen best checkpoint'ten yeni ve ayri bir stage'e 50 bin adim ekler.
# Eski dosyayi ezmez. Reward degismedigi icin critic sifirlanmaz.
& .\.venv\Scripts\python.exe .\train.py `
  --stage packaged_best_continue_50k `
  --radius 7.5 `
  --steps 50000 `
  --source .\models\fresh_cpa_v6_hardpocket_7p5m_300k\best_model.zip `
  --learning-rate 0.000005 `
  --action-std 0.03 `
  --observation-version v2 `
  --seed 43 `
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
  --sphere-ground-clearance-m 100 `
  --sphere-auto-raise-target `
  --passed-target-arm-progress-m 100 `
  --closest-approach-training `
  --predictive-cpa-reward `
  --n-envs 12 `
  --eval-episodes 120 `
  --eval-seed 995000 `
  --eval-frequency 25000 `
  --eval-sector-count 12


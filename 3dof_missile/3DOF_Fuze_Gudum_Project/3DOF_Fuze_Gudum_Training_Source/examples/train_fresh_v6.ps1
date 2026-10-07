$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

# Yeni actor ve critic ile gercekten num_timesteps=0'dan baslar.
# Bu, final modelin birebir yeniden uretimi degil; V6 egitim akisini temiz bir
# deney olarak gosteren 200 bin adimlik ornek calismadir.
& .\.venv\Scripts\python.exe .\train.py `
  --stage demo_fresh_v6_100m `
  --radius 100 `
  --steps 200000 `
  --fresh-start `
  --record-initial-evaluation `
  --observation-version v2 `
  --reward-profile predictive_cpa `
  --learning-rate 0.0003 `
  --action-std 0.5 `
  --seed 42 `
  --randomize `
  --randomization-scale 0.5 `
  --geometry-mode target_sphere `
  --range-min-m 5000 `
  --range-max-m 5000 `
  --launch-azimuth-min-deg -180 `
  --launch-azimuth-max-deg 180 `
  --launch-elevation-min-deg -30 `
  --launch-elevation-max-deg 30 `
  --aim-azimuth-error-deg 0 `
  --aim-elevation-error-deg 0 `
  --sphere-ground-clearance-m 100 `
  --sphere-auto-raise-target `
  --passed-target-arm-progress-m 100 `
  --n-envs 12 `
  --eval-episodes 120 `
  --eval-seed 700000 `
  --eval-frequency 20000 `
  --eval-sector-count 12


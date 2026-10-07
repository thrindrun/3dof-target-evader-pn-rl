$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

# Egitim baslatmaz. Altı reward ve uc observation kosusunun komutlarini ve
# manifestini uretir; deney tasarimini hizli denetlemek icindir.
& .\.venv\Scripts\python.exe .\training\run_learning_study.py `
  --phase all `
  --tag teacher_demo `
  --dry-run `
  --no-plots


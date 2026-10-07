@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Once setup.bat dosyasini calistirin.
    pause
    exit /b 1
)

set "PY=.venv\Scripts\python.exe"
"%PY%" -m py_compile air_to_air_3dof_ppo.py train.py training\train_ppo.py training\run_learning_study.py tests\plot_learning_study.py
if errorlevel 1 goto :fail

"%PY%" train.py --help >nul
if errorlevel 1 goto :fail

"%PY%" training\run_learning_study.py --phase reward --tag package_dry_run --dry-run --no-plots
if errorlevel 1 goto :fail

echo Kaynak, CLI ve dry-run dogrulamasi basarili.
pause
exit /b 0

:fail
echo Dogrulama basarisiz. Yukaridaki hata metnini inceleyin.
pause
exit /b 1


@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.13 -m venv .venv 2>nul
    if errorlevel 1 python -m venv .venv
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo Kurulum tamamlanamadi.
    pause
    exit /b 1
)

echo Kurulum tamamlandi.
echo Secenekleri gormek icin: .venv\Scripts\python.exe train.py --help
pause
endlocal


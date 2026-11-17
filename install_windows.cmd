@echo off
setlocal
cd /d "%~dp0"
py -3.12 -c "import struct,tkinter; assert struct.calcsize('P') == 8" >nul 2>&1
if errorlevel 1 (
  echo Install Python 3.12 x64 with Tcl/Tk and Python Launcher first.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-models.txt -r requirements-geometry.txt
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" tools\prepare_native_python.py
if errorlevel 1 exit /b 1
if exist "vendor\camera_win64\AngstrongCameraSdk.dll" (
  ".venv\Scripts\python.exe" tools\verify_installation.py --windows-sdk
) else (
  ".venv\Scripts\python.exe" -c "import main; print('Source UI imports OK. Provision camera/native assets before capture.')"
)
if errorlevel 1 exit /b 1
echo Ready. Use start_main.cmd or start_beta.cmd.
pause

@echo off
setlocal
cd /d "%~dp0.."
set "APP_PYTHON="

rem An explicit interpreter takes priority; never silently ignore a bad override.
if defined RECONST_PYTHON (
  call :probe "%RECONST_PYTHON%"
  if errorlevel 1 goto invalid_override
  goto launch
)

rem Prefer the installed project environment, then an existing host environment.
call :probe "%CD%\.venv\Scripts\python.exe"
if defined APP_PYTHON goto launch
for /f "delims=" %%P in ('where.exe python.exe 2^>nul') do if not defined APP_PYTHON call :probe "%%P"
if defined APP_PYTHON goto launch
rem Python Launcher may be installed even when Python itself is absent from PATH.
for /f "delims=" %%P in ('py -c "import sys; print(sys.executable)" 2^>nul') do if not defined APP_PYTHON call :probe "%%P"
if defined APP_PYTHON goto launch

echo No usable 64-bit Python environment was found.
echo Required: Python 3.10 or newer, Tcl/Tk, NumPy, SciPy, OpenCV and Pillow.
echo Run install_windows.cmd to create the project environment,
echo or set RECONST_PYTHON to an existing configured python.exe.
exit /b 1

:invalid_override
echo RECONST_PYTHON is not a usable project interpreter: "%RECONST_PYTHON%"
echo Check its path and project dependencies, or clear RECONST_PYTHON for automatic detection.
exit /b 1

:launch
echo Using Python: "%APP_PYTHON%"
"%APP_PYTHON%" main.py %*
exit /b %errorlevel%

:probe
if not exist "%~1" exit /b 1
"%~1" -c "import sys,struct,tkinter,numpy,scipy,cv2,PIL; assert sys.version_info >= (3,10) and struct.calcsize('P') == 8; import main" >nul 2>&1
if errorlevel 1 exit /b 1
set "APP_PYTHON=%~f1"
exit /b 0

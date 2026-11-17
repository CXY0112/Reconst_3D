@echo off
setlocal
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "RTAB_VS_ROOT=%%i"
if not defined RTAB_VS_ROOT exit /b 2
call "%RTAB_VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
cl /nologo /EHsc /std:c++17 /utf-8 /O2 /MD "%~dp0rtab_host.cpp" /Fo:"%~dp0..\vendor\online_geometry\rtab_host.obj" /Fe:"%~dp0..\vendor\online_geometry\rtab_host.exe"
exit /b %errorlevel%

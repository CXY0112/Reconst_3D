@echo off
setlocal
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "BUFFERX_VS_ROOT=%%i"
if not defined BUFFERX_VS_ROOT exit /b 2
call "%BUFFERX_VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
set "BUFFERX_BUILD=%~dp0..\..\experiments\isam2_build"
set "BUFFERX_OUT=%~dp0..\vendor\online_bufferx"
if not exist "%BUFFERX_OUT%" mkdir "%BUFFERX_OUT%"
cl /nologo /EHsc /std:c++17 /utf-8 /O2 /MD /LD /DNOMINMAX /D_USE_MATH_DEFINES /I"%BUFFERX_BUILD%\include" /I"%~dp0..\vendor\nksr_windows\env\Library\include\eigen3" "%~dp0online_bridge.cpp" /Fo:"%BUFFERX_OUT%\online_bridge.obj" /Fe:"%BUFFERX_OUT%\online_bridge.dll" /link "%BUFFERX_BUILD%\rtabmap_core.lib" "%BUFFERX_BUILD%\rtabmap_utilite.lib" "%BUFFERX_BUILD%\opencv_core4.lib"
if errorlevel 1 exit /b 1
cl /nologo /EHsc /std:c++17 /utf-8 /O2 /MD "%~dp0rtab_host.cpp" /Fo:"%BUFFERX_OUT%\rtab_host.obj" /Fe:"%BUFFERX_OUT%\rtab_host.exe"
if errorlevel 1 exit /b 1
copy /y "%~dp0..\vendor\online_geometry\msvcr110.dll" "%BUFFERX_OUT%\" >nul
copy /y "%~dp0..\vendor\online_geometry\msvcp110.dll" "%BUFFERX_OUT%\" >nul
exit /b %errorlevel%

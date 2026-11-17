@echo off
setlocal
set "GCVO_BUILD=%~dp0..\vendor\gcvo\build_cpu"
cmake -S "%~dp0..\gcvo_native" -B "%GCVO_BUILD%" -DEIGEN3_INCLUDE_DIR="%~dp0..\vendor\nksr_windows\env\Library\include\eigen3" -DRECONST_GCVO_CUDA=OFF
if errorlevel 1 exit /b 1
cmake --build "%GCVO_BUILD%" --config Release --parallel 4
if errorlevel 1 exit /b 1
if not exist "%~dp0..\vendor\gcvo\bin" mkdir "%~dp0..\vendor\gcvo\bin"
copy /y "%GCVO_BUILD%\Release\gcvo_cpu.dll" "%~dp0..\vendor\gcvo\bin\gcvo_cpu.dll" >nul
exit /b %errorlevel%

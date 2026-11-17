@echo off
setlocal
call "%~dp0tools\run_windows.cmd" beta %*
set "launch_result=%errorlevel%"
if not "%launch_result%"=="0" pause
exit /b %launch_result%

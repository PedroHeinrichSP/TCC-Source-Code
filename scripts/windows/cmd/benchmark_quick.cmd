@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\powershell\benchmark_quick.ps1" %*
exit /b %ERRORLEVEL%

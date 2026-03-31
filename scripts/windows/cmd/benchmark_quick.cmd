@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0..\powershell\benchmark_quick.ps1" %*
exit /b %ERRORLEVEL%

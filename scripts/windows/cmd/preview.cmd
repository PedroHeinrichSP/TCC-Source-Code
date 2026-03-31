@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\powershell\preview.ps1" %*
exit /b %ERRORLEVEL%

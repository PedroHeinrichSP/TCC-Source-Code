@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\powershell\setup.ps1" %*
exit /b %ERRORLEVEL%

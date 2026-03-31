@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0..\powershell\preview.ps1" %*
exit /b %ERRORLEVEL%

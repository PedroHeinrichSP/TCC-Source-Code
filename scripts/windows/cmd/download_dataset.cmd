@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0..\powershell\download_dataset.ps1" %*
exit /b %ERRORLEVEL%

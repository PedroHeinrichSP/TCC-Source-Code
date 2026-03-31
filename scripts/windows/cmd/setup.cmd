@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0..\powershell\setup.ps1" %*
exit /b %ERRORLEVEL%

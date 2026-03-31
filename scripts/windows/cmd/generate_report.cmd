@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0..\powershell\generate_report.ps1" %*
exit /b %ERRORLEVEL%

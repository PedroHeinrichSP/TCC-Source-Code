@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\powershell\run_ui_preview.ps1" %*
exit /b %ERRORLEVEL%

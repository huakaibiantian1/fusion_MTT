@echo off
setlocal
cd /d "%~dp0"
"%USERPROFILE%\.conda\envs\minimind\python.exe" GUI.py
endlocal

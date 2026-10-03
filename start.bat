@echo off
rem EyeCam launcher. Double-click for Web Bluetooth mode, or pass flags, e.g.:
rem   start.bat --muse        (bridge a Muse through muselsl)
rem   start.bat --muse --lan  (also serve flicker.html to a phone)
cd /d "%~dp0"
python server.py %*
pause

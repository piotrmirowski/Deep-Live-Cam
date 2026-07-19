ECHO off
ECHO Run deep fake using Logitech webcam
call ..\gemini_api_key.bat
call venv\Scripts\python.exe run_deep_fake.py --width 720 --height 540 --execution-provider cuda --device "Logitech"

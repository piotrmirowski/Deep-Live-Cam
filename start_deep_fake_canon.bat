ECHO off
ECHO Run deep fake using Canon EOS Camera
call ..\gemini_api_key.bat
call venv\Scripts\python.exe run_deep_fake.py --width 1280 --height 720 --execution-provider cuda --device "EOS Webcam Utility"

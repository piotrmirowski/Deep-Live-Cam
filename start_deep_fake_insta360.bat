ECHO off
ECHO Run deep fake using Insta360 Link 2
call ..\gemini_api_key.bat
call venv\Scripts\python.exe run_deep_fake.py --width 1280 --height 720 --execution-provider cuda --device "Insta360 Link 2"

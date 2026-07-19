ECHO off
ECHO Run deep fake using laptop webcam
call ..\gemini_api_key.bat
call venv\Scripts\python.exe run_deep_fake.py --width 720 --height 540 --execution-provider cuda --device "Integrated Webcam"

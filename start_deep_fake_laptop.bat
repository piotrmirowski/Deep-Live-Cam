ECHO off
ECHO Run deep fake using laptop webcam
call ..\gemini_api_key.bat
call venv\Scripts\python.exe run_deep_fake.py --width 960 --height 600 --execution-provider cuda --device "Integrated Webcam"

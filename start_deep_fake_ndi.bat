ECHO off
ECHO Run deep fake using NDI Webcam Video 1
call ..\gemini_api_key.bat
call venv\Scripts\python.exe run_deep_fake.py --width 960 --height 540 --execution-provider cuda --device "NDI Webcam Video 1"

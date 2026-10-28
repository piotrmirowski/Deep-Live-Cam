#!/bin/sh

# Switch to venv
source venv/bin/activate

# Always replace last deep fake by Einstein
cp templates/einstein.jpg images/temp.jpg

# Start deep fake program
python3 run_deep_fake.py  \
--source images/temp.jpg \
--width 1280 \
--height 720 \
--execution-provider coreml \
--camera_index 1200



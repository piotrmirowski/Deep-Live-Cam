#!/bin/sh

# Always replace last deep fake by Einstein
cp templates/einstein.jpg images/temp.jpg

# Start deep fake program
python3 run_deep_fake.py  \
--source images/temp.jpg \
--width 960 \
--height 540 \
--execution-provider coreml \
--device 0



#!/bin/bash

# Check if the required Python script exists
if [ ! -f "launch_scientist_step1.py" ]; then
  echo "Error: launch_scientist_step1.py not found!"
  exit 1
fi

# Run the Python script with the desired arguments
python3 launch_scientist_step1.py \
  --experiment "nanoGPT" \
  --model "gemini-2.0-flash" \
  --writeup "latex" \
  --parallel 1 \
  --improvement \
  --gpus "0" \
  --num-ideas 5 \
  --engine "openalex"


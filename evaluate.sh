#!/bin/bash

# img path
if [ $# -eq 0 ]; then
    echo "Usage: ./evaluate.sh <image_path>"
    exit 1
fi

# parameters
MODEL_PATH="best_model1.pth"   # 确保这是你的模型文件路径
IMAGE_PATH="$1"
CLASS_NAMES="battery biological cardboard glass metal paper plastic textile"

# evaluate
python evaluate.py \
    --model_path "$MODEL_PATH" \
    --image_path "$IMAGE_PATH" \
    --class_names $CLASS_NAMES


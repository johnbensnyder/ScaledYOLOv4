#!/bin/bash

cd /opt/ml/code

pip install .

GPU_COUNT=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

torchrun \
    --nproc_per_node=${GPU_COUNT} \
    /opt/ml/code/train.py \
    --batch-size 8 \
    --data coco.yaml \
    --cfg yolov4-p5.yaml \
    --sync-bn True \
    --name yolov4-p5
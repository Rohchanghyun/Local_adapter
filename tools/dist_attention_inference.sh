#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
CONFIG=$1
CHECKPOINT=$2
GPU_ID=${3:-0}  # 기본값 0
PORT=${PORT:-29502}

# GPU ID를 환경변수로 설정
export CUDA_VISIBLE_DEVICES=$GPU_ID

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 $(dirname "$0")/attention_inference.py $CONFIG $CHECKPOINT --launcher none ${@:4}
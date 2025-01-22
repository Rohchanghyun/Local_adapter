#!/bin/bash

# GPU ID를 1로 고정
GPU_ID=1

# Python 스크립트 실행
./tools/dist_attention_inference.sh \
    configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v2_stemnet_coco_256x192_inference.py \
    ./checkpoints/sdpose_s_v2.pth \
    $GPU_ID
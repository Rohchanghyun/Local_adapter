#!/bin/bash

# NCCL 환경 변수 설정
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

# Python 스크립트 실행
./tools/dist_test.sh configs/body/2d_kpt_sview_rgb_img/sdpose/coco/sdpose_s_v2_stemnet_coco_256x192.py \ ./checkpoints/sdpose_s_v2.pth 2
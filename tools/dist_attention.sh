#!/usr/bin/env bash
# Copyright (c) OpenMMLab. All rights reserved.
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29502}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/attention.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

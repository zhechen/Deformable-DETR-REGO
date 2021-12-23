#!/usr/bin/env bash

set -x

EXP_DIR=exps/r50_deformable_detr_rego
PY_ARGS=${@:1}

python -u main.py \
    --use_rego \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}

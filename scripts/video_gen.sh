#!/usr/bin/env bash
trap 'kill 0' SIGINT


GPUS=(0 1 2 3)
START_EPS=0
EPS_PER_JOB=120
TOTAL_JOB_NUM=10
for ((i=0;i<$TOTAL_JOB_NUM;++i)); do
    GPU_IDX=$(( i % ${#GPUS[@]} ))
    GPU=${GPUS[GPU_IDX]}
    EPS_START=$((START_EPS + i * EPS_PER_JOB))
    EPS_END=$((EPS_START + EPS_PER_JOB))
    echo "GPU: $GPU, EPS_START: $EPS_START, EPS_END: $EPS_END"
    python projects/harobo/video_gen.py \
        --no_render --no_interactive \
        --split_dataset $EPS_START $EPS_END --gt_semantic --allow_sliding \
        --save_video --gpu $GPU --exp_name demo_video &
done


wait


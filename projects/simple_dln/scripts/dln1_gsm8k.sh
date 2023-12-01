#!/bin/bash
set -x  # print commands to terminal

dataset="gsm8k"
dir=log/dln1/${dataset}

for seed in 13 42 25; do
    python dln1.py \
        --config llm_config.yaml \
        --fwd_model gpt-3-fwd \
        --bwd_model gpt-3-bwd \
        --dataset ${dataset} \
        --output_scoring_function accuracy \
        --max_train_size 400 \
        --batch_size 20 \
        --iters 50 \
        --patience 2 \
        --num_samples 10 \
        --seed ${seed} \
        --out_dir ${dir}
done
#!/bin/bash
set -x  # print commands to terminal

dataset="gsm8k"
dir=log/dwln2/concat/${dataset}

for width in 2 5 10; do
    for seed in 13 42 25; do
        python dwln2.py \
            --config llm_config.yaml \
            --fwd_model gpt-3-fwd \
            --bwd_model gpt-3-bwd \
            --dataset ${dataset} \
            --output_scoring_function accuracy \
            --max_train_size 400 \
            --batch_size 20 \
            --iters 50 \
            --patience 2 \
            --num_samples 20 \
            --aggregation concat \
            --seed ${seed} \
            --out_dir ${dir}/width${width} \
            --width ${width}
    done
done
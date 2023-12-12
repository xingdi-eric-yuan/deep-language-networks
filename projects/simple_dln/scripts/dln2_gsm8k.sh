#!/bin/bash
set -x  # print commands to terminal

dataset="gsm8k"
dir=log/dec12/dln2/${dataset}
prompt_backward_template="ln_prompt_backward:1.0"  # "ln_prompt_backward:2.0"
input_backward_template="ln_input_backward:1.0"  # "ln_input_backward:2.0"

# for num_samples in 10 50; do
for num_samples in 10; do
    for normalize_score in True; do
        for first_layer_contrastive in False True; do
            for score_input_phx in False True; do
                for seed in 13 42 25; do
                    python dln2.py \
                        --config llm_config.yaml \
                        --fwd_model gpt-3-fwd \
                        --bwd_model gpt-3-bwd \
                        --dataset ${dataset} \
                        --max_train_size 400 \
                        --batch_size 10 \
                        --iters 10 \
                        --patience 2 \
                        --num_samples ${num_samples} \
                        --seed ${seed} \
                        --out_dir ${dir}/${prompt_backward_template}_${input_backward_template}_contrastive${first_layer_contrastive}_phx${score_input_phx}_sample${num_samples}_norm${normalize_score} \
                        --prompt_backward_template ${prompt_backward_template} \
                        --input_backward_template ${input_backward_template} \
                        --first_layer_contrastive ${first_layer_contrastive} \
                        --score_input_phx ${score_input_phx} \
                        --normalize_score ${normalize_score}
                done
            done
        done
    done
done

#!/bin/bash
set -x  # print commands to terminal

dataset="gsm8k"
dir=log/dec18/dln2/${dataset}
prompt_backward_template="ln_prompt_backward:2.0"  # "ln_prompt_backward:2.0"
input_backward_template="ln_input_backward:1.0"  # "ln_input_backward:2.0"
first_layer_contrastive=True
num_samples=10
normalize_score=True
score_input_phx=True
skip_good_h=True

for diverse_h_sample in True; do
    for normalize_by_length in False True; do
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
                --out_dir ${dir}/${prompt_backward_template}_${input_backward_template}_contrastive${first_layer_contrastive}_phx${score_input_phx}_sample${num_samples}_norm${normalize_score}_lennorm${normalize_by_length}_skiph${skip_good_h}_divh${diverse_h_sample} \
                --prompt_backward_template ${prompt_backward_template} \
                --input_backward_template ${input_backward_template} \
                --first_layer_contrastive ${first_layer_contrastive} \
                --score_input_phx ${score_input_phx} \
                --normalize_score ${normalize_score} \
                --normalize_by_length ${normalize_by_length} \
                --skip_good_h ${skip_good_h} \
                --diverse_h_sample ${diverse_h_sample}
        done
    done
done
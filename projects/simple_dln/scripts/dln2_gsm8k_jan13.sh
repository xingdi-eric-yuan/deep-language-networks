#!/bin/bash
set -x  # print commands to terminal

dataset="gsm8k"
dir=log/jan15/dln2/${dataset}
prompt_backward_template="ln_prompt_backward:1.0"  # "ln_prompt_backward:2.0"
input_backward_template="ln_input_backward:1.0"  # "ln_input_backward:2.0"
first_layer_contrastive=True
num_samples=10
normalize_score=True
score_input_phx=True
skip_good_h=True
normalize_by_length=True
two_step_h_sample=True
two_step_pi_sample=True

for residual in False True; do
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
            --out_dir ${dir}/${prompt_backward_template}_${input_backward_template}_contrastive${first_layer_contrastive}_phx${score_input_phx}_sample${num_samples}_norm${normalize_score}_lennorm${normalize_by_length}_skiph${skip_good_h}_divh${two_step_h_sample}_divpi${two_step_pi_sample}_res${residual} \
            --prompt_backward_template ${prompt_backward_template} \
            --input_backward_template ${input_backward_template} \
            --first_layer_contrastive ${first_layer_contrastive} \
            --score_input_phx ${score_input_phx} \
            --normalize_score ${normalize_score} \
            --normalize_by_length ${normalize_by_length} \
            --skip_good_h ${skip_good_h} \
            --two_step_h_sample ${two_step_h_sample} \
            --two_step_pi_sample ${two_step_pi_sample} \
            --residual ${residual}
    done
done
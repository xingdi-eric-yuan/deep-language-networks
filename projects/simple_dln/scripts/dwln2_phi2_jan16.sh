#!/bin/bash
set -x  # print commands to terminal

dataset="gsm8k"
# dataset="hyperbaton"
# dataset="navigate"
# dataset="date_understanding"
# dataset="logical_deduction_seven_objects"
# dataset="mpqa"
# dataset="trec"
# dataset="subj

dir=log/jan16-phi2/dwln2/${dataset}
prompt_backward_template="ln_prompt_backward:1.0"  # "ln_prompt_backward:2.0",  "ln_prompt_backward:3.0"
input_backward_template="ln_input_backward:1.0"  # "ln_input_backward:2.0"
first_layer_contrastive=True
num_samples=10
normalize_score=False
score_input_phx=False
skip_good_h=True
normalize_by_length=True
two_step_h_sample=False
two_step_pi_sample=False
residual=False

for agg in concat summary; do
    for width in 2 5; do
        for seed in 13 42 25; do
            python dwln2.py \
                --config phi2_config.yaml \
                --fwd_model gpt-3-fwd \
                --bwd_model gpt-3-bwd \
                --dataset ${dataset} \
                --max_train_size 400 \
                --batch_size 5 \
                --iters 10 \
                --patience 2 \
                --num_samples ${num_samples} \
                --aggregation ${agg} \
                --seed ${seed} \
                --out_dir ${dir}/${prompt_backward_template}_${input_backward_template}_contrastive${first_layer_contrastive}_phx${score_input_phx}_sample${num_samples}_norm${normalize_score}_lennorm${normalize_by_length}_skiph${skip_good_h}_divh${two_step_h_sample}_divpi${two_step_pi_sample}_res${residual}_width${width}_agg${agg} \
                --width ${width} \
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
done
#!/bin/bash

ROOT_PATH="/opt/ml/input"
TRAINING_SETS=("path/to/local/json/train_data1.json" "path/to/local/json/train_data2.json")
for data in "${TRAINING_SETS[@]}";
do
    TRAINING_DATA="${TRAINING_DATA:+${TRAINING_DATA} }${ROOT_PATH}/${data}"
done

export WANDB_API_KEY="your wandb api key"

nohup deepspeed --exclude="localhost:0,7" --master_port 29501 train_ppo.py \
    --actor_model_name_or_path "actor_model_name_or_path" \
    --critic_model_name_or_path "critic_model_name_or_path" \
    --stop_words "<|im_start|>" "<|im_end|>"\
    --transformer_name_in_causal_lm "transformer" \
    --data_path ${TRAINING_DATA} \
    --output_dir "output_dir" \
    --num_train_epochs 1 \
    --ppo_epochs 1 \
    --generation_batches 1 \
    --per_device_generation_batch_size 2 \
    --per_device_training_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_steps 300 \
    --save_total_limit 3 \
    --actor_learning_rate 1e-5 \
    --critic_learning_rate 5e-6 \
    --seed 2023 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "constant_with_warmup" \
    --dtype "fp16" \
    --max_prompt_seq_len 64 \
    --max_answer_seq_len 64 \
    --actor_gradient_checkpointing \
    --critic_gradient_checkpointing \
    --actor_zero_stage 2 \
    --critic_zero_stage 3 \
    --lazy_preprocess \
    --offload \
    --align_overflow \
    --offload_reference_model \
    --deepspeed \
    --print_answers \
    --report_to "wandb" \
    --report_name "wandb-group-name" \
    > logs/train_ppo_log.log 2>&1 &

    # --enable_test_mode \
    # --enable_hybrid_engine \

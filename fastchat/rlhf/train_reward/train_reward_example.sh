#!/bin/bash

ROOT_PATH="/opt/ml/input"
TRAINING_SETS=("path/to/local/json/train_data1.json" "path/to/local/json/train_data2.json")
for data in "${TRAINING_SETS[@]}";
do
    TRAINING_DATA="${TRAINING_DATA:+${TRAINING_DATA} }${ROOT_PATH}/${data}"
done

EVAL_SETS=("UltraFeedback/all_feedback_convs_eval.json")
for data in "${EVAL_SETS[@]}";
do
    EVAL_DATA="${EVAL_DATA:+${EVAL_DATA} }${ROOT_PATH}/${data}"
done

export WANDB_API_KEY="your wandb api key"

nohup deepspeed --include="localhost:2,3,4,5" --master_port 29501 fastchat/rlhf/train_reward/train_rm.py \
    --model_name_or_path "/opt/ml/model/path-to-base-model" \
    --transformer_name_in_causal_lm "transformer" \
    --data_path ${TRAINING_DATA} \
    --eval_data_path ${EVAL_DATA} \
    --output_dir "/opt/ml/model/path-to-oupt" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_steps 300 \
    --save_total_limit 3 \
    --eval_steps 100 \
    --learning_rate 9e-6 \
    --weight_decay 0. \
    --seed 2023 \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "constant_with_warmup" \
    --dtype "fp16" \
    --max_seq_len 2048 \
    --gradient_checkpointing \
    --zero_stage 3 \
    --lazy_preprocess \
    --offload \
    --deepspeed \
    --report_to "wandb" \
    --report_name "wandb group name" \
    > fastchat/rlhf/logs/train_log.log 2>&1 &

    
    
    # --eval_steps 50 \
#     --model_name_or_path "/opt/ml/model/Llama-2-13b-hf" \
# ./s5cmd sync /tmp/llama_out s3://sagemaker-spatio-models/models_output/llama_13b/$(date +%Y-%m-%d-%H-%M-%S)/


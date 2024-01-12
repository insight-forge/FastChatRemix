#!/bin/bash

# train chat
TRAINING_DATA="{TRAINING_DATA}"
TEST_DATA="{TEST_DATA}"

#cd FastChat && pip install -e . && cd ..
# pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install flash-attn -i https://pypi.tuna.tsinghua.edu.cn/simple

# chmod +x ./s5cmd
# ./s5cmd sync s3://sagemaker-spatio-models/models/llama_13b/* /tmp/llama_pretrain/
# ./s5cmd cp s3://sagemaker-spatio-models/datasets/${TRAINING_DATA} /tmp/datasets/${TRAINING_DATA}

export WANDB_PROJECT="{WANDB_PROJECT}"
export WANDB_API_KEY="{WANDB_API_KEY}"

deepspeed --include="localhost:0,1,2,3,4,5,6,7" fastchat/train/train.py \
    --deepspeed ds_config_zero3_auto.json \
    --model_name_or_path "/opt/ml/model/Yi-6B/" \
    --data_path ${TRAINING_DATA} \
    --eval_data_path ${TEST_DATA} \
    --output_dir "/opt/ml/model/Yi-6B-sft" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size  1 \
    --gradient_accumulation_steps 128 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --cache_dir "/tmp" \
    --fp16_full_eval \
    --fp16 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --run_name "Yi-6B-sft" \
    --report_to "wandb"
    #--report_to "none"

# ./s5cmd sync /tmp/llama_out s3://sagemaker-spatio-models/models_output/llama_13b/$(date +%Y-%m-%d-%H-%M-%S)/

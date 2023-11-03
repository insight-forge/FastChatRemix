#!/usr/bin/env python

import argparse
import os
import math
import sys
import json
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers import (
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from fastchat.rlhf.utils.model.model_utils import create_critic_model
from fastchat.rlhf.utils.data.data_utils import DataCollatorReward
from fastchat.rlhf.utils.utils import print_rank_0, to_device, save_rm_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from fastchat.rlhf.utils.ds_utils import get_train_ds_config
from fastchat.rlhf.utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from fastchat.model.model_adapter import get_conversation_template



def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--eval_data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the evaluation dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument(
        '--lazy_preprocess',
        action='store_true',
        help='Enable lazy preprocess')
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--resume_ckpt_path",
        default=None,
        type=str,
        help=
        "Reward param checkpoint path.",
    )
    parser.add_argument(
        "--transformer_name_in_causal_lm",
        type=str,
        choices=["transformer", "model"],
        help=
        "The transformer variable name in causal_lm. If the model can not load by transformers.AutoModel, please specify this argument",
        default=None,
    )
    parser.add_argument(
        "--use_flash_attn",
        action='store_true',
        help="user flash attention",
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=0,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--save_steps",
                        type=int,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--save_total_limit",
                        type=int,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--eval_steps",
                        type=int,
                        default=0,
                        help="default(0) means to evaluate per epoch, otherwise, evaluate per eval_steps.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Where to store the model.")
    parser.add_argument("--cache_dir",
                        type=str,
                        default='/tmp',
                        help="Where to cache the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype', type=str, default='fp16',
                        choices=['fp16', 'bf16'],
                        help = 'Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## logging
    parser.add_argument('--report_to',
                        type=str,
                        choices=['tensorboard', 'wandb'],
                        help='Enable logging',
                        default=None)
    ## wandb
    parser.add_argument('--report_project',
                        type=str,
                        default='deepspeed',
                        help='Report tensorboard path or project of wandb')
    parser.add_argument('--report_name',
                        type=str,
                        default='default_run_name',
                        help='Report tensorboard file name or group name of wandb')


    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str = "vicuna"
):
    conv = get_conversation_template(model_path)
    roles = {
        "human": conv.roles[0],
        "user": conv.roles[0],
        "gpt": conv.roles[1],
        "assistant": conv.roles[1],
    }

    chosen, rejects = [], []
    for source in sources:
        conv.messages = []
        for i, conversations in enumerate(source["conversations"]):
            conv.append_message(roles[conversations["from"].lower()], conversations["value"])
        conv.append_message(conv.roles[1], source["chosen"]["value"])
        chosen.append(conv.get_prompt())

        conv.update_last_message(source["rejected"]["value"])
        rejects.append(conv.get_prompt())

    chosen = tokenizer(
        chosen,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    rejects = tokenizer(
        rejects,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )

    return chosen["input_ids"], chosen["attention_mask"], \
           rejects["input_ids"], rejects["attention_mask"],

class LazyRewardDataset(Dataset):
    """Dataset for training reward model."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_name):
        super(LazyRewardDataset, self).__init__()
        self.tokenizer = tokenizer
        self.model_path = model_name
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer, self.model_path)
        ret = [_[0:1] for _ in ret]
        self.cached_data_dict[i] = ret

        return ret

class RewardDataset(Dataset):
    """Dataset for training reward model."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_name):
        super(RewardDataset, self).__init__()
        self.tokenizer = tokenizer

        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.data = preprocess(self.raw_data, self.tokenizer, model_name)


    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return [_[i:i+1] for _ in self.data]



def make_reward_dataset(
        tokenizer: transformers.PreTrainedTokenizer,
        args
):
    dataset_cls = (
        LazyRewardDataset if args.lazy_preprocess else RewardDataset
    )
    print_rank_0("Loading data...", args.global_rank)
    # json format :
    # [{
    # "id":"1",
    #  "conversations": [{"from":"", "value":""},...],
    #  "chosen": {"from":"", "value":""},
    #  "rejected": {"from":"", "value":""}
    #  }]
    train_json = []
    for path in args.data_path:
        train_json.extend(json.load(open(path, "r")))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, model_name=args.model_name_or_path)

    if args.eval_data_path:
        eval_json = []
        for path in args.eval_data_path:
            eval_json.extend(json.load(open(path, "r")))
        # eval_json = json.load(open(args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, model_name=args.model_name_or_path)
    else:
        eval_dataset = None

    return train_dataset, eval_dataset

def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    report_to=args.report_to,
                                    report_project=args.report_project,
                                    report_name=args.report_name
                                    )
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    args.global_rank = torch.distributed.get_rank()
    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.max_seq_len,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.eos_token_id = tokenizer.eod_id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    rm_model = create_critic_model(args.model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   args.num_padding_at_beginning,
                                   disable_dropout=args.disable_dropout,
                                   trust_remote_code=True,
                                   transformer_name_in_causal_lm=args.transformer_name_in_causal_lm,
                                   use_flash_attn=args.use_flash_attn,
                                   resume_ckpt_path=args.resume_ckpt_path,
                                   zero_stage=args.zero_stage
                                   )
    print_rank_0(f"{args.model_name_or_path} config:", args.global_rank)
    print_rank_0(rm_model.config, args.global_rank)

    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model,
                                                args.lora_module_name,
                                                args.lora_dim)
        if args.only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)
            rm_model = make_model_gradient_checkpointing_compatible(rm_model)

    train_dataset, eval_dataset = make_reward_dataset(tokenizer=tokenizer, args=args)

    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    # eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        r_scores = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            scores += outputs["chosen_mean_scores"].mean().float()
            r_scores += outputs["rejected_mean_scores"].mean().float()
            # if step == 99:  # For faster evaluation and debugging
            #     break
        acc = correct_predictions / total_predictions
        scores = scores / (step + 1)
        r_scores = r_scores / (step + 1)
        try:
            acc = get_all_reduce_mean(acc).item()
            scores = get_all_reduce_mean(scores).item()
            r_scores = get_all_reduce_mean(r_scores).item()
        except:
            pass

        if model.monitor.enabled and model.global_rank == 0:
            print_rank_0("Writing evaluation results...")
            summary_events = [
                (f"Eval/Samples/acc", acc, model.global_samples),
                (f"Eval/Samples/reward_score", scores, model.global_samples),
                (f"Eval/Samples/rejected_reward_score", r_scores, model.global_samples),
            ]
            model.monitor.write_events(summary_events)
        return scores, acc, r_scores

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    num_warmup_steps = args.num_train_epochs * num_update_steps_per_epoch * args.warmup_ratio
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    print_rank_0(f"***** Evaluating init reward *****", args.global_rank)
    reward_score, acc, r_reward_score = evaluation_reward(rm_model, eval_dataloader)
    print_rank_0(
        f"Init chosen_last_scores (higher is better) : {reward_score}, rejected_last_scores (lower is better): {r_reward_score},  acc (higher is better) : {acc}",
        args.global_rank)
    args.global_step = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            args.global_step += 1
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]

            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()

            if args.save_steps and args.global_step % (args.save_steps * args.gradient_accumulation_steps) == 0:
                save_rm_hf_format(rm_model, tokenizer, args)
            if args.eval_steps > 0 and args.global_step % (args.eval_steps * args.gradient_accumulation_steps) == 0:
                reward_score, acc, r_reward_score = evaluation_reward(rm_model, eval_dataloader)
                rm_model.train()
                print_rank_0(
                    f"step = {args.global_step // args.gradient_accumulation_steps} "
                    f"chosen_last_scores (higher is better): {reward_score}, "
                    f"rejected_last_scores (lower is better): {r_reward_score}, "
                    f"acc (higher is better) : {acc}",
                    args.global_rank)

        print_rank_0(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}",
            args.global_rank)

        if args.eval_steps == 0:
            # Evaluate reward_loss on the validation set.
            print_rank_0(
                f"***** Evaluating reward, Epoch {epoch + 1}/{args.num_train_epochs} *****",
                args.global_rank)
            reward_score, acc, r_reward_score = evaluation_reward(rm_model, eval_dataloader)
            print_rank_0(f"Epoch {epoch+1}/{args.num_train_epochs}, chosen_last_scores (higher is better): {reward_score}, "
                    f"rejected_last_scores (lower is better): {r_reward_score}, "
                    f"acc (higher is better) : {acc}", args.global_rank)
        rm_model.tput_timer.update_epoch_count()

    save_rm_hf_format(rm_model, tokenizer, args, final=True)



if __name__ == "__main__":
    main()

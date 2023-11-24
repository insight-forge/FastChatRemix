#!/usr/bin/env python

import argparse
import os
import math
import sys
import random
import time
import json
from typing import Dict
from itertools import chain

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers import (
    SchedulerType,
    default_data_collator
)

import deepspeed
from deepspeed.accelerator import get_accelerator

from fastchat.rlhf.utils.data.data_utils import DataCollatorRLHF, MiniDataset
from fastchat.rlhf.utils.utils import print_rank_0, to_device, set_random_seed, get_all_reduce_mean, moving_average, save_ppo_model_hf_format
from fastchat.rlhf.utils.perf import print_throughput_step3
from fastchat.model.model_adapter import get_conversation_template
from rlhf_engine import DeepSpeedRLHFEngine
from ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from datasets import load_dataset
from transformers.models.llama import LlamaForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        required=True,
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument('--eval_data_path',
                        nargs='*',
                        default=None,
                        help='Path to the evaluation dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument("--unsup_coef",
                        type=float,
                        default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        '--lazy_preprocess',
        action='store_true',
        help='Enable lazy preprocess')
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--critic_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--transformer_name_in_causal_lm",
        type=str,
        choices=["transformer", "model"],
        help=
        "The transformer variable name in causal_lm for load reward model. If the model can not load by transformers.AutoModel, please specify this argument",
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
        "--per_device_generation_batch_size",
        type=int,
        default=16,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
    parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=16,
        help=
        "Mini Batch size (per device) for the MiniDataset."
    )
    parser.add_argument("--generation_batches",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
    parser.add_argument(
        "--ppo_epochs",
        type=int,
        default=1,
        help="For generated data, how many ppo training epochs to run.")
    parser.add_argument("--max_prompt_seq_len",
                        type=int,
                        default=1024,
                        help="The maximum sequence length.")
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=1024,
                        help="The maximum sequence length.")
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use."
    )
    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--critic_weight_decay",
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
        "--actor_dropout",
        type=float,
        default=None,
        help="If actor dropout configured, use it. "
             "Otherwise, keep the default dropout configuration of the actor model."
    )
    parser.add_argument('--critic_dropout',
                        action='store_true',
                        help='Disable the dropout of the critical model.')
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        '--critic_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Critic model.')
    parser.add_argument('--stop_words',
                        nargs='*',
                        default=None,
                        help='Path to the training dataset. Accepted format:'
                             '1) a single data path, 2) multiple datasets in the'
                             'form: dataset1-path dataset2-path ...')
    ## Actor/critic model overflow alignment
    parser.add_argument(
        '--align_overflow',
        action='store_true',
        help='Align loss scale overflow between actor and critic')
    # deepspeed features
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for the inference-optimization. Please note hybrid-engine need to be enabled when using this feature."
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation. This makes generation slower but requires less memory."
    )
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference. This makes generation preparation slower but might increase e2e throughput by using larger batch size."
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine. Please note hybrid-engine and tp_inference_size > 1 need to be true when using this feature."
    )
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype', type=str, default='fp16',
                        choices=['fp16', 'bf16'],
                        help = 'Training data type')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    parser.add_argument(
        '--critic_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Critic model (and reward).')
    parser.add_argument(
        '--ref_zero_stage',
        type=int,
        default=None,
        help='ZeRO optimization stage for reference model, default eq actor_zero_stage')
    ## Make EMA as an optional feature
    parser.add_argument('--enable_ema',
                        action='store_true',
                        help='Enable EMA checkpoint for the model.')
    ## LoRA for efficient training setting
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument("--critic_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--critic_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')

    parser.add_argument(
        "--actor_lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial actor LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--critic_lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial critic LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )
    ## Mixed Precision ZeRO++
    parser.add_argument(
        '--enable_mixed_precision_lora',
        action='store_true',
        help='Enable Mixed Precision ZeRO++ for training and generation.')
    ## logging
    parser.add_argument('--report_to',
                        type=str,
                        choices=['tensorboard', 'wandb'],
                        help='Enable logging',
                        default=None)
    ## wandb
    parser.add_argument('--wandb_api_key',
                        type=str,
                        default=None,
                        help='wandb api key')
    parser.add_argument('--report_project',
                        type=str,
                        default='deepspeed',
                        help='Report tensorboard root path or project name of wandb')
    parser.add_argument('--report_name',
                        type=str,
                        default='step3_tensorboard_logs',
                        help='Report tensorboard dir name or group name of wandb')
    ## debug
    parser.add_argument('--print_answers',
                        action='store_true',
                        help='Print prompt and answers during training')
    parser.add_argument(
        "--print_answers_interval",
        type=int,
        default=10,
        help="If --print_answers enabled, controls the printing interval.")
    ## Testing
    parser.add_argument(
        '--enable_test_mode',
        action='store_true',
        help=
        'Enable a testing mode that terminates training based on args.test_stop_step'
    )
    parser.add_argument(
        "--test_stop_step",
        type=int,
        default=0,
        help=
        "Training non-overflow step at which to terminate training during testing."
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.inference_tp_size > 1:
        assert (
                args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    if args.actor_zero_stage == 2 and args.critic_zero_stage == 2 and args.enable_hybrid_engine and args.offload and args.actor_lora_dim == 0:
        raise ValueError(
            "The combination of [actor_zero_stage==2, critic_zero_stage==2, enable_hybrid_engine=True, offload=True, lora=False] is currently unsupported due to training instability!"
        )

    # wandb settings
    if args.wandb_api_key is not None:
        os.environ['WANDB_API_KEY'] = args.wandb_api_key
    return args

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str = "vicuna",
    max_prompt_seq_len: int = 1024,
):
    conv = get_conversation_template(model_path)
    roles = {
        "human": conv.roles[0],
        "user": conv.roles[0],
        "gpt": conv.roles[1],
        "assistant": conv.roles[1],
    }

    prompt_dataset = []
    for source in sources:
        conv.messages = []
        for i, conversations in enumerate(source["conversations"]):
            conv.append_message(roles[conversations["from"].lower()], conversations["value"])
        conv.append_message(conv.roles[1], None)
        prompt_token = tokenizer(
            conv.get_prompt(),
            return_tensors="pt",
            padding="max_length",
            max_length=max_prompt_seq_len,
            truncation=True,
        )
        prompt_dataset.append([prompt_token["input_ids"].squeeze(0),
                               prompt_token["attention_mask"].squeeze(0)])

    return prompt_dataset


class LazyPromptDataset(Dataset):
    """Dataset for training reward model."""
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_name, max_prompt_seq_len):
        super(LazyPromptDataset, self).__init__()
        self.tokenizer = tokenizer
        self.model_path = model_name
        self.tokenizer = tokenizer
        self.max_prompt_seq_len = max_prompt_seq_len
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer, self.model_path, self.max_prompt_seq_len)[0]
        # ret = [_[0:1] for _ in ret]
        self.cached_data_dict[i] = ret
        return ret

class PromptDataset(Dataset):
    """Dataset for training reward model."""
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_name, max_prompt_seq_len):
        super(PromptDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.data = preprocess(self.raw_data, tokenizer, model_name, max_prompt_seq_len)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.data[i]


def make_prompt_dataloader(
        args
):
    dataset_cls = (
        LazyPromptDataset if args.lazy_preprocess else PromptDataset
    )
    print_rank_0("Loading data...", args.global_rank)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.actor_model_name_or_path,
        cache_dir=args.cache_dir,
        # model_max_length=args.max_prompt_seq_len,
        padding_side="left", # prompt left padding for batch generate exp
        truncation_side='left',
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.eos_token_id = tokenizer.eod_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
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
    prompt_train_dataset = dataset_cls(
        train_json,
        tokenizer=tokenizer,
        model_name=args.actor_model_name_or_path,
        max_prompt_seq_len=args.max_prompt_seq_len)

    if args.eval_data_path:
        eval_json = []
        for path in args.eval_data_path:
            eval_json.extend(json.load(open(path, "r")))
        # eval_json = json.load(open(args.eval_data_path, "r"))
        prompt_eval_dataset = dataset_cls(
            eval_json,
            tokenizer=tokenizer,
            model_name=args.actor_model_name_or_path,
            max_prompt_seq_len=args.max_prompt_seq_len)
    else:
        prompt_eval_dataset = None

    prompt_data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                           args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if args.eval_data_path:
            prompt_eval_sampler = RandomSampler(prompt_eval_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if args.eval_data_path:
            prompt_eval_sampler = DistributedSampler(prompt_eval_dataset)

    prompt_train_dataloader = DataLoader(prompt_train_dataset,
                                         collate_fn=prompt_data_collator,
                                         sampler=prompt_train_sampler,
                                         batch_size=args.per_device_generation_batch_size)
    prompt_eval_dataloader = None
    if args.eval_data_path:
        prompt_eval_dataloader = DataLoader(prompt_eval_dataset,
                                             collate_fn=prompt_data_collator,
                                             sampler=prompt_eval_sampler,
                                             batch_size=args.per_device_generation_batch_size)


    return prompt_train_dataloader, prompt_eval_dataloader

def make_unsupervised_dataloader(args, tokenizer):
    unsupervised_raw_datasets = load_dataset(
        args.unsupervised_dataset_name, args.unsupervised_dataset_config_name)
    column_names = unsupervised_raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = unsupervised_raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    block_size = args.max_prompt_seq_len + args.max_answer_seq_len

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k]))
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k:
            [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    unsupervised_train_dataset = lm_datasets["train"]

    if args.local_rank == -1:
        unsupervised_train_sampler = RandomSampler(unsupervised_train_dataset)
    else:
        unsupervised_train_sampler = DistributedSampler(unsupervised_train_dataset)

    unsupervised_train_dataloader = DataLoader(
        unsupervised_train_dataset,
        collate_fn=default_data_collator,
        sampler=unsupervised_train_sampler,
        batch_size=args.per_device_generation_batch_size)

    return unsupervised_train_dataloader

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

    args.global_rank = torch.distributed.get_rank()

    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.actor_model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    tokenizer.eos_token_id = tokenizer.eod_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print_rank_0("***** tokenizer ******", args.global_rank)
    print_rank_0(tokenizer, args.global_rank)

    # DataLoaders creation:
    prompt_train_dataloader, _ = make_prompt_dataloader(args=args)
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = make_unsupervised_dataloader(args, tokenizer)
    else:
        unsupervised_train_dataloader = [None] * len(prompt_train_dataloader)  # basically a dummy dataloader

    min_dataloader_size = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))
    num_update_steps_per_epoch = min_dataloader_size * \
                                 (args.per_device_generation_batch_size / args.per_device_training_batch_size) * \
                                 args.ppo_epochs / args.gradient_accumulation_steps

    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)

    args.end_of_conversation_token = "<|endoftext|>"

    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(args.generation_batches,
                                   args.per_device_training_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batches,
                                     args.per_device_training_batch_size)

    # monitor
    if rlhf_engine.actor.monitor.enabled:
        monitor = rlhf_engine.actor.monitor
    elif rlhf_engine.critic.monitor.enabled:
        monitor = rlhf_engine.critic.monitor
    else:
        monitor = None

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    non_overflow_step_count = 0
    args.global_step = 0
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Generation Batches {min_dataloader_size}",
            args.global_rank)
        for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)):

            batch_prompt = to_device(batch_prompt, device)

            # prompts = batch_prompt['prompt']
            # length = prompts.size(-1)
            # if length > args.max_prompt_seq_len:
            #     prompts = prompts[:, length - args.max_prompt_seq_len:]
            #     raise ValueError("Prompt length is too long")
            # print_rank_0(f"generate experience ...", args.global_rank)
            out = trainer.generate_experience(batch_prompt['prompt'],
                                              batch_prompt['prompt_att_mask'],
                                              step)

            training_start = time.time()
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_generation_batch_size])

            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                print_rank_0(f"generate experience done", args.global_rank)
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                # average_reward = 0
                monitor_dict = {"rewards": 0, "kl_divergence": 0, "response_len": 0, "actor_ppl": 0, "ref_ppl": 0}

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()

                for ppo_ep in range(args.ppo_epochs):
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        args.global_step += 1
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                        actor_loss_sum += actor_loss
                        critic_loss_sum += critic_loss
                        # average_reward += exp_data["rewards"].mean()
                        for monitor_key in monitor_dict.keys():
                            monitor_dict[monitor_key] += exp_data[monitor_key].mean()

                        if unsupervised_training_enabled:
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)
                            unsup_loss_sum += unsup_loss

                        inner_iter += 1
                        if args.enable_ema:
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)

                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)

                end = time.time()
                training_time = end - training_start
                e2e_time = training_time + trainer.generate_time * args.generation_batches  # it is an approximation, we did not include, e.g., rw forward time etc

                global_samples = rlhf_engine.actor.global_samples

                actor_loss_avg = get_all_reduce_mean(actor_loss_sum).item() / inner_iter
                critic_loss_avg = get_all_reduce_mean(critic_loss_sum).item() / inner_iter
                unsup_loss_avg = get_all_reduce_mean(unsup_loss_sum).item() / inner_iter if unsupervised_training_enabled else 0
                print_rank_0(
                    f'Epoch: {epoch} | Step: {step} | samples: {global_samples} | PPO Epoch: {ppo_ep + 1} | Actor Loss: {actor_loss_avg} | Critic Loss: {critic_loss_avg} | Unsupervised Loss: {unsup_loss_avg}',
                    args.global_rank)
                print_throughput_step3(rlhf_engine.actor.module,
                                       rlhf_engine.critic, args, e2e_time,
                                       trainer.generate_time, training_time,
                                       args.global_rank)
                # average_reward = get_all_reduce_mean(average_reward).item() / inner_iter
                summary_events = []
                for monitor_key in monitor_dict.keys():
                    monitor_dict[monitor_key] = get_all_reduce_mean(monitor_dict[monitor_key]).item() / inner_iter
                    print_rank_0(f"Average {monitor_key}: {monitor_dict[monitor_key]}", args.global_rank)
                    if args.report_to and args.global_rank == 0:
                        summary_events.append((f'Train/Samples/{monitor_key}', monitor_dict[monitor_key], global_samples))
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)
                if args.report_to and args.global_rank == 0:
                    # summary_events.append(('Train/Samples/reward_avg', average_reward, global_samples))
                    summary_events.append(('Train/Samples/actor_loss', actor_loss.item(), global_samples))
                    summary_events.append(('Train/Samples/actor_loss_avg', actor_loss_avg, global_samples))
                    summary_events.append(('Train/Samples/critic_loss', critic_loss.item(), global_samples))
                    summary_events.append(('Train/Samples/critic_loss_avg', critic_loss_avg, global_samples))
                    monitor.write_events(summary_events)

            gloabal_gen_step = epoch * min_dataloader_size + step + 1
            if args.save_steps and gloabal_gen_step % args.save_steps == 0:
                save_ppo_model_hf_format(rlhf_engine, tokenizer, args)

            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()

            actor_overflow, critic_overflow = trainer.get_overflow()

            if not actor_overflow and not critic_overflow:
                non_overflow_step_count += 1

            if args.enable_test_mode and non_overflow_step_count == args.test_stop_step:
                break

        if args.enable_test_mode:
            break

    print_rank_0('saving model ...', args.global_rank)
    save_ppo_model_hf_format(rlhf_engine, tokenizer, args, sub_folder='final')

if __name__ == "__main__":
    main()

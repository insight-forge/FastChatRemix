# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import os
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from qwen_vl_utils import process_vision_info

## for qwen-vl
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_flash_attn: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_image_tokens: int = field(
        default=1280,
        metadata={
            "help": "Maximum number of image tokens."
        },
    )

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    processor: transformers.Qwen2VLProcessor,
) -> Dict:
    assistant_token = processor.tokenizer.encode("assistant")[0]
    # Apply prompt templates

    conversations = []
    image_inputs = []
    video_inputs = []
    image_splits = []
    video_splits = []
    for i, msg in enumerate(sources):
        prompt = processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=False
        )
        conversations.append(prompt)
        msg_image_inputs, msg_video_inputs = process_vision_info(msg)
        image_splits.append(len(msg_image_inputs))
        video_splits.append(len(msg_video_inputs))
        image_inputs.extend(msg_image_inputs)
        video_inputs.extend(msg_video_inputs)

    # Tokenize conversations
    inputs = processor(
        conversations,
        images=image_inputs,
        videos=video_inputs,
        padding="max_length",
        max_length=processor.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs.input_ids
    targets = input_ids.clone()

    image_grid_thw = torch.split(inputs.image_grid_thw, image_splits, dim=0)
    video_grid_thw = torch.split(inputs.video_grid_thw, video_splits, dim=0)
    image_pixel_splits = [grid.prod(dim=-1).sum().item() for grid in image_grid_thw]
    videos_pixel_splits = [grid.prod(dim=-1).sum().item() for grid in video_grid_thw]
    pixel_values = torch.split(inputs.pixel_values, image_pixel_splits, dim=0)
    pixel_values_videos = torch.split(inputs.pixel_values_videos, videos_pixel_splits, dim=0)

    # Mask targets. Only compute loss on the assistant outputs.
    for source, conversation, target in zip(sources, conversations, targets):
        starts = (target == 151644).nonzero(as_tuple=False)
        ends = (target == 151645).nonzero(as_tuple=False)
        if len(starts)==0 or len(ends)==0 or len(starts) == len(ends) and len(source) != len(starts) - 1:
            target[:] = IGNORE_TOKEN_ID
            rank0_print(
                f"WARNING: truncation or special tokenization mismatch: len(target): {len(target)} len(source): {len(source)}, len(starts): {len(starts)}, len(ends): {len(ends)}"
                f" (ignored)"
            )
            continue
        for i, (start, end) in enumerate(zip(starts, ends)):
            if target[start+1] == assistant_token:
                target[start: start + 3] = IGNORE_TOKEN_ID
                target[end + 1: end + 2] = IGNORE_TOKEN_ID
            else:
                target[start: end + 2] = IGNORE_TOKEN_ID
        # target[:starts[0]] = IGNORE_TOKEN_ID
        target[ends[-1]+1:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.pad_token_id, z)
            rank0_print(tokenizer.decode(z))
            print(tokenizer.decode(z))


    return dict(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, processor: transformers.Qwen2VLProcessor,):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, processor)

        self.input_ids = data_dict["input_ids"]
        self.pixel_values = data_dict["pixel_values"]
        self.image_grid_thw = data_dict["image_grid_thw"]
        self.pixel_values_videos = data_dict["pixel_values_videos"]
        self.video_grid_thw = data_dict["video_grid_thw"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            pixel_values=self.pixel_values[i],
            image_grid_thw=self.image_grid_thw[i],
            pixel_values_videos=self.pixel_values_videos[i],
            video_grid_thw=self.video_grid_thw[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, processor: transformers.Qwen2VLProcessor,):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = processor

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["messages"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            pixel_values=ret['pixel_values'][0],
            image_grid_thw=ret['image_grid_thw'][0],
            pixel_values_videos=ret['pixel_values_videos'][0],
            video_grid_thw=ret['video_grid_thw'][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    processor: transformers.Qwen2VLProcessor, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    train_json = []
    for path in data_args.data_path.split(','):
        train_json.extend(json.load(open(path, "r", encoding='utf8')))
    # train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, processor=processor)

    if data_args.eval_data_path:
        eval_json = []
        for path in data_args.eval_data_path.split(','):
            eval_json.extend(json.load(open(path, "r", encoding='utf8')))
        # eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=processor)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, torch_dtype="auto", device_map="auto"
    )

    min_pixels = 256 * 28 * 28
    max_pixels = training_args.max_image_tokens * 28 * 28
    processor = AutoProcessor.from_pretrained(training_args.model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels, model_max_length=training_args.model_max_length)

    # Load data
    data_module = make_supervised_data_module(processor=processor, data_args=data_args)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=processor.tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.config.use_cache = True
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    final_model_path = os.path.join(training_args.output_dir, "final")
    # tokenizer.save_pretrained(final_model_path)
    trainer.save_model(final_model_path)


if __name__ == "__main__":
    train()

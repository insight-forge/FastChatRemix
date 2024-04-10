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
import pathlib
from typing import Dict, Optional, List
import os
from packaging import version

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import dtype as Dtype
from contextlib import nullcontext
# from PIL import Image
import PIL
from PIL.Image import Image

import transformers
from transformers import Trainer, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

from fastchat.model.model_deepseek_vl import MultiModalityCausalLM, VLChatProcessorOutput, BatchedVLChatProcessorOutput, VLChatProcessor
# from fastchat.modules.deepseek_vl_visual import VLChatProcessor, VLChatProcessorOutput, BatchedVLChatProcessorOutput


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_flash_attn : bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use flash attention"
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )


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


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def add_image_token(
        image_indices: List[int],
        input_ids: torch.LongTensor,
        image_id: int = 100015,
        num_image_tokens: int = 576,
        add_special_token: bool = False,
        model_max_length:int  = 2048
    ):
        """

        Args:
            image_indices (List[int]): [index_0, index_1, ..., index_j]
            input_ids (torch.LongTensor): [N]
            labels (torch.LongTensor): [N]

        Returns:
            input_ids (torch.LongTensor): [N + image tokens]
            num_image_tokens (torch.IntTensor): [n_images]
        """

        input_slices = []
        slices_images = []
        start = 0
        for index in image_indices:
            if add_special_token:
                end = index + 1
            else:
                end = index
            # original text tokens
            input_slices.append(input_ids[start:end])
            slices_images.append(0)

            # add image tokens, and set the mask as False
            input_slices.append(
                image_id * torch.ones((num_image_tokens,), dtype=torch.long)
            )
            slices_images.append(1)
            start = index + 1

        # the left part
        input_slices.append(input_ids[start:])
        slices_images.append(0)

        # truncation and concat all slices
        total_len = imags_num = 0
        i = 0
        while i < len(input_slices):
            if total_len + input_slices[i].size()[0] > model_max_length:
                break
            if slices_images[i]:
                imags_num+=1
            i+=1

        # input_ids = torch.cat(input_slices, dim=0)
        input_ids = input_slices[:model_max_length] if i==0 else torch.cat(input_slices[:i], dim=0)
        num_image_tokens = torch.IntTensor([num_image_tokens] * imags_num)

        return input_ids, num_image_tokens

def load_pil_images(image_list: List[str]) -> List[PIL.Image.Image]:
    pil_images = []
    for image_path in image_list:
        pil_img = PIL.Image.open(image_path)
        pil_img = pil_img.convert("RGB")
        pil_images.append(pil_img)

    return pil_images


def get_labels(input_ids, assistant_id, user_id):
    labels = input_ids.clone()
    assistant_indices = (labels == assistant_id).nonzero(as_tuple=False)
    user_indices = (labels == user_id).nonzero(as_tuple=False)
    # assert assistant_indices.size()[0] == user_indices.size()[0]
    # print(assistant_indices[0], user_indices)

    labels[: user_indices[0]] = IGNORE_TOKEN_ID
    for start, end in zip(user_indices, assistant_indices):
        labels[start: end + 3] = IGNORE_TOKEN_ID

    if len(user_indices) > len(assistant_indices):
        labels[user_indices[-1]:] = IGNORE_TOKEN_ID
    return labels

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    roles = {
        "human": 'User',
        "Human": 'User',
        "user": 'User',
        "User": 'User',
        "gpt": 'Assistant',
        "assistant": 'Assistant',
        "Assistant": 'Assistant',
        "system": ""
     }
    # conv = get_conversation_template("vicuna")
    conv = get_conversation_template('deepseek-llm-chat')
    # print(conv.name, conv.roles)
    default_sys_msg = conv.system_message
    assistant_id = tokenizer.vocab.get("Assistant")
    user_id = tokenizer.vocab.get("User")
    image_id = tokenizer.vocab.get("<image_placeholder>")

    prepare_list = []
    for i, source in enumerate(sources):
        conv.system_message = source[0]["value"] if source[0]["from"] == "system" else default_sys_msg
        images = []
        if roles[source[0]["from"]] != "User":
            # Skip the first one if it is not from human
            source = source[1:]
            sources[i] = source
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"].strip())
            if 'images' in sentence:
                images.extend(sentence['images'])
        sft_prompt = conv.get_prompt().strip()
        input_ids = tokenizer.encode(sft_prompt)
        input_ids = torch.LongTensor(input_ids)

        # add image tokens to the input_ids
        image_token_mask: torch.BoolTensor = input_ids == image_id
        image_indices = image_token_mask.nonzero()
        input_ids, num_image_tokens = add_image_token(
            image_indices=image_indices,
            input_ids=input_ids
        )

        labels = get_labels(input_ids, assistant_id, user_id)

        pil_images = load_pil_images(images)

        images_outputs = tokenizer.vl_chat_processor.image_processor(pil_images, return_tensors="pt")
        # prepare = dict(
        #     sft_format=sft_prompt,
        #     input_ids=input_ids,
        #     pixel_values=images_outputs.pixel_values,
        #     num_image_tokens=num_image_tokens,
        #     labels=labels
        # )
        prepare = VLChatProcessorOutput(
            sft_format=sft_prompt,
            input_ids=input_ids,
            pixel_values=images_outputs.pixel_values,
            num_image_tokens=num_image_tokens,
            labels=labels
        )
        # prepare = tokenizer.vl_chat_processor.batchify([prepare])
        prepare_list.append(prepare)

    return prepare_list


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        self.prepare_list = preprocess(sources, tokenizer)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.prepare_list[i]


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        prepare = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)[0]
        self.cached_data_dict[i] = prepare

        return prepare


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    train_json = []
    for path in data_args.data_path.split(','):
        train_json.extend(json.load(open(path, "r", encoding='utf-8')))
    # train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = []
        for path in data_args.eval_data_path.split(','):
            eval_json.extend(json.load(open(path, "r", encoding='utf-8')))
        # eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def get_model_tokenizer_deepseek_vl(model_dir: str,
                                    load_model: bool = True,
                                    use_flash_attn: bool = False):
    vl_chat_processor = VLChatProcessor.from_pretrained(model_dir)
    tokenizer = vl_chat_processor.tokenizer

    # flash_attn
    model_config = AutoConfig.from_pretrained(
        model_dir, trust_remote_code=True)
    if version.parse(transformers.__version__) >= version.parse('4.36'):
        if use_flash_attn:
            model_config.language_config._attn_implementation = 'flash_attention_2'
    else:
        model_config.language_config._flash_attn_2_enabled = use_flash_attn

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

    tokenizer.vl_chat_processor = vl_chat_processor
    if load_model:
        model.generate = model.language_model.generate
        model.get_input_embeddings = model.language_model.get_input_embeddings
        model.gradient_checkpointing_enable = model.language_model.gradient_checkpointing_enable
        model.forward = model.language_model.forward
        model.config = model.language_model.config
    return model, tokenizer


class DataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.image_processor = tokenizer.vl_chat_processor.image_processor
        self.num_image_tokens = tokenizer.vl_chat_processor.num_image_tokens

    def __call__(self, prepare_list: List[VLChatProcessorOutput]):

        batch_size = len(prepare_list)
        sft_format = []
        n_images = []
        seq_lens = []
        for prepare in prepare_list:
            n_images.append(len(prepare.num_image_tokens))
            seq_lens.append(len(prepare))

        input_token_max_len = max(seq_lens)
        max_n_images = max(1, max(n_images))

        batched_input_ids = torch.full(
            (batch_size, input_token_max_len), self.pad_token_id
        ).long()  # FIXME
        batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
        batched_pixel_values = torch.zeros(
            (batch_size, max_n_images, *self.image_processor.default_shape)
        ).float()
        batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
        batched_images_emb_mask = torch.zeros(
            (batch_size, max_n_images, self.num_image_tokens)
        ).bool()

        for i, prepare in enumerate(prepare_list):
            input_ids = prepare.input_ids
            seq_len = len(prepare)
            n_image = len(prepare.num_image_tokens)
            # left-padding
            batched_attention_mask[i, -seq_len:] = 1
            batched_input_ids[i, -seq_len:] = torch.LongTensor(input_ids)
            batched_images_seq_mask[i, -seq_len:] = input_ids == self.pad_token_id

            if n_image > 0:
                batched_pixel_values[i, :n_image] = prepare.pixel_values
                for j, n_image_tokens in enumerate(prepare.num_image_tokens):
                    batched_images_emb_mask[i, j, :n_image_tokens] = True

            sft_format.append(prepare.sft_format)

        batched_prepares = dict(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            pixel_values=batched_pixel_values,
            images_seq_mask=batched_images_seq_mask,
            images_emb_mask=batched_images_emb_mask,
            sft_format=sft_format,
        )

        return batched_prepares

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Load model and tokenizer

    model, tokenizer = get_model_tokenizer_deepseek_vl(
        model_args.model_name_or_path,
        load_model=True,
        use_flash_attn=model_args.use_flash_attn
    )
    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollator(tokenizer)

    rank0_print("training_args:", training_args)
    rank0_print("model_args:", model_args)
    rank0_print("data_args:", data_args)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, data_collator=data_collator, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    # trainer.save_state()
    # if trainer.is_deepspeed_enabled:
    #     trainer.save_model()
    # else:
    #     trainer_save_model_safe(trainer)
    final_model_path = os.path.join(training_args.output_dir, "final")
    tokenizer.save_pretrained(final_model_path)
    trainer.save_model(final_model_path)

if __name__ == "__main__":
    train()
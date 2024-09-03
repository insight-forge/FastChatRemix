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
from collections import Counter

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle, register_conv_template, Conversation
from fastchat.model.model_adapter import get_conv_template

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
local_rank = None
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'


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
    max_dynamic_image_num: int = field(
        default=6, metadata={"help": "The max image num to be splited."}
    )
    dynamic_image_size: bool = True


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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6, dynamic_image_size=True):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    if dynamic_image_size:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


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


register_conv_template(
    Conversation(
        name='phi3-chat',
        system_template='<|system|>\n{system_message}<|end|>',
        system_message='You are an AI assistant whose name is Phi-3.',
        roles=('<|user|>\n', '<|assistant|>\n'),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep='<|end|>',
        stop_token_ids=[
            2,
            32000,
            32007
        ]
    )
)

register_conv_template(
    Conversation(
        name='internlm2-chat',
        system_template='<|im_start|>system\n{system_message}<|im_end|>',
        system_message='You are an AI assistant whose name is InternLM (书生·浦语).',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep='<|im_end|>',
        stop_token_ids=[
            2,
            92543,
            92542
        ]
    )
)


def preprocess_phi3(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        conv_template: Conversation,
        data_args
) -> Dict:
    roles = {
        "human": conv_template.roles[0],
        "user": conv_template.roles[0],
        "Human": conv_template.roles[0],
        "User": conv_template.roles[0],
        "gpt": conv_template.roles[1],
        "assistant": conv_template.roles[1],
        "Assistant": conv_template.roles[1],
        "system": "<|system|>\n",
        "System": "<|system|>\n"
    }
    default_sys_msg = conv_template.system_message
    assistant_token_ids = tokenizer.convert_tokens_to_ids("<|assistant|>")
    user_token_ids = tokenizer.convert_tokens_to_ids("<|user|>")
    end_token_ids = tokenizer.convert_tokens_to_ids("<|end|>")

    # Apply prompt templates
    conversations = []
    pixel_values_all = []
    image_flags_all = []
    for i, source in enumerate(sources):
        conv_template.system_message = source[0]["value"] if roles[source[0][
            "from"]].strip() == "<|system|>" else default_sys_msg
        if source[0]["from"] != "human":
            # Skip the first one if it is not from human
            source = source[1:]
            sources[i] = source
        conv_template.messages = []
        pixel_values = []
        image_flags = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # assert role == conv.roles[j % 2], f"{i}"
            imgs = sentence.get("images", None)
            sentence = sentence["value"].strip()
            img_context_token_ct = sentence.count(IMG_CONTEXT_TOKEN)
            if imgs and img_context_token_ct == 1:
                sentence_pixel_values = torch.cat(
                    [load_image(img, max_num=data_args.max_dynamic_image_num,
                                dynamic_image_size=data_args.dynamic_image_size) for img in imgs])
                sentence_num_patches = sentence_pixel_values.shape[0]
                sentence_image_flags = torch.tensor([1] * sentence_num_patches, dtype=torch.long)
                sentence = sentence.replace(IMG_CONTEXT_TOKEN,
                                            IMG_CONTEXT_TOKEN * sentence_num_patches * data_args.num_image_token)
                pixel_values.append(sentence_pixel_values)
                image_flags.append(sentence_image_flags)
            elif imgs or img_context_token_ct != 0:
                raise (
                    f"Input include {len(imgs)} images, the IMG_CONTEXT_TOKEN number must to be 1, but got {img_context_token_ct}")

            conv_template.append_message(role, sentence)
        if not pixel_values:
            ## fake images
            pixel_values = torch.zeros((1, 3, 448, 448))
            image_flags = torch.tensor([0], dtype=torch.long)
        else:
            pixel_values = torch.cat(pixel_values, dim=0)
            image_flags = torch.cat(image_flags, dim=0)
        assert pixel_values.shape[0] == image_flags.shape[0]
        pixel_values_all.append(pixel_values)
        image_flags_all.append(image_flags)
        conversations.append(conv_template.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    for source, conversation, target in zip(sources, conversations, targets):
        starts = (target == user_token_ids).nonzero(as_tuple=False)
        ends = (target == assistant_token_ids).nonzero(as_tuple=False)
        utterance_ends = (target == end_token_ids).nonzero(as_tuple=False)
        if len(starts) == 0 or len(ends) == 0 or len(starts) == len(ends) and len(source) != len(starts) * 2:
            target[:] = IGNORE_TOKEN_ID
            rank0_print(
                f"WARNING: truncation or special tokenization mismatch: len(target): {len(target)} len(source): {len(source)}, len(starts): {len(starts)}, len(ends): {len(ends)}"
                f" (ignored)"
            )
            continue

        for i, (start, end) in enumerate(zip(starts, ends)):
            target[start: end + 1] = IGNORE_TOKEN_ID
        target[:starts[0]] = IGNORE_TOKEN_ID
        target[utterance_ends[-1] + 1:] = IGNORE_TOKEN_ID
        if len(starts) == len(ends) + 1:
            target[starts[-1]:] = IGNORE_TOKEN_ID
        # ends = (target == end_token_ids).nonzero(as_tuple=False)
        # target[ends[-1]+1:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.pad_token_id, z)
            rank0_print(tokenizer.decode(z))
            # print(tokenizer.decode(z))

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        pixel_values=pixel_values_all,
        image_flags=image_flags_all,
        prompts=conversations
    )


def preprocess_internlm2(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        conv_template: Conversation,
        data_args
) -> Dict:
    roles = {
        "human": conv_template.roles[0],
        "user": conv_template.roles[0],
        "Human": conv_template.roles[0],
        "User": conv_template.roles[0],
        "gpt": conv_template.roles[1],
        "assistant": conv_template.roles[1],
        "Assistant": conv_template.roles[1],
        "system": '<|im_start|>system\n',
        "System": '<|im_start|>system\n',
    }
    default_sys_msg = conv_template.system_message
    im_start_token_ids = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_token_ids = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Apply prompt templates
    conversations = []
    pixel_values_all = []
    image_flags_all = []
    for i, source in enumerate(sources):
        conv_template.system_message = source[0]["value"] if roles[source[0][
            "from"]].strip() == '<|im_start|>system' else default_sys_msg
        if source[0]["from"] != "human":
            # Skip the first one if it is not from human
            source = source[1:]
            # sources[i] = source
        conv_template.messages = []
        pixel_values = []
        image_flags = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # assert role == conv.roles[j % 2], f"{i}"
            imgs = sentence.get("images", None)
            sentence = sentence["value"].strip()
            img_context_token_ct = sentence.count(IMG_CONTEXT_TOKEN)
            if imgs and img_context_token_ct == 1:
                sentence_pixel_values = torch.cat(
                    [load_image(img, max_num=data_args.max_dynamic_image_num,
                                dynamic_image_size=data_args.dynamic_image_size) for img in imgs])
                sentence_num_patches = sentence_pixel_values.shape[0]
                sentence_image_flags = torch.tensor([1] * sentence_num_patches, dtype=torch.long)
                sentence = sentence.replace(IMG_CONTEXT_TOKEN,
                                            IMG_CONTEXT_TOKEN * sentence_num_patches * data_args.num_image_token)
                pixel_values.append(sentence_pixel_values)
                image_flags.append(sentence_image_flags)
            elif imgs or img_context_token_ct != 0:
                raise (
                    f"Input include {len(imgs)} images, the IMG_CONTEXT_TOKEN number must to be 1, but got {img_context_token_ct}")

            conv_template.append_message(role, sentence)
        if not pixel_values:
            ## fake images
            pixel_values = torch.zeros((1, 3, 448, 448))
            image_flags = torch.tensor([0], dtype=torch.long)
        else:
            pixel_values = torch.cat(pixel_values, dim=0)
            image_flags = torch.cat(image_flags, dim=0)
        assert pixel_values.shape[0] == image_flags.shape[0]
        pixel_values_all.append(pixel_values)
        image_flags_all.append(image_flags)
        conversations.append(conv_template.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.\
    for source, conversation, target in zip(sources, conversations, targets):
        starts = (target == im_start_token_ids).nonzero(as_tuple=False)
        ends = (target == im_end_token_ids).nonzero(as_tuple=False)
        if len(starts) == 0 or len(ends) == 0 or len(starts) != len(ends):
            target[:] = IGNORE_TOKEN_ID
            rank0_print(
                f"WARNING: truncation or special tokenization mismatch: len(target): {len(target)} len(source): {len(source)}, len(starts): {len(starts)}, len(ends): {len(ends)}"
                f" (ignored)"
            )
            continue
        for i, (start, end) in enumerate(zip(starts, ends)):
            role = roles[source[i]["from"]]
            # if source[i]
            if role == conv_template.roles[1]:
                # assistant
                target[start: start + 4] = IGNORE_TOKEN_ID
            else:
                # system or user
                target[start: end + 1] = IGNORE_TOKEN_ID
        target[:starts[0]] = IGNORE_TOKEN_ID
        target[ends[-1]:] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.pad_token_id, z)
            rank0_print(tokenizer.decode(z))
            print(tokenizer.decode(z))

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        pixel_values=pixel_values_all,
        image_flags=image_flags_all,
        prompts=conversations
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        conv_template = get_conv_template(tokenizer.template)
        rank0_print("conv_template name:", conv_template.name)
        preprocess = preprocess_phi3 if tokenizer.template == "phi3-chat" else preprocess_internlm2
        data_dict = preprocess(sources, tokenizer, conv_template, data_args)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.pixel_values = data_dict["pixel_values"]
        self.image_flags = data_dict["image_flags"]
        self.prompts = data_dict["prompts"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
            pixel_values=self.pixel_values[i],
            image_flags=self.image_flags[i],
            prompts=self.prompts[i]
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.conv_template = get_conv_template(tokenizer.template)
        rank0_print("conv_template name:", self.conv_template.name)
        self.data_args = data_args
        self.preprocess = preprocess_phi3 if tokenizer.template == "phi3-chat" else preprocess_internlm2

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = self.preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.conv_template, self.data_args)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=ret["pixel_values"][0],
            image_flags=ret["image_flags"][0],
            prompts=ret["prompts"][0]
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments
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
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, data_args=data_args)

    if data_args.eval_data_path:
        eval_json = []
        for path in data_args.eval_data_path.split(','):
            eval_json.extend(json.load(open(path, "r", encoding='utf8')))
        # eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, data_args=data_args)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


class DataCollator:
    def __call__(self, prepare_list: list):
        input_ids = torch.stack([_["input_ids"] for _ in prepare_list])
        labels = torch.stack([_["labels"] for _ in prepare_list])
        attention_mask = torch.stack([_["attention_mask"] for _ in prepare_list])

        pixel_values = [_["pixel_values"] for _ in prepare_list if _["pixel_values"] is not None]
        image_flags = [_["image_flags"] for _ in prepare_list if _["image_flags"] is not None]
        pixel_values = torch.cat(pixel_values) if pixel_values else torch.zeros((0, 1), dtype=torch.long)
        image_flags = torch.cat(image_flags) if image_flags else torch.zeros((0, 1), dtype=torch.long)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_flags=image_flags
        )


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        use_flash_attn=model_args.use_flash_attn,
        dynamic_image_size=data_args.dynamic_image_size
    )
    # Load model and tokenizer
    model = transformers.AutoModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
        trust_remote_code=True
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    model.supports_gradient_checkpointing = True
    model.config.hidden_size = model.language_model.config.hidden_size
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tokenizer.template = model.template
    data_args.num_image_token = model.num_image_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollator()
    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, data_collator=data_collator, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    model.config.use_cache = True
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    final_model_path = os.path.join(training_args.output_dir, "final")
    tokenizer.save_pretrained(final_model_path)
    trainer.save_model(final_model_path)


def test_dataset():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        use_flash_attn=model_args.use_flash_attn
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.template = config.template
    image_size = config.force_image_size or config.vision_config.image_size
    patch_size = config.vision_config.patch_size
    data_args.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
    rank0_print("Loading data...")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    eval_dataset = data_module['eval_dataset']
    # train_sampler = RandomSampler(train_dataset)
    # data_collator = DataCollator()
    # dataloader = DataLoader(train_dataset,
    #            collate_fn=data_collator,
    #            sampler=train_sampler,
    #            batch_size=training_args.per_device_train_batch_size)
    if local_rank == 0:
        for i in range(0, len(eval_dataset), 20):
            print("===" * 30)
            print("prompts: " + eval_dataset[i]['prompts'])
            print("===" * 30)
            input_ids = eval_dataset[i]['input_ids']
            print("input_ids stat:", Counter(input_ids.tolist()))
            print("decode input_ids: " + tokenizer.decode(input_ids))
            print("--" * 30)
            labels = eval_dataset[i]['labels']
            print("label stat:", Counter(labels.tolist()))
            z = torch.where(labels == IGNORE_TOKEN_ID, tokenizer.pad_token_id, labels)
            print("decode labels: " + tokenizer.decode(z))
            print("===" * 30)

    rank0_print("Done!")


if __name__ == "__main__":
    train()
    # test_dataset()

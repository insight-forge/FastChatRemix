import os
from tqdm import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import re
import pandas as pd
import json
import argparse

CHOICES = ["A", "B", "C", "D"]
NTRAIN = 5
SUBJECTS = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics',
            'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics',
            'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics',
            'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology',
            'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine', 'college_economics',
            'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification',
            'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography',
            'modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law',
            'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional',
            'high_school_chinese', 'high_school_history', 'middle_school_history', 'civil_servant', 'sports_science',
            'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant',
            'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']


class HfModel_Evaluator:
    def __init__(self, choices, k, model_name, model_path, max_new_tokens):
        self.choices = choices
        self.model_name = model_name
        self.k = k
        self.max_new_tokens = max_new_tokens

        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )
        self.tokenizer.padding_side = "right"
        print()
        if 'qwen' in model_path.lower():
            print(model_path)
            self.config.use_flash_attn = False
            # https://github.com/QwenLM/Qwen-7B/blob/main/examples/tokenizer_showcase.ipynb
            self.tokenizer.eos_token_id = self.tokenizer.eod_id
            self.tokenizer.pad_token_id = self.tokenizer.special_tokens['<|extra_0|>']
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                revision='main',
                torch_dtype=torch.float16
            ).cuda()
            self.model.config.eos_token_id = [self.tokenizer.im_end_id, self.tokenizer.eos_token_id]
            self.model.config.pad_token_id = self.tokenizer.pad_token
        else:
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                revision='main',
                torch_dtype=torch.float16
            ).cuda()

    def format_example(self, line, include_answer=True, cot=False):
        example = line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

        example += '\n答案：'
        if include_answer:
            if cot:
                ans = line["answer"]
                content = "让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{ans}。"
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": content}
                ]
            else:
                return [
                    {"role": "user", "content": example},
                    {"role": "assistant", "content": line["answer"]}
                ]
        else:
            return [
                {"role": "user", "content": example},
            ]

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"你是一个中文人工智能助手，以下有几个中国关于{subject}考试的单项选择题示例和一个问题，请回答最后给出的问题答案，如示例所示回答A、B、C、D中的一个，不需要多余的回答。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp = self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
            user = tmp[0]['content']
            model = tmp[1]['content']
            prompt += f"示例{i}：{user}{model}\n"
        return prompt
    def generate_few_shot_chat_prompt(self, subject, dev_df, cot=False):
        prompt = f"你是一个中文人工智能助手，以下有几个中国关于{subject}考试的单项选择题示例，接下来用户会提出相同类型的考试问题，请直接回答最终的答案，如示例所示回答A、B、C、D中的一个，不需要多余的回答或解析。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            tmp = self.format_example(dev_df.iloc[i, :], include_answer=True, cot=cot)
            user = tmp[0]['content']
            model = tmp[1]['content']
            prompt += f"示例{i}：{user}{model}\n"
        return prompt

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None, cot=False):
        correct_num = 0
        res_dict = {}
        if save_result_dir:
            result = []
            processed_res = []
        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            few_shot_prompt = f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n"
        message_list = []
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False)
            full_prompt = few_shot_prompt + "问题：" + question[0]['content']
            if 'cpm' in self.model_name:
                full_prompt = "USER: " + few_shot_prompt + "问题：" + question[0]['content']
                full_prompt = full_prompt.replace('<', '<<')
            elif 'qwen' in self.model_path.lower() and 'chat' in self.model_path.lower():
                # full_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" \
                #               + few_shot_prompt + "问题：" + question[0]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
                few_shot_prompt = self.generate_few_shot_chat_prompt(subject_name, dev_df, cot=cot)
                full_prompt = f"<|im_start|>system\n{few_shot_prompt}<|im_end|>\n<|im_start|>user\n" \
                             + "问题：" + question[0]['content'] + "<|im_end|>\n<|im_start|>assistant\n"
            # print(full_prompt)
            message_list.append(full_prompt)
            if len(message_list) % 1 == 0 or row_index == len(test_df) - 1:
                response_list = []
                if 'cpm' not in self.model_name:
                    inputs = self.tokenizer(message_list, return_tensors="pt", padding=True)
                    for k in inputs:
                        inputs[k] = inputs[k].cuda()
                    outputs = self.model.generate(**inputs, do_sample=True, temperature=0.2, top_p=0.8,
                                                  repetition_penalty=1.02,
                                                  max_new_tokens=512 if self.max_new_tokens <= 0 else self.max_new_tokens)
                    input_len = torch.max(torch.sum(inputs.attention_mask, axis=1))
                    response_list = [
                        self.tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
                        for i in range(outputs.shape[0])
                    ]
                else:
                    try:
                        inputs = self.tokenizer(message_list, return_tensors="pt", padding=True)
                        for k in inputs:
                            inputs[k] = inputs[k].cuda()
                        max_tok = 2040 - inputs.input_ids.shape[1]
                        outputs = self.model.generate(**inputs, do_sample=True, temperature=0.2, top_p=0.8,
                                                      repetition_penalty=1.02,
                                                      max_new_tokens=max_tok if self.max_new_tokens <= 0 else self.max_new_tokens)
                        input_len = torch.max(torch.sum(inputs.attention_mask, axis=1))
                        response_list = [
                            self.tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
                            for i in range(outputs.shape[0])
                        ]
                    except:
                        print(full_prompt)
                        response_list = [None]
                for i, response_str in enumerate(response_list):
                    # print('--------\n' + response_str + '\n------------')
                    response_str = response_str.strip()
                    try:
                        if not response_str:
                            res = None
                        elif response_str[0].upper() in ['A', 'B', 'C', 'D']:
                            res = response_str[0].upper()
                        elif re.findall(r"答案是选项(.+)", response_str):
                            res = (re.findall(r"答案是选项(.+)", response_str)[-1]).strip()[0].upper()
                        elif re.findall(r"答案为选项(.+)", response_str):
                            res = (re.findall(r"答案为选项(.+)", response_str)[-1]).strip()[0].upper()
                        elif re.findall(r"答案是(.+)", response_str):
                            res = (re.findall(r"答案是(.+)", response_str)[-1]).strip()[0].upper()
                        elif re.findall(r"答案为(.+)", response_str):
                            res = (re.findall(r"答案为(.+)", response_str)[-1]).strip()[0].upper()
                        elif re.findall(r"答案：(.+)", response_str):
                            res = (re.findall(r"答案：(.+)", response_str)[-1]).strip()[0].upper()
                        elif re.findall(r"选项(.+)正确", response_str):
                            res = (re.findall(r"选项(.+)正确", response_str)[-1]).strip()[0].upper()
                        else:
                            res = response_str[0].upper()
                    except:
                        res = None
                    if save_result_dir:
                        result.append(response_str)
                        processed_res.append(res)
                        res_dict[str(row_index)] = res

                message_list = []

        if save_result_dir:
            test_df['model_output'] = result
            test_df['processed_result'] = processed_res
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'), encoding="utf-8", index=False)
        return res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="The path to the ceval data dir",
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="./output.json",
        help="JSON format file, saving the results to be uploaded",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=-1,
        help="max new tokens for generating",
    )
    args = parser.parse_args()

    model_name = args.model_path.split('/')[-1]
    evaluator = HfModel_Evaluator(choices=CHOICES, k=NTRAIN, model_name=model_name, model_path=args.model_path, max_new_tokens=args.max_new_tokens)

    if not os.path.exists(r"logs"):
        os.mkdir(r"logs")
    save_result_dir = os.path.join(r"logs", f"{model_name}")
    if not os.path.exists(save_result_dir):
        os.mkdir(save_result_dir)

    res_dict = {}
    for subject in SUBJECTS:
        print(subject)
        val_file_path = os.path.join(args.data_dir, f'test/{subject}_test.csv')
        val_df = pd.read_csv(val_file_path)

        dev_file_path = os.path.join(args.data_dir, f'dev/{subject}_dev.csv')
        dev_df = pd.read_csv(dev_file_path)

        cur_dict = evaluator.eval_subject(subject, val_df, dev_df, few_shot=True, save_result_dir=save_result_dir)
        res_dict[subject] = cur_dict

    with open(args.out_file, 'w') as f:
        f.write(json.dumps(res_dict))

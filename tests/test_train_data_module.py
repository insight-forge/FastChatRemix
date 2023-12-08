import transformers

from fastchat.train.train import make_supervised_data_module, ModelArguments, DataArguments, TrainingArguments

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    print(data_module)

if __name__ == "__main__":
    # base_dir = /data1/llm/
    # --model_max_length 4096
    # --model_name_or_path f"{base_dir}/tigerbot-13b-base-sft0901"
    # --cache_dir f"{base_dir}/tmp"
    # --data_path f"{base_dir}/glaive-function-calling-v2/glaive-function-calling-v2-fastchat.json"
    # --output_dir f"{base_dir}/output"
    main()

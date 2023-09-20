import argparse
import json

from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = max_length = None


def main(args):

    global tokenizer, max_length
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    max_length = args.max_length

    content = json.load(open(args.in_file, "r"))
    result = []

    for sample in tqdm(content):
        total_len = 0
        conversations = sample["conversations"]
        for sentence in conversations:
            total_len += (len(tokenizer(sentence["value"]).input_ids) + 8)
            if total_len > max_length:
                print(f"The content length is longer than {max_length}: ", sample)
                break

        if total_len <= max_length:
            result.append(sample)

    print(f"#in: {len(content)}, #out: {len(result)}")
    json.dump(result, open(args.out_file, "w"), indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, default="sharegpt_split.json")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()
    print("args: ", args)
    main(args)

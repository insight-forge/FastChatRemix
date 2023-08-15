from fastchat.train.train_qwen7b import preprocess
import json
from transformers import AutoTokenizer
def test_preprocess():
    path = 'path/to/conversations'
    with open(path, 'r', encoding='utf8') as f:
        raw_data = json.load(f)
        # raw_data[0]["conversations"] = raw_data[0]["conversations"] + raw_data[1]["conversations"] + raw_data[2]["conversations"]
        # raw_data = raw_data[:1]
        # print(len(raw_data[0]["conversations"]))
    print("=====")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True, model_max_length=1024)
    tokenizer.eos_token_id = tokenizer.eod_id
    tokenizer.pad_token_id = tokenizer.special_tokens['<|extra_0|>']
    sources = [example["conversations"] for example in raw_data]
    ret = preprocess(sources, tokenizer)
    for key in ret.keys():
        ret[key] = ret[key].tolist()
    # print(ret)
    with open("test_data.json", 'w', encoding='utf8') as f:
        f.write(json.dumps(ret, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    test_preprocess()
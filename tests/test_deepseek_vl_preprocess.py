from fastchat.model.model_deepseek_vl import VLChatProcessor
from fastchat.train.train_deepseek_vl import preprocess, DataCollator
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']=True
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model_name_or_path = '/Users/admin/Desktop/models/deepseek-vl-1.3b-base'
vl_chat_processor = VLChatProcessor.from_pretrained(model_name_or_path)
tokenizer = vl_chat_processor.tokenizer
tokenizer.vl_chat_processor = vl_chat_processor

def test_local():
    path = '/Users/admin/Desktop/图片 1.png'
    sources = [
        [

            {
                "from": "User",
                "value": "<image_placeholder>你好",
                "images": [path]
            },
            {
                "from": "Assistant",
                "value": '一幅画中，一束白色玫瑰花在绿色花瓶中盛开，背景是浅蓝色。'
            }
        ],
        [
            {
                "from": "User",
                "value": "<image_placeholder> <image_placeholder>用中文简洁描述图片中的场景和关键信息，70字以内.",
                "images": [path, path]
            },
            {
                "from": "Assistant",
                "value": '一幅画中，一束白色玫瑰花在绿色花瓶中盛开，背景是浅蓝色。'
            }
        ]

    ]

    model_name_or_path = '/Users/admin/Desktop/models/deepseek-vl-1.3b-base'
    vl_chat_processor = VLChatProcessor.from_pretrained(model_name_or_path)
    tokenizer = vl_chat_processor.tokenizer
    tokenizer.vl_chat_processor = vl_chat_processor
    dl = DataCollator(tokenizer)
    prepare_list = preprocess(
        sources,
        tokenizer
    )
    dl(prepare_list)

    print(prepare_list)
    print('done')
    # return True

def test_on_service():
    path = '/opt/ml/input/zheli/story-uimage/ZL_cnt_2_01GWy551ksdL'
    sources = [
        [

            {
                "from": "User",
                "value": "<image_placeholder>你好",
                "images": [path]
            },
            {
                "from": "Assistant",
                "value": '一幅画中，一束白色玫瑰花在绿色花瓶中盛开，背景是浅蓝色。'
            }
        ],
        [
            {
                "from": "User",
                "value": "<image_placeholder> <image_placeholder>用中文简洁描述图片中的场景和关键信息，70字以内.",
                "images": [path, path]
            },
            {
                "from": "Assistant",
                "value": '一幅画中，一束白色玫瑰花在绿色花瓶中盛开，背景是浅蓝色。'
            }
        ]

    ]

    model_name_or_path = '/opt/ml/model/deepseek-vl-1.3b-chat'
    model, tokenizer = get_model_tokenizer_deepseek_vl(
        model_name_or_path,
        load_model=True,
        use_flash_attn=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    dl = DataCollator(tokenizer)
    prepare_list = preprocess(
        sources,
        tokenizer
    )

    batch_prepare = dl(prepare_list)
    # model.forward(**batch_prepare)
    model.prepare_inputs_embeds(**batch_prepare)
    # print(batch_prepare['images_seq_mask'].shape, batch_prepare['images_emb_mask'].shape)

    for _ in ['input_ids', 'pixel_values', 'attention_mask', 'images_seq_mask', 'images_emb_mask']:
        if isinstance(_, list):
            print(len(_))
        else:
            print(_, batch_prepare[_].shape)

    # model.prepare_inputs_embeds(**batch_prepare)
    print('done')


if __name__ == '__main__':
    test()
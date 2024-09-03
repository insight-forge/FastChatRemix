# -*- coding: gbk -*-
from fastchat.train.train_internVL import preprocess

import json
from transformers import AutoTokenizer

default_test_convs = [
    {
        "id": "7kkIcQi_16",
        "conversations": [
            {
                "from": "system",
                "value": "你是一个有用的人工只能助手！"
            },
            {
                "from": "human",
                "value": "小张：我想了解美国的教育体系，你能介绍一下吗？"
            },
            {
                "from": "gpt",
                "value": "鲍勃：当然可以。美国的教育体系是一个由各个州独立管理的联邦制度，各个州都有自己的教育政策和法规。然而，大多数州都会遵循联邦政府的教育标准和课程，这些标准由美国教育部制定并更新。美国的教育系统可以分为以下几个阶段：\n\n1. 幼儿教育：通常从3岁开始，到5或6岁结束。幼儿教育通常包括游戏、音乐、艺术和基础数学和语言学习等方面。\n2. 基础教育：通常从6岁开始，持续到18岁，包括小学、初中和高中。在这个阶段，学生需要学习核心科目，如英语、数学、科学和社会研究，并在高中阶段选择其他选修课程。\n3. 高等教育：包括大学和职业教育学院，通常在18岁后开始。在这个阶段，学生可以选择各种各样的专业，例如商业、医学、法律、科学、艺术等。\n\n美国的教育体系注重培养学生的创造力、批判性思维和解决问题的能力，许多学校提供多样化的课程和活动，以便学生能够获得全面的教育和个人发展。同时，教育也是美国政府和社会的一个重要领域，政府每年都会投入大量的经费和资源，以确保所有学生都能获得公平和高质量的教育。"
            },
            {
                "from": "human",
                "value": "小张：我在美国看到很多人玩橄榄球，这是美国的传统运动吗？"
            },
            {
                "from": "gpt",
                "value": "鲍勃：是的，橄榄球是美国的一项传统运动。它起源于英国的橄榄球比赛，并于19世纪传入美国。在美国，橄榄球在学校和大学校园内非常流行，而且是一项备受关注的职业运动。每年的超级碗比赛就是橄榄球界的一项盛事，吸引了全球数亿观众的关注。不过，除了橄榄球之外，美国还有很多其他的传统运动，例如篮球、棒球和美式足球等。每种运动都有其独特的规则和特点，吸引了许多美国人的喜爱和支持。"
            },
            {
                "from": "human",
                "value": "我听说美国的食物种类很多，你能介绍一些美国的特色食品吗？"
            },
            {
                "from": "gpt",
                "value": "鲍勃：当然可以。美国是一个多元文化的国家，其食品种类也非常丰富多样。以下是一些美国的特色食品：\n\n1. 汉堡：汉堡是一种经典的美国快餐食品，由肉饼和面包构成，通常还会加上各种配料，例如生菜、番茄、酸黄瓜、洋葱和芝士等。\n2. 烤肉：烤肉是美国家庭烹饪的常见方式，经常在夏季进行户外烧烤活动。常见的烤肉食材包括牛排、猪排、鸡肉、香肠和汉堡等。\n3. 炸鸡：炸鸡是美国南部的一种传统食品，以其酥脆的外皮和多汁的肉质而闻名。通常配有饼干、饭团或薯条等。\n4. 美式早餐：美式早餐包括煎蛋、培根、烤面包、香肠和煎饼等，通常会配以咖啡或橙汁。\n5. 热狗：热狗是一种快餐食品，由烤面包和香肠组成，常配有芥末、番茄酱和薯片等。\n\n当然，这只是美国食品中的一小部分，每个州和地区都有其独特的食品文化，您可以在旅行时尝试不同的食品，体验美国的饮食文化。"
            }
        ]
    },
    {
        "id": "Step 7: I'm troubleshooting a network issue and suspect it's related to the DNS configuration. Can you perform an nslookup for the domain name 'example.com' and provide me with the A, AAAA, MX, NS, and TXT records? Additionally, I need to perform a traceroute to 'example.com' to determine the network path.",
        "conversations": [
            {
                "from": "system",
                "value": "You are AutoGPT, you can use many tools(functions) to do the following task.\nFirst I will give you the task description, and your task start.\nAt each step, you need to give your thought to analyze the status now and what to do next, with a function call to actually excute your step. Your output should follow this format:\nThought:\nAction\nAction Input:\n\nAfter the call, you will get the call result, and you are now in a new state.\nThen you will analyze your status now, then decide what to do next...\nAfter many (Thought-call) pairs, you finally perform the task, then you can give your finial answer.\nRemember: \n1.the state change is irreversible, you can't go back to one of the former state, if you want to restart the task, say \"I give up and restart\".\n2.All the thought is short, at most in 5 sentence.\n3.You can do more then one trys, so if your plan is to continusly try some conditions, you can do one of the conditions per try.\nLet's Begin!\nTask description: You should use functions to help handle the real time user querys. Remember:\n1.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.\n2.Do not use origin tool names, use only subfunctions' names.\nYou have access of the following tools:\n1.viewdns: Your one source for DNS related tools! dns, info, reverse ip, pagerank, portscan, port scan, lookup, records, whois, ipwhois, dnstools, web hosting, hosting, traceroute, dns report, dnsreport, ip location, ip location finder, spam, spam database, dnsbl, propagation, dns propagation checker, checker, china, chinese, firewall, great firewall, is my site down, is site down, site down, down, dns propagate\n2.dns_lookup: DNS Lookup API lets you gather a domain' corresponding IP address/A record, mail server/MX record, DNS servers/NS nameservers, as well as other items like SPF records/TXT records.\n\nSpecifically, you have access to the following APIs: [{'name': 'traceroute_for_viewdns', 'description': 'This is the subfunction for tool \"viewdns\", you can use this tool.The description of this function is: \"Determines the series of servers that data traverses from the ViewDNS server to the specified domain name or IP address.\"', 'parameters': {'type': 'object', 'properties': {'domain': {'type': 'string', 'description': 'the domain or IP address to perform a traceroute on'}, 'output': {'type': 'string', 'description': \"the output format required ('xml' or 'json')\"}}, 'required': ['domain', 'output'], 'optional': []}}, {'name': 'dns_record_lookup_for_viewdns', 'description': 'This is the subfunction for tool \"viewdns\", you can use this tool.The description of this function is: \"View all configured DNS records (A, MX, CNAME etc.) for a specified domain name.\"', 'parameters': {'type': 'object', 'properties': {'domain': {'type': 'string', 'description': 'the domain name to lookup DNS records for'}, 'output': {'type': 'string', 'description': \"the output format required ('xml' or 'json')\"}, 'recordtype': {'type': 'string', 'description': \"the type of DNS record you wish to retrieve (default 'ANY')\"}}, 'required': ['domain', 'output'], 'optional': ['recordtype']}}, {'name': 'mac_address_lookup_for_viewdns', 'description': 'This is the subfunction for tool \"viewdns\", you can use this tool.The description of this function is: \"This tool will display the name of the company that manufactured a specific network device based on its MAC Address.\"', 'parameters': {'type': 'object', 'properties': {'mac': {'type': 'string', 'description': 'the MAC address to lookup', 'example_value': '00-05-02-34-56-78'}, 'output': {'type': 'string', 'description': \"the output format required ('xml' or 'json')\"}}, 'required': ['mac', 'output'], 'optional': []}}, {'name': 'reverse_for_dns_lookup', 'description': 'This is the subfunction for tool \"dns_lookup\", you can use this tool.The description of this function is: \"Get name servers from an IP\"', 'parameters': {'type': 'object', 'properties': {'ip': {'type': 'string', 'description': '', 'example_value': '8.8.8.8'}}, 'required': ['ip'], 'optional': []}}, {'name': 'nslookup_for_dns_lookup', 'description': 'This is the subfunction for tool \"dns_lookup\", you can use this tool.The description of this function is: \"NsLookup queries the specified DNS server and retrieves the requested records that are associated with the domain name you provided. These records contain information like the domain name\u2019s IP addresses.\\n\\nThe following types of DNS records are especially u\"', 'parameters': {'type': 'object', 'properties': {'domain': {'type': 'string', 'description': '', 'example_value': 'google.com'}}, 'required': ['domain'], 'optional': []}}, {'name': 'whoisserver_dnsservice_for_dns_lookup', 'description': 'This is the subfunction for tool \"dns_lookup\", you can use this tool.The description of this function is: \"DNS Lookup API\"', 'parameters': {'type': 'object', 'properties': {'type': {'type': 'string', 'description': 'DNS type: A, NS, SOA, MX, etc. You can specify multiple comma-separated values, e.g., A,SOA,TXT; all records can be retrieved with type=_all.\\nAcceptable values: [A, NS, SOA, MX, etc.](https://dns-lookup.whoisxmlapi.com/api/documentation/making-requests#Sup', 'example_value': 'A'}, 'domainname': {'type': 'string', 'description': 'The target domain name.', 'example_value': 'google.com'}, 'outputformat': {'type': 'string', 'description': 'Response output format.\\nAcceptable values: JSON | XML\\nDefault: XML', 'example_value': 'JSON'}, 'callback': {'type': 'string', 'description': 'A javascript function used when outputFormat is JSON; this is an implementation known as JSONP which invokes the callback on the returned response.'}}, 'required': ['type', 'domainname'], 'optional': ['outputformat', 'callback']}}, {'name': 'Finish', 'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.', 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"'}}, 'required': ['return_type']}}]"
            },
            {
                "from": "user",
                "value": "\nI'm troubleshooting a network issue and suspect it's related to the DNS configuration. Can you perform an nslookup for the domain name 'example.com' and provide me with the A, AAAA, MX, NS, and TXT records? Additionally, I need to perform a traceroute to 'example.com' to determine the network path.\nBegin!\n"
            },
            {
                "from": "assistant",
                "value": "\nThought: \nAction: dns_record_lookup_for_viewdns\nAction Input: {\n  \"domain\": \"example.com\",\n  \"output\": \"json\",\n  \"recordtype\": \"A,AAAA,MX,NS,TXT\"\n}"
            },
            # {
            #     "from": "function",
            #     "value": "{\"error\": \"Message error...\", \"response\": \"There was an error processing your API key - it may be invalid or disabled.  Please try again later.  If the problem continues to occur please contact the webmaster.\"}"
            # },
            {
                "from": "user",
                "value": "This is not the first time you try this task, all previous trails failed.\nBefore you generate my thought for this state, I will first show you your previous actions for this state, and then you must generate actions that is different from all of them. Here are some previous actions candidates:\n[\n  {\n    \"name\": \"nslookup_for_dns_lookup\",\n    \"arguments\": \"{\\n  \\\"domain\\\": \\\"example.com\\\"\\n}\",\n    \"function_output\": \"{\\\"error\\\": \\\"\\\", \\\"response\\\": \\\"{'host': 'example.com', 'result': {'A': ['93.184.216.34'], 'AAAA': ['2606:2800:220:1:248:1893:25c8:1946'], 'MX': [{'exchange': '', 'priority': 0}], 'NS': ['a.iana-servers.net', 'b.iana-servers.net'], 'SOA': {'nsname': 'ns.icann.org', 'hostmaster': 'noc.dns.icann.org', 'serial': 2022091303, 'refresh': 7200, 'retry': 3600, 'expire': 1209600, 'minttl': 3600}, 'TXT': ['v=spf1 -all', 'wgyf8z8cgvm2qmxpnbnldrcltvk4xqfn']}}\\\"}\",\n    \"mento-carlo-action-value\": 0.0\n  }\n]\nagain, your former observation: {\"error\": \"Message error...\", \"response\": \"There was an error processing your API key - it may be invalid or disabled.  Please try again later.  If the problem continues to occur please contact the webmaster.\"}\n\nRemember you are now in the intermediate state of a trail, you will first analyze the now state and previous action candidates, then make actions that is different from all the previous."
            },
            {
                "from": "assistant",
                "value": "\nThought: \nAction: nslookup_for_dns_lookup\nAction Input: {\n  \"domain\": \"example.com\"\n}"
            },
            # {
            #     "from": "function",
            #     "value": "{\"error\": \"\", \"response\": \"{'host': 'example.com', 'result': {'A': ['93.184.216.34'], 'AAAA': ['2606:2800:220:1:248:1893:25c8:1946'], 'MX': [{'exchange': '', 'priority': 0}], 'NS': ['a.iana-servers.net', 'b.iana-servers.net'], 'SOA': {'nsname': 'ns.icann.org', 'hostmaster': 'noc.dns.icann.org', 'serial': 2022091303, 'refresh': 7200, 'retry': 3600, 'expire': 1209600, 'minttl': 3600}, 'TXT': ['v=spf1 -all', 'wgyf8z8cgvm2qmxpnbnldrcltvk4xqfn']}}\"}"
            # },
            # {
            #     "from": "assistant",
            #     "value": "\nThought: \nAction: traceroute_for_viewdns\nAction Input: {\n  \"domain\": \"example.com\",\n  \"output\": \"json\"\n}"
            # }
        ]
    },

]
def test_preprocess():
    # ## 文件中读取对话数据
    # path = "path/to/conversation/data"
    # with open(path, 'r', encoding='utf8') as f:
    #     raw_data = json.load(f)
    #     # raw_data[0]["conversations"] = raw_data[0]["conversations"] + raw_data[1]["conversations"] + raw_data[2]["conversations"]
    #     # raw_data = raw_data[:1]
    #     # print(len(raw_data[0]["conversations"]))
    #     # print(raw_data)
    #######
    ## 默认数据
    raw_data = default_test_convs[1:2]
    #######

    tokenizer = AutoTokenizer.from_pretrained('/Users/admin/Desktop/models/Mini-InternVL-Chat-4B-V1-5', trust_remote_code=True, model_max_length=2048)
    # tokenizer.eos_token_id = tokenizer.eod_id
    # tokenizer.pad_token_id = tokenizer.special_tokens['<|extra_0|>']
    sources = [example["conversations"] for example in raw_data]
    ret = preprocess(sources, tokenizer)
    for key in ret.keys():
        ret[key] = ret[key].tolist()
    print('labels', tokenizer.decode([_ if _!= -100 else tokenizer.pad_token_id for _ in ret['labels'][0]]))
    print('input', tokenizer.decode([_ if _ != -100 else tokenizer.pad_token_id for _ in ret['input_ids'][0]]))
    # with open("test_data.json", 'w', encoding='utf8') as f:
    #     f.write(json.dumps(ret, ensure_ascii=False, indent=2))

def test_InterVL_preprocess():
    # ## 文件中读取对话数据
    # path = "path/to/conversation/data"
    # with open(path, 'r', encoding='utf8') as f:
    #     raw_data = json.load(f)
    #     # raw_data[0]["conversations"] = raw_data[0]["conversations"] + raw_data[1]["conversations"] + raw_data[2]["conversations"]
    #     # raw_data = raw_data[:1]
    #     # print(len(raw_data[0]["conversations"]))
    #     # print(raw_data)
    #######

    ## 默认数据
    raw_data = default_test_convs[0:1]
    #######

    tokenizer = AutoTokenizer.from_pretrained('/Users/admin/Desktop/models/Mini-InternVL-Chat-4B-V1-5', trust_remote_code=True, model_max_length=2048)
    from fastchat.model.model_adapter import get_conv_template, get_conversation_template
    from fastchat.conversation import SeparatorStyle, register_conv_template, Conversation

    special_tokens_dict = {'additional_special_tokens': ['<|user|>', '<|assistant|>']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
    print(isinstance(tokenizer, (PreTrainedTokenizerBase, SequenceFeatureExtractor)))

    # register_conv_template(
    #     Conversation(
    #         name='phi3-chat',
    #         system_template='<|system|>\n{system_message}<|end|>',
    #         system_message='You are an AI assistant whose name is Phi-3.',
    #         roles=('<|user|>\n', '<|assistant|>\n'),
    #         sep_style=SeparatorStyle.NO_COLON_SINGLE,
    #         sep='<|end|>',
    #         stop_token_ids=[
    #             2,
    #             32000,
    #             32007
    #         ]
    #     )
    # )
    class data_args:
        num_image_token = 256
        max_image_num = 2
    conversation_template = get_conv_template('phi3-chat')
    print(conversation_template.name)
    # tokenizer.eos_token_id = tokenizer.eod_id
    # tokenizer.pad_token_id = tokenizer.special_tokens['<|extra_0|>']
    sources = [example["conversations"] for example in raw_data]
    ret = preprocess(sources, tokenizer, conversation_template, data_args)
    for key in ret.keys():
        ret[key] = ret[key]
    print('labels', tokenizer.decode([_ if _!= -100 else tokenizer.pad_token_id for _ in ret['labels'][0]]))
    print('='*100)
    print('input', tokenizer.decode([_ if _ != -100 else tokenizer.pad_token_id for _ in ret['input_ids'][0]]))

    print(ret['input_ids'][0][:200])
    # with open("test_data.json", 'w', encoding='utf8') as f:
    #     f.write(json.dumps(ret, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # test_preprocess()
    test_InterVL_preprocess()

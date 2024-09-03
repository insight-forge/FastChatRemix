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
                "value": "����һ�����õ��˹�ֻ�����֣�"
            },
            {
                "from": "human",
                "value": "С�ţ������˽������Ľ�����ϵ�����ܽ���һ����"
            },
            {
                "from": "gpt",
                "value": "��������Ȼ���ԡ������Ľ�����ϵ��һ���ɸ����ݶ�������������ƶȣ������ݶ����Լ��Ľ������ߺͷ��档Ȼ����������ݶ�����ѭ���������Ľ�����׼�Ϳγ̣���Щ��׼�������������ƶ������¡������Ľ���ϵͳ���Է�Ϊ���¼����׶Σ�\n\n1. �׶�������ͨ����3�꿪ʼ����5��6��������׶�����ͨ��������Ϸ�����֡������ͻ�����ѧ������ѧϰ�ȷ��档\n2. ����������ͨ����6�꿪ʼ��������18�꣬����Сѧ�����к͸��С�������׶Σ�ѧ����Ҫѧϰ���Ŀ�Ŀ����Ӣ���ѧ����ѧ������о������ڸ��н׶�ѡ������ѡ�޿γ̡�\n3. �ߵȽ�����������ѧ��ְҵ����ѧԺ��ͨ����18���ʼ��������׶Σ�ѧ������ѡ����ָ�����רҵ��������ҵ��ҽѧ�����ɡ���ѧ�������ȡ�\n\n�����Ľ�����ϵע������ѧ���Ĵ�������������˼ά�ͽ����������������ѧУ�ṩ�������Ŀγ̺ͻ���Ա�ѧ���ܹ����ȫ��Ľ����͸��˷�չ��ͬʱ������Ҳ����������������һ����Ҫ��������ÿ�궼��Ͷ������ľ��Ѻ���Դ����ȷ������ѧ�����ܻ�ù�ƽ�͸������Ľ�����"
            },
            {
                "from": "human",
                "value": "С�ţ��������������ܶ�������������������Ĵ�ͳ�˶���"
            },
            {
                "from": "gpt",
                "value": "�������ǵģ��������������һ�ͳ�˶�������Դ��Ӣ������������������19���ʹ������������������������ѧУ�ʹ�ѧУ԰�ڷǳ����У�������һ��ܹ�ע��ְҵ�˶���ÿ��ĳ�������������������һ��ʢ�£�������ȫ�����ڹ��ڵĹ�ע�����������������֮�⣬�������кܶ������Ĵ�ͳ�˶����������򡢰������ʽ����ȡ�ÿ���˶���������صĹ�����ص㣬��������������˵�ϲ����֧�֡�"
            },
            {
                "from": "human",
                "value": "����˵������ʳ������ܶ࣬���ܽ���һЩ��������ɫʳƷ��"
            },
            {
                "from": "gpt",
                "value": "��������Ȼ���ԡ�������һ����Ԫ�Ļ��Ĺ��ң���ʳƷ����Ҳ�ǳ��ḻ������������һЩ��������ɫʳƷ��\n\n1. ������������һ�־�����������ʳƷ���������������ɣ�ͨ��������ϸ������ϣ��������ˡ����ѡ���ƹϡ���к�֥ʿ�ȡ�\n2. ���⣺������������ͥ��⿵ĳ�����ʽ���������ļ����л����տ���������Ŀ���ʳ�İ���ţ�š����š����⡢�㳦�ͺ����ȡ�\n3. ը����ը���������ϲ���һ�ִ�ͳʳƷ�������ִ����Ƥ�Ͷ�֭�����ʶ�������ͨ�����б��ɡ����Ż������ȡ�\n4. ��ʽ��ͣ���ʽ��Ͱ����嵰���������������㳦�ͼ���ȣ�ͨ�������Կ��Ȼ��֭��\n5. �ȹ����ȹ���һ�ֿ��ʳƷ���ɿ�������㳦��ɣ������н�ĩ�����ѽ�����Ƭ�ȡ�\n\n��Ȼ����ֻ������ʳƷ�е�һС���֣�ÿ���ݺ͵�����������ص�ʳƷ�Ļ���������������ʱ���Բ�ͬ��ʳƷ��������������ʳ�Ļ���"
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
    # ## �ļ��ж�ȡ�Ի�����
    # path = "path/to/conversation/data"
    # with open(path, 'r', encoding='utf8') as f:
    #     raw_data = json.load(f)
    #     # raw_data[0]["conversations"] = raw_data[0]["conversations"] + raw_data[1]["conversations"] + raw_data[2]["conversations"]
    #     # raw_data = raw_data[:1]
    #     # print(len(raw_data[0]["conversations"]))
    #     # print(raw_data)
    #######
    ## Ĭ������
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
    # ## �ļ��ж�ȡ�Ի�����
    # path = "path/to/conversation/data"
    # with open(path, 'r', encoding='utf8') as f:
    #     raw_data = json.load(f)
    #     # raw_data[0]["conversations"] = raw_data[0]["conversations"] + raw_data[1]["conversations"] + raw_data[2]["conversations"]
    #     # raw_data = raw_data[:1]
    #     # print(len(raw_data[0]["conversations"]))
    #     # print(raw_data)
    #######

    ## Ĭ������
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

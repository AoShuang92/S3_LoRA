# Basic LLM node that calls for a Large Language Model for completion.
import os
import torch
import openai
import re
from nodes.Node import Node
from nodes.NodeCofig import *
from utils.util import *
from alpaca.lora import AlpacaLora

from datetime import datetime
import json
from pathlib import Path
import os

from pytz import timezone

import random
import time
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastchat.conversation import Conversation, SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
import requests
from requests.exceptions import Timeout
from tokenizers import Tokenizer
import urllib3

urllib3.disable_warnings()

def get_time() -> str:
    return datetime.now(timezone('Asia/Shanghai')).strftime('%m%d%H%M%S')

MODEL = os.environ['MODEL']
METHOD = os.environ['METHOD']
TASK = os.environ['TASK']
NOW = get_time()
ID = f'{TASK}_{NOW}'
LOG = Path('logs') / f'{MODEL}-{METHOD}' / TASK / f'{NOW}.json'

os.makedirs(Path('logs') / f'{MODEL}-{METHOD}' / TASK, exist_ok=True)

traj = []

tokenizer = AutoTokenizer.from_pretrained('THUDM/agentlm-70b')
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/agentlm-7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

def token_count(text):
    return len(tokenizer.encode(text))

def get_prompt(conv: Conversation) -> str:
    if conv.name == 'openchat':
        ret = ''
        for role, message in conv.messages:
            if message:
                ret += role + ": " + message.strip() + conv.sep
            else:
                ret += role + ":"
        return ret
    else:
        return conv.get_prompt()

# def llm_llama(prompt: str) -> str:
#     CONTROLLER_ADDR = os.environ['CONTROLLER_ADDR'].split(',')
#     data = {
#         "inputs": prompt,
#         "parameters": {
#             "max_new_tokens": 256,
#             "do_sample": True,
#             'temperature': 0.5,
#             'frequency_penalty': 0,
#             'presence_penalty': 0,
#             'truncate': 4000,
#         }
#     }
#     if True or os.getenv('GREEDY'):
#         data['parameters']['do_sample'] = False
#         data['parameters'].pop('temperature')
#         print('greedy mode enabled')
#     for _ in range(5):
#         try:
#             response = requests.post(
#                 random.choice(CONTROLLER_ADDR) + "/generate",
#                 json=data,
#                 timeout=120,
#                 proxies={'http': '', 'https': ''},
#             )
#             print(response.content)
#             text = response.json()["generated_text"]
#             print(text)
#             return text.split('[INST]')[0].split('<|end_of_turn|>')[0].strip()
#         # if timeout or connection error, retry
#         except Timeout: 
#             print("Timeout, retrying...")
#         except ConnectionError:
#             print("Connection error, retrying...")
#         except Exception:
#             traceback.print_exc()
#             try:
#                 print(response)
#                 print(response.text)
#             except:
#                 pass
#         time.sleep(5)
#     else:
#         raise Exception("Timeout after 5 retries.")
    
def llm_llama(prompt: str) -> str:
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.5,
            "frequency_penalty": 0,       # not supported directly
            "presence_penalty": 0,        # not supported directly
            "truncate": 4000              # handled manually if needed
        }
    }

    # Greedy mode override (as in original code)
    if True or os.getenv('GREEDY'):
        data['parameters']['do_sample'] = False
        data['parameters'].pop('temperature', None)
        print('greedy mode enabled')

    try:
        prompt = data["inputs"]
        params = data["parameters"]

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=params.get("truncate", 4000)).to(model.device)

        # Build generation kwargs
        generation_kwargs = {
            "max_new_tokens": params["max_new_tokens"],
            "do_sample": params["do_sample"],
            "pad_token_id": tokenizer.eos_token_id,
        }
        if params.get("temperature") is not None:
            generation_kwargs["temperature"] = params["temperature"]

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        # Decode and clean
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print("final generation plann", text)
        
        plan_temp = re.search(r"\[/INST\](.*?)(?=\nQuestion:|$)", text, re.DOTALL).group(1).strip()
        # print("after split plan", plan_temp)
        # return text.split('[INST]')[0].split('<|end_of_turn|>')[0].strip() 
        return plan_temp

    except Exception as e:
        print("Error:", e)
        raise

# Refresh traj log
def refresh(label: str):
    global NOW, ID, LOG, traj
    # print("Now", NOW, label)
    # print("lognamefile", LOG.rename(LOG.with_name(f'{NOW}_{label}.json')))
    LOG.rename(LOG.with_name(f'{NOW}_{label}.json'))
    NOW = get_time()
    ID = f'{TASK}_{NOW}'
    LOG = Path('logs') / f'{MODEL}-{METHOD}' / TASK / f'{NOW}.json'
    traj = []

class LLMNode(Node):
    def __init__(self, name="BaseLLMNode", model_name="THUDM/agentlm-7b", stop=None, input_type=str, output_type=str):
        super().__init__(name, input_type, output_type)
        self.model_name = model_name
        self.stop = stop

        # Initialize to load shards only once
        if self.model_name in LLAMA_WEIGHTS:
            self.al = AlpacaLora(lora_weights=self.model_name)

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        response = self.call_llm(input, self.stop)
        completion = response["output"]
        if log:
            return response
        return completion

    def call_llm(self, prompt, stop):
        # print("self.model_name", self.model_name)
        if self.model_name in OPENAI_COMPLETION_MODELS:
            response = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                temperature=OPENAI_CONFIG["temperature"],
                max_tokens=OPENAI_CONFIG["max_tokens"],
                top_p=OPENAI_CONFIG["top_p"],
                frequency_penalty=OPENAI_CONFIG["frequency_penalty"],
                presence_penalty=OPENAI_CONFIG["presence_penalty"],
                stop=stop
            )
            return {"input": prompt,
                    "output": response["choices"][0]["text"],
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "completion_tokens": response["usage"]["completion_tokens"]}
        elif self.model_name in OPENAI_CHAT_MODELS:
            print('*****GPT-4*****')
            messages = [{"role": "user", "content": prompt}]
            while True:
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=messages,
                        # temperature=OPENAI_CONFIG["temperature"],
                        temperature=0.0,
                        max_tokens=OPENAI_CONFIG["max_tokens"],
                        # top_p=OPENAI_CONFIG["top_p"],
                        # frequency_penalty=OPENAI_CONFIG["frequency_penalty"],
                        # presence_penalty=OPENAI_CONFIG["presence_penalty"],
                        # stop=stop
                    )
                    break
                except:
                    continue

            traj.append({
                'id': f'{ID}_{len(traj)}',
                'conversations': [
                    {
                        'from': 'human',
                        'value': prompt,
                    },
                    {
                        'from': 'gpt',
                        'value': response["choices"][0]["message"]["content"],
                    },
                ]
            })
            json.dump(traj, open(LOG, 'w'))

            return {"input": prompt,
                    "output": response["choices"][0]["message"]["content"],
                    "prompt_tokens": response["usage"]["prompt_tokens"],
                    "completion_tokens": response["usage"]["completion_tokens"]}
        elif self.model_name in LLAMA_WEIGHTS:
            print("************LoRA***********")
            instruction, input = prompt[0], prompt[1]
            # print("instruction", instruction)
            # print("input", input) only the question
            output, prompt = self.al.lora_generate(instruction, input)
            return {"input": prompt,
                    "output": output,
                    "prompt_tokens": len(prompt)/4,
                    "completion_tokens": len(output)/4
            }
        
        elif 'llama' in self.model_name or 'openchat' in self.model_name or 'vicuna' in self.model_name or "agentlm" in self.model_name:
            print('*****LLAMA2*****')
            if 'llama' in self.model_name or "agentlm" in self.model_name:
                conv = get_conversation_template('llama-2')
                conv.set_system_message("You are a helpful, respectful and honest assistant.")
            elif 'vicuna' in self.model_name:
                conv = get_conversation_template('vicuna')
            elif 'openchat' in self.model_name:
                conv = Conversation(
                    name="openchat",
                    roles=("GPT4 User", "GPT4 Assistant"),
                    messages=[],
                    offset=0,
                    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
                    sep="<|end_of_turn|>",
                )
            else:
                raise NotImplementedError
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            output = llm_llama(get_prompt(conv))

            traj.append({
                'id': f'{ID}_{len(traj)}',
                'conversations': [
                    {
                        'from': 'human',
                        'value': prompt,
                    },
                    {
                        'from': 'gpt',
                        'value': output,
                    },
                ]
            })
            json.dump(traj, open(LOG, 'w'))

            return {
                'input': prompt,
                'output': output,
                'prompt_tokens': token_count(prompt),
                'completion_tokens': token_count(output),
            }
        else:
            raise ValueError("Model not supported")

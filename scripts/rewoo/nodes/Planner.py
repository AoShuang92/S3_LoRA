from nodes.LLMNode import LLMNode
from nodes.Worker import WORKER_REGISTRY
from prompts.planner import *
from utils.util import LLAMA_WEIGHTS
import re

class Planner(LLMNode):
    def __init__(self, workers, prefix=DEFAULT_PREFIX, suffix=DEFAULT_SUFFIX, fewshot=DEFAULT_FEWSHOT,
                 model_name="THUDM/agentlm-7b", stop=None):
        super().__init__("Planner", model_name, stop, input_type=str, output_type=str)
        self.workers = workers
        self.prefix = prefix
        self.worker_prompt = self._generate_worker_prompt()
        self.suffix = suffix
        self.fewshot = fewshot

    def run(self, input, log=False):
        assert isinstance(input, self.input_type)
        # print("self.fewshot", self.fewshot)
        # print("self.prefix", self.prefix)
        # print("self.worker_prompt", self.worker_prompt)
        # print("self.suffix", self.suffix)
        # print("LLAMA_WEIGHTS", LLAMA_WEIGHTS)

        prompt = self.prefix + self.worker_prompt + self.fewshot + self.suffix + input + '\n'
        # prompt = self.prefix + self.worker_prompt + self.suffix + input + '\n'
        # print("prompt", prompt)
        if self.model_name in LLAMA_WEIGHTS:
            # prompt = [self.prefix + self.worker_prompt, input]
            prompt = [self.prefix + self.worker_prompt +  self.suffix, input]
        response = self.call_llm(prompt, self.stop)
        # print("response", response)
        completion = response["output"]
        if log:
            return response
        return completion

    def _get_worker(self, name):
        # print("name", name) Wikipedia[@]
        name = re.sub(r"\[\@.*?\]", "", name)
        if name in WORKER_REGISTRY:
            return WORKER_REGISTRY[name]
        else:
            raise ValueError("Worker not found")

    def _generate_worker_prompt(self):
        prompt = "Tools can be one of the following:\n"
        for name in self.workers:
            worker = self._get_worker(name)
            prompt += f"{worker.name}[input]: {worker.description}\n"
        return prompt + "\n"

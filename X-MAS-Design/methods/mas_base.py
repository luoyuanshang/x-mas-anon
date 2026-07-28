import json
import os
import random
import openai
import requests
from tenacity import retry, wait_exponential, stop_after_attempt

from .utils import handle_retry_error

class MAS():

    def __init__(self, general_config, method_config_name=None):
        
        self.model_api_config = general_config["model_api_config"]
        self.model_name = general_config["model_name"]
        self.model_temperature = general_config["model_temperature"]
        self.model_max_tokens = general_config["model_max_tokens"]
        self.model_timeout = general_config["model_timeout"]
        
        # Tracking compute costs
        self.token_stats = {
            self.model_name: {"num_llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
        }

        self.memory_bank = {}
        self.tools = {}
        self.call_history = []
        
    
    def inference(self, query):
        """
        query: Query to be passed to the MAS
        """
        response = self.call_llm(prompt=query)
        return response

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5), retry_error_callback=handle_retry_error)
    def call_llm(self, prompt=None, system_prompt=None, messages=None, model_name=None, temperature=None):
        model_name = model_name if model_name is not None else self.model_name
        try:
            model_dict = random.choice(self.model_api_config[model_name]["model_list"])
        except Exception as e:
            print(f"🔥 Failed to select model_dict for {model_name}: {e}")
            raise
        # print(f"model_dict of {model_name}: {model_dict}")
        model_name, model_url, api_key = model_dict['model_name'], model_dict['model_url'], model_dict['api_key']
        
        if messages is None:
            assert prompt is not None, "'prompt' must be provided if 'messages' is not provided."
            if system_prompt is not None:
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            else:
                messages = [{"role": "user", "content": prompt}]
        
        model_temperature = temperature if temperature is not None else self.model_temperature
        model_max_tokens = self.model_max_tokens

        request_dict = {
            "model": model_name,
            "messages": messages,
            "max_tokens": model_max_tokens,
            "timeout": self.model_timeout
        }
        if "o1" not in model_name:              # OpenAI's o1 models do not support temperature
            request_dict["temperature"] = model_temperature
        
        llm = openai.OpenAI(base_url=model_url, api_key=api_key)
        try:
            completion = llm.chat.completions.create(**request_dict)
            message = completion.choices[0].message
            raw_response = message.content or ""
            reasoning_content = getattr(message, "reasoning_content", None)
            response = raw_response
            num_prompt_tokens = getattr(completion.usage, "prompt_tokens", None) or 0
            num_completion_tokens = getattr(completion.usage, "completion_tokens", None) or 0
            finish_reason = completion.choices[0].finish_reason
        finally:
            llm.close()

        protocol_status = "ok"
        if finish_reason in {"length", "max_tokens"}:
            protocol_status = "token_truncation"
        if "r1" in model_name.lower() or "qwq" in model_name.lower():
            if "<think>" in response and "</think>" not in response:
                if protocol_status == "ok":
                    protocol_status = "unclosed_think"
            elif "</think>" in response:
                response = response.split("</think>")[-1].strip()
        elif "openthinker" in model_name:
            if '<|begin_of_solution|>' in response and '<|end_of_solution|>' in response:
                response = response.split('<|begin_of_solution|>')[-1].split('<|end_of_solution|>')[0].strip()
            else:
                protocol_status = "missing_solution_marker"
        elif "huatuo" in model_name:
            if '## Final Response' in response:
                response = response.split('## Final Response')[-1].strip()
            else:
                protocol_status = "missing_final_response_marker"

        if not response and protocol_status == "ok":
            protocol_status = "empty_final_response"

        self.call_history.append({
            "model_name": model_name,
            "raw_response": raw_response,
            "reasoning_content": reasoning_content,
            "final_response": response,
            "finish_reason": finish_reason,
            "protocol_status": protocol_status,
            "num_prompt_tokens": num_prompt_tokens,
            "num_completion_tokens": num_completion_tokens,
        })

        if isinstance(response, str):       # in cases where response is None or an error message
            if model_name not in self.token_stats:
                self.token_stats[model_name] = {"num_llm_calls": 1, "prompt_tokens": num_prompt_tokens, "completion_tokens": num_completion_tokens}
            else:
                self.token_stats[model_name]["num_llm_calls"] += 1
                self.token_stats[model_name]["prompt_tokens"] += num_prompt_tokens
                self.token_stats[model_name]["completion_tokens"] += num_completion_tokens
        else:
            raise ValueError(f"Invalid response from LLM: {response}")
        
        return response

    def get_token_stats(self):
        return self.token_stats

    def get_call_history(self):
        return self.call_history
    
    def optimizing(self, val_data):
        """
        For methods that requires validation data such as GPTSwarm and ADAS
        """
        pass

    def retrieve_memory(self):
        pass

    def update_memory(self):
        pass
    
    def get_tool(self):
        pass

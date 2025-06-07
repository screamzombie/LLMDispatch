# -*- coding: utf-8 -*-
import os
import re
import requests
from abc import ABC, abstractmethod
from openai import OpenAI
# from LLMHandle.config import DEEPSEEK_API_KEY, QWEN_API_KEY, DOUBAO_API_KEY, XFYUN_API_KEY
from LLMDispatch.LLMHandle import config
from LLMDispatch.LLMHandle.LLMWorker.PromptLoader import load_prompt
from dataclasses import dataclass


# --- 抽象基类 ---
@dataclass
class BaseMindMapModelAPI(ABC):
    temperature: float = 0.7 # 默认温度是0.7
    @abstractmethod
    def change_temperature(self, temperature: float): # 改变温度
        self.temperature = temperature
    
    @abstractmethod
    def generate_code(self, query: str) -> str: # 生成代码
        pass
    
    @abstractmethod
    def postprocess_code(self) -> str: # 后处理代码
        pass

    @abstractmethod
    def execute(self, query: str,temperature:float) -> str: # 执行
        pass
    
    @abstractmethod
    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        pass

    @abstractmethod
    def get_prompt(self) -> dict: # 获取提示词
        pass        

    @abstractmethod
    def reload_prompt(self,role:str="mindmap"):
        pass


# --- XunFei MindMap Model API ---
class XunFeiMindMapModelAPI(BaseMindMapModelAPI):
    def __init__(self,
                 api_key: str = None,
                 base_url: str = "https://spark-api-open.xf-yun.com/v2",  # 使用兼容OpenAI的V2地址
                 model: str = "x1",  # V2接口使用的模型标识符为 "x1"
                 role: str = "mindmap",
                 temperature: float = 0.5):
        super().__init__(temperature=temperature)
        # 注意：讯飞V2接口的API Key格式为 "APIKey:APISecret" 或直接使用控制台的 "APIPassword"
        self.api_key = api_key if api_key is not None else config.XUNFEI_API_KEY
        self.base_url = base_url
        self.model = model
        self.role = role
        self.prompt_config = load_prompt(role)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_code(self, query: str, temperature: float = None) -> str:  # 默认情况下不需要传入温度，模型中已经设置了默认值
        if temperature is None:
            temperature = self.temperature
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt_config["system"]},
                {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
            ],
            temperature=temperature,
            stream=False
        )
        return response.choices[0].message.content.strip()

    def postprocess_code(self, code: str) -> str:
        if code.startswith("```mermaid\n"):
            code = code[len("```mermaid\n"):]
        if code.endswith("\n```"):
            code = code[:-len("\n```")]
        elif code.endswith("```"):
            code = code[:-len("```")]
        return code.strip()

    def execute(self, query: str, temperature: float = None) -> str:
        if temperature is None:
            temperature = self.temperature
        code = self.generate_code(query, temperature)
        processed_code = self.postprocess_code(code)
        return processed_code

    def change_temperature(self, temperature: float):
        self.temperature = temperature

    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        if mode == "replace":
            self.prompt_config["user_prefix"] = user_prompt
        elif mode == "append":
            self.prompt_config["user_prefix"] += "\n" + user_prompt
        else:
            raise ValueError("mode 只能是 'replace' 或 'append'")

    def get_prompt(self) -> dict:
        return {
            "system": self.prompt_config.get("system", ""),
            "user_prefix": self.prompt_config.get("user_prefix", "")
        }

    def reload_prompt(self, role: str = "mindmap"):
        self.prompt_config = load_prompt(role)  # 重新加载提示词


# --- Deepseek MindMap Model API ---
class DeepseekMindMapModelAPI(BaseMindMapModelAPI):
    def __init__(self,
                 api_key: str = None,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat",
                 role: str = "mindmap",
                 temperature: float = 0.7): 
        
        super().__init__(temperature=temperature)
        self.api_key = api_key if api_key is not None else config.DEEPSEEK_API_KEY
        # self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_config = load_prompt(role)  # 加载提示词
        self.temperature = temperature

    def generate_code(self, query: str, temperature: float = None) -> str: # 默认情况下不需要传入温度，模型中已经设置了默认值
        if temperature is None:
            temperature = self.temperature
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt_config["system"]},
                {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
            ],
            temperature=temperature,
            stream=False
        )
        return response.choices[0].message.content.strip()

    def postprocess_code(self, code: str) -> str:        
        if code.startswith("```mermaid\n"):
            code = code[len("```mermaid\n"):]        
        if code.endswith("\n```"):
            code = code[:-len("\n```")]    
        elif code.endswith("```"):
            code = code[:-len("```")]
        return code.strip() 

    def execute(self, query: str, temperature: float = None) -> str:
        if temperature is None:
            temperature = self.temperature
        code = self.generate_code(query, temperature)
        processed_code = self.postprocess_code(code)
        return processed_code

    def change_temperature(self, temperature: float):
        self.temperature = temperature

    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        if mode == "replace":
            self.prompt_config["user_prefix"] = user_prompt
        elif mode == "append":
            self.prompt_config["user_prefix"] += "\n" + user_prompt
        else:
            raise ValueError("mode 只能是 'replace' 或 'append'")

    def get_prompt(self) -> dict:
        return {
            "system": self.prompt_config.get("system", ""),
            "user_prefix": self.prompt_config.get("user_prefix", "")
        }

    def reload_prompt(self,role:str="mindmap"):
        self.prompt_config = load_prompt(role)  # 重新加载提示词


# --- Doubao MindMap Model API ---
class DoubaoMindMapModelAPI(BaseMindMapModelAPI):
    def __init__(self,
                 api_key: str = None,
                 base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
                 model: str = "doubao-1-5-thinking-pro-250415",
                 role: str = "mindmap",
                 temperature: float = 0.7):

        super().__init__(temperature=temperature)
        self.api_key = api_key if api_key is not None else config.DOUBAO_API_KEY
        # self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_config = load_prompt(role)
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_code(self, query: str, temperature: float = None) -> str:
        if temperature is None:
            temperature = self.temperature
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt_config["system"]},
                {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
            ],
            temperature=temperature,
            stream=False
        )
        return response.choices[0].message.content.strip()

    def postprocess_code(self, code: str) -> str:
        if code.startswith("```mermaid\n"):
            code = code[len("```mermaid\n"):]
        if code.endswith("\n```"):
            code = code[:-len("\n```")]
        elif code.endswith("```"):
            code = code[:-len("```")]
        return code.strip()

    def execute(self, query: str, temperature: float = None) -> str:
        if temperature is None:
            temperature = self.temperature
        code = self.generate_code(query, temperature)
        return self.postprocess_code(code)

    def change_temperature(self, temperature: float):
        self.temperature = temperature

    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        if mode == "replace":
            self.prompt_config["user_prefix"] = user_prompt
        elif mode == "append":
            self.prompt_config["user_prefix"] += "\n" + user_prompt
        else:
            raise ValueError("mode 只能是 'replace' 或 'append'")

    def get_prompt(self) -> dict:
        return {
            "system": self.prompt_config.get("system", ""),
            "user_prefix": self.prompt_config.get("user_prefix", "")
        }

    def reload_prompt(self, role: str = "mindmap"):
        self.prompt_config = load_prompt(role)


# --- Qwen MindMap Model API ---
class QwenMindMapModelAPI(BaseMindMapModelAPI):
    def __init__(self,
                 api_key: str = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model: str = "qwen3-235b-a22b",
                 role: str = "mindmap",
                 temperature: float = 0.7):

        super().__init__(temperature=temperature)
        self.api_key = api_key if api_key is not None else config.QWEN_API_KEY
        # self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_config = load_prompt(role)
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_code(self, query: str, temperature: float = None) -> str:
        if temperature is None:
            temperature = self.temperature
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt_config.get("system", "")},
                {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
            ],
            temperature=temperature,
            extra_body={"enable_thinking": False},
            stream=False
        )
        return response.choices[0].message.content.strip()

    def postprocess_code(self, code: str) -> str:
        if code.startswith("```mermaid\n"):
            code = code[len("```mermaid\n"):]
        if code.endswith("\n```"):
            code = code[:-len("\n```")]
        elif code.endswith("```"):
            code = code[:-len("```")]
        return code.strip()

    def execute(self, query: str, temperature: float = None) -> str:
        if temperature is None:
            temperature = self.temperature
        code = self.generate_code(query, temperature)
        return self.postprocess_code(code)

    def change_temperature(self, temperature: float):
        self.temperature = temperature

    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        if mode == "replace":
            self.prompt_config["user_prefix"] = user_prompt
        elif mode == "append":
            self.prompt_config["user_prefix"] += "\n" + user_prompt
        else:
            raise ValueError("mode 只能是 'replace' 或 'append'")

    def get_prompt(self) -> dict:
        return {
            "system": self.prompt_config.get("system", ""),
            "user_prefix": self.prompt_config.get("user_prefix", "")
        }

    def reload_prompt(self, role: str = "mindmap"):
        self.prompt_config = load_prompt(role)


# --- 总调度类 ---
class MindMapGenerationManager:
    _registry = {
        "xunfei": XunFeiMindMapModelAPI,
        "deepseek": DeepseekMindMapModelAPI,
        "doubao": DoubaoMindMapModelAPI,
        "qwen": QwenMindMapModelAPI       
    }

    def __init__(self, use_api: str = "deepseek", role: str = "mindmap"):
        if use_api not in self._registry:
            raise ValueError(f"不支持的 API: {use_api}")
        self.use_api = use_api
        self.client = self._registry[use_api](role=role)

    def execute(self, query: str) -> str:
        return self.client.execute(query)


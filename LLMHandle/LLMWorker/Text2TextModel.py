# -*- coding: utf-8 -*-
import os
import re
import requests
from abc import ABC, abstractmethod
from openai import OpenAI
from .config import DEEPSEEK_API_KEY, QWEN_API_KEY, DOUBAO_API_KEY, XFYUN_API_KEY
from .prompt_loader import load_prompt 


# --- 抽象基类 ---
class BaseTextModelAPI(ABC):
    @abstractmethod
    def generate_text(self, query: str, **kwargs) -> str:
        pass

# --- 讯飞 API 封装 ---
class XunfeiTextModelAPI(BaseTextModelAPI):
    def __init__(self,
                 api_key: str = XFYUN_API_KEY,
                 url: str = "https://spark-api-open.xf-yun.com/v1/chat/completions",
                 model: str = "4.0Ultra",
                 max_tokens: int = 4096,
                 role: str = "summarizer"):
        self.api_key = api_key
        self.url = url
        self.model = model
        self.max_tokens = max_tokens
        self.prompt_config = load_prompt(role)

    def generate_text(self, query: str, temperature: float = 0.5, **kwargs) -> str:
        headers = {"Authorization": self.api_key}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.prompt_config["system"]},
                {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
            ],
            "temperature": temperature,
            "top_k": 4,
            "max_tokens": self.max_tokens,
            "stream": False
        }

        response = requests.post(self.url, headers=headers, json=payload)
        response.encoding = "utf-8"        
        result = response.text
        match = re.search(r'"content":"(.*?)"', result)
        result = match.group(1).replace("\\n", "\n")        
        return result if match else "No content found"


# --- DeepSeek 封装 ---
class DeepseekTextModelAPI(BaseTextModelAPI):
    def __init__(self,
                 api_key: str = DEEPSEEK_API_KEY,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat",
                 role: str = "summarizer"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_config = load_prompt(role)

    def generate_text(self, query: str, temperature: float = 0.7, **kwargs) -> str:
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


# --- Qwen 封装 ---
class QwenTextModelAPI(BaseTextModelAPI):
    def __init__(self,
                 api_key: str = QWEN_API_KEY,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model: str = "qwen-plus",
                 role: str = "summarizer"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_config = load_prompt(role)

    def generate_text(self, query: str, temperature: float = 1.0, **kwargs) -> str:
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


# --- 豆包封装 ---
class DoubaoTextModelAPI(BaseTextModelAPI):
    def __init__(self,
                 api_key: str = DOUBAO_API_KEY,
                 base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
                 model: str = "doubao-1-5-thinking-pro-250415",
                 role: str = "summarizer"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.prompt_config = load_prompt(role)

    def generate_text(self, query: str, temperature: float = 0.7, **kwargs) -> str:
        messages = []
        if self.prompt_config.get("system"):
            messages.append({"role": "system", "content": self.prompt_config["system"]})
        messages.append({"role": "user", "content": self.prompt_config.get("user_prefix", "") + query})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False,
            **kwargs
        )
        return response.choices[0].message.content.strip()


# --- 总调度类 ---
class TextGenerationManager:
    _registry = {
        "deepseek": DeepseekTextModelAPI,
        "qwen": QwenTextModelAPI,
        "doubao": DoubaoTextModelAPI,
        "xunfei": XunfeiTextModelAPI,
    }

    def __init__(self, use_api: str = "deepseek", role: str = "summarizer"):
        if use_api not in self._registry:
            raise ValueError(f"不支持的 API: {use_api}")
        self.use_api = use_api
        self.client = self._registry[use_api](role=role)

    def generate_text(self, query: str, **kwargs) -> str:
        return self.client.generate_text(query, **kwargs)
# -*- coding: utf-8 -*-
import os
import re
import requests
from abc import ABC, abstractmethod
from openai import OpenAI
from LLMHandle.config import DEEPSEEK_API_KEY, QWEN_API_KEY, DOUBAO_API_KEY, XFYUN_API_KEY
from LLMHandle.LLMWorker.PromptLoader import load_prompt
from dataclasses import dataclass


# --- 抽象基类 ---
@dataclass
class BaseTextModelAPI(ABC):
    temperature: float = 0.7 # 默认温度

    @abstractmethod
    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        pass

    @abstractmethod
    def change_temperature(self, temperature: float):
        self.temperature = temperature

    @abstractmethod
    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        pass

    @abstractmethod
    def get_prompt(self) -> dict:
        pass

    @abstractmethod
    def reload_prompt(self, role: str):
        pass


# --- 讯飞 API 封装 ---
class XunfeiTextModelAPI(BaseTextModelAPI):
    def __init__(self,
                 api_key: str = XFYUN_API_KEY,
                 url: str = "https://spark-api-open.xf-yun.com/v1/chat/completions",
                 model: str = "4.0Ultra",
                 max_tokens: int = 4096,
                 role: str = "summarizer",
                 temperature: float = 0.5):
        super().__init__(temperature=temperature)
        self.api_key = api_key
        self.url = url
        self.model = model
        self.max_tokens = max_tokens
        self.role = role
        self.prompt_config = load_prompt(role)

    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        current_temp = temperature if temperature is not None else self.temperature
        headers = {"Authorization": self.api_key}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.prompt_config.get("system", "")},
                {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
            ],
            "temperature": current_temp,
            "top_k": kwargs.get("top_k", 4),
            "max_tokens": self.max_tokens,
            "stream": False
        }

        response = requests.post(self.url, headers=headers, json=payload)
        response.encoding = "utf-8"
        result = response.text
        try:
            # 尝试解析 JSON 获取 content
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", None)
            if content:
                return content.strip()
            # 如果 JSON 解析失败或未找到 content，尝试正则匹配 (保持兼容性)
            match = re.search(r'"content":"(.*?)"', result)
            if match:
                return match.group(1).replace("\\n", "\n").strip()
            return "No content found in response"
        except requests.exceptions.JSONDecodeError:
            # JSON 解析失败，回退到正则匹配
            match = re.search(r'"content":"(.*?)"', result)
            if match:
                return match.group(1).replace("\\n", "\n").strip()
            return "Failed to parse response and no content found"

    def change_temperature(self, temperature: float):
        self.temperature = temperature

    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        if mode == "replace":
            self.prompt_config["user_prefix"] = user_prompt
        elif mode == "append":
            current_prefix = self.prompt_config.get("user_prefix", "")
            self.prompt_config["user_prefix"] = current_prefix + "\n" + user_prompt if current_prefix else user_prompt
        else:
            raise ValueError("mode 只能是 'replace' 或 'append'")

    def get_prompt(self) -> dict:
        return {
            "system": self.prompt_config.get("system", ""),
            "user_prefix": self.prompt_config.get("user_prefix", "")
        }

    def reload_prompt(self, role: str = None):
        current_role = role if role is not None else self.role
        self.prompt_config = load_prompt(current_role)
        self.role = current_role


# --- DeepSeek 封装 ---
class DeepseekTextModelAPI(BaseTextModelAPI):
    def __init__(self,
                 api_key: str = DEEPSEEK_API_KEY,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat",
                 role: str = "summarizer",
                 temperature: float = 0.7):
        super().__init__(temperature=temperature)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.role = role
        self.prompt_config = load_prompt(role)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        current_temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt_config.get("system", "")},
                {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
            ],
            temperature=current_temp,
            stream=False,
            **kwargs
        )
        return response.choices[0].message.content.strip()

    def change_temperature(self, temperature: float):
        self.temperature = temperature

    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        if mode == "replace":
            self.prompt_config["user_prefix"] = user_prompt
        elif mode == "append":
            current_prefix = self.prompt_config.get("user_prefix", "")
            self.prompt_config["user_prefix"] = current_prefix + "\n" + user_prompt if current_prefix else user_prompt
        else:
            raise ValueError("mode 只能是 'replace' 或 'append'")

    def get_prompt(self) -> dict:
        return {
            "system": self.prompt_config.get("system", ""),
            "user_prefix": self.prompt_config.get("user_prefix", "")
        }

    def reload_prompt(self, role: str = None):
        current_role = role if role is not None else self.role
        self.prompt_config = load_prompt(current_role)
        self.role = current_role


# --- Qwen 封装 ---
class QwenTextModelAPI(BaseTextModelAPI):
    def __init__(self,
                 api_key: str = QWEN_API_KEY,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model: str = "qwen-plus",
                 role: str = "summarizer",
                 temperature: float = 1.0):
        super().__init__(temperature=temperature)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.role = role
        self.prompt_config = load_prompt(role)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        current_temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt_config.get("system", "")},
                {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
            ],
            temperature=current_temp,
            stream=False,
            **kwargs
        )
        return response.choices[0].message.content.strip()

    def change_temperature(self, temperature: float):
        self.temperature = temperature

    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        if mode == "replace":
            self.prompt_config["user_prefix"] = user_prompt
        elif mode == "append":
            current_prefix = self.prompt_config.get("user_prefix", "")
            self.prompt_config["user_prefix"] = current_prefix + "\n" + user_prompt if current_prefix else user_prompt
        else:
            raise ValueError("mode 只能是 'replace' 或 'append'")

    def get_prompt(self) -> dict:
        return {
            "system": self.prompt_config.get("system", ""),
            "user_prefix": self.prompt_config.get("user_prefix", "")
        }

    def reload_prompt(self, role: str = None):
        current_role = role if role is not None else self.role
        self.prompt_config = load_prompt(current_role)
        self.role = current_role


# --- 豆包封装 ---
class DoubaoTextModelAPI(BaseTextModelAPI):
    def __init__(self,
                 api_key: str = DOUBAO_API_KEY,
                 base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
                 model: str = "doubao-1-5-thinking-pro-250415",
                 role: str = "summarizer",
                 temperature: float = 0.7):
        super().__init__(temperature=temperature)
        self.api_key = api_key # Store api_key although client uses it directly
        self.base_url = base_url
        self.model = model
        self.role = role
        self.prompt_config = load_prompt(role)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        current_temp = temperature if temperature is not None else self.temperature
        messages = []
        if self.prompt_config.get("system"): # Check if system prompt exists
            messages.append({"role": "system", "content": self.prompt_config["system"]})
        messages.append({"role": "user", "content": self.prompt_config.get("user_prefix", "") + query})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=current_temp,
            stream=False,
            **kwargs
        )
        return response.choices[0].message.content.strip()

    def change_temperature(self, temperature: float):
        self.temperature = temperature

    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        if mode == "replace":
            self.prompt_config["user_prefix"] = user_prompt
        elif mode == "append":
            current_prefix = self.prompt_config.get("user_prefix", "")
            self.prompt_config["user_prefix"] = current_prefix + "\n" + user_prompt if current_prefix else user_prompt
        else:
            raise ValueError("mode 只能是 'replace' 或 'append'")

    def get_prompt(self) -> dict:
        return {
            "system": self.prompt_config.get("system", ""),
            "user_prefix": self.prompt_config.get("user_prefix", "")
        }

    def reload_prompt(self, role: str = None):
        current_role = role if role is not None else self.role
        self.prompt_config = load_prompt(current_role)
        self.role = current_role


# --- 总调度类 ---
class TextGenerationManager:
    _registry = {
        "deepseek": DeepseekTextModelAPI,
        "qwen": QwenTextModelAPI,
        "doubao": DoubaoTextModelAPI,
        "xunfei": XunfeiTextModelAPI,
    }

    def __init__(self, use_api: str = "deepseek", role: str = "summarizer", temperature: float = None):
        if use_api not in self._registry:
            raise ValueError(f"不支持的 API: {use_api}")
        self.use_api = use_api
        # Pass temperature if provided, otherwise let the class use its default
        init_kwargs = {'role': role}
        if temperature is not None:
            init_kwargs['temperature'] = temperature
        self.client: BaseTextModelAPI = self._registry[use_api](**init_kwargs)

    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        # Pass temperature explicitly if provided, otherwise the client uses its internal state
        return self.client.generate_text(query, temperature=temperature, **kwargs)

    def change_temperature(self, temperature: float):
        self.client.change_temperature(temperature)

    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        self.client.set_custom_prompt(user_prompt, mode)

    def get_prompt(self) -> dict:
        return self.client.get_prompt()

    def reload_prompt(self, role: str = None):
        # If role is None, the client's reload_prompt will use its current role
        self.client.reload_prompt(role)

    def switch_api(self, use_api: str, role: str = None, temperature: float = None):
        if use_api not in self._registry:
            raise ValueError(f"不支持的 API: {use_api}")
        self.use_api = use_api
        current_role = role if role is not None else self.client.role
        current_temp = temperature if temperature is not None else self.client.temperature
        init_kwargs = {'role': current_role, 'temperature': current_temp}
        self.client = self._registry[use_api](**init_kwargs)
        print(f"Switched to API: {use_api} with role: {current_role} and temperature: {current_temp}")

    def get_current_config(self) -> dict:
        return {
            "api": self.use_api,
            "role": self.client.role,
            "temperature": self.client.temperature,
            "model": getattr(self.client, 'model', 'N/A'), # Get model if available
            "prompts": self.get_prompt()
        }
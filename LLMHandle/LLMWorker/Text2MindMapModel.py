# -*- coding: utf-8 -*-
import os
import re
import requests
from abc import ABC, abstractmethod
from openai import OpenAI
from .config import DEEPSEEK_API_KEY, QWEN_API_KEY, DOUBAO_API_KEY, XFYUN_API_KEY
from .prompt_loader import load_prompt 
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

# --- Deepseek MindMap Model API ---
class DeepseekMindMapModelAPI(BaseMindMapModelAPI):
    def __init__(self,
                 api_key: str = DEEPSEEK_API_KEY,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat",
                 role: str = "mindmap",
                 temperature: float = 0.7): 
        
        super().__init__(temperature=temperature) 
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_config = load_prompt(role)
        self.temperature = temperature

    def generate_code(self, query: str, temperature: float = self.temperature) -> str: # 默认情况下不需要传入温度，模型中已经设置了默认值
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

    def execute(self, query: str, temperature: float = self.temperature) -> str:
        code = self.generate_code(query, temperature)
        processed_code = self.postprocess_code(code)
        return processed_code



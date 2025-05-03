# -*- coding: utf-8 -*-
import os
import re
import requests
from abc import ABC, abstractmethod
from openai import OpenAI
from .config import DEEPSEEK_API_KEY, QWEN_API_KEY, DOUBAO_API_KEY, XFYUN_API_KEY
from .prompt_loader import load_prompt 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    prompt_config = load_prompt("summarizer")
    print(prompt_config)
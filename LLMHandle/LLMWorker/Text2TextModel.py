# -*- coding: utf-8 -*-
import os
import re
import requests
from abc import ABC, abstractmethod
from openai import OpenAI
from .config import DEEPSEEK_API_KEY, QWEN_API_KEY, DOUBAO_API_KEY, XFYUN_API_KEY
from .prompt_loader import load_prompt
from ..celery_app import celery_app # Import celery app


# --- 抽象基类 ---
class BaseSummaryAPI(ABC):
    @abstractmethod
    def summarize(self, query: str, **kwargs) -> str:
        pass

# --- 讯飞 API 封装 ---
class XunfeiSummaryAPI(BaseSummaryAPI):
    def __init__(self,
                 api_key: str = XFYUN_API_KEY,
                 url: str = "https://spark-api-open.xf-yun.com/v1/chat/completions",
                #  url: str = "https://spark-api-open.xf-yun.com/v1",
                 model: str = "4.0Ultra",
                 max_tokens: int = 4096,
                 role: str = "summarizer"):
        self.api_key = api_key
        self.url = url
        self.model = model
        self.max_tokens = max_tokens
        self.prompt_config = load_prompt(role)

    # Note: This method runs synchronously within the task
    def _sync_summarize(self, query: str, temperature: float = 0.5, **kwargs) -> str:
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

    # This is the method called directly, now delegates to the task
    def summarize(self, query: str, **kwargs) -> str:
        # This method is kept for potential synchronous use or compatibility
        # For async, call summarize_task.delay() directly
        return self._sync_summarize(query, **kwargs)

# --- DeepSeek 封装 ---
class DeepseekSummaryAPI(BaseSummaryAPI):
    def __init__(self,
                 api_key: str = DEEPSEEK_API_KEY,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat",
                 role: str = "summarizer"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_config = load_prompt(role)

    # Note: This method runs synchronously within the task
    def _sync_summarize(self, query: str, temperature: float = 0.7, **kwargs) -> str:
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

    # This is the method called directly, now delegates to the task
    def summarize(self, query: str, **kwargs) -> str:
        # This method is kept for potential synchronous use or compatibility
        # For async, call summarize_task.delay() directly
        return self._sync_summarize(query, **kwargs)

# --- Qwen 封装 ---
class QwenSummaryAPI(BaseSummaryAPI):
    def __init__(self,
                 api_key: str = QWEN_API_KEY,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model: str = "qwen-plus",
                 role: str = "summarizer"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_config = load_prompt(role)

    # Note: This method runs synchronously within the task
    def _sync_summarize(self, query: str, temperature: float = 1.0, **kwargs) -> str:
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

    # This is the method called directly, now delegates to the task
    def summarize(self, query: str, **kwargs) -> str:
        # This method is kept for potential synchronous use or compatibility
        # For async, call summarize_task.delay() directly
        return self._sync_summarize(query, **kwargs)

# --- 豆包封装 ---
class DoubaoSummaryAPI(BaseSummaryAPI):
    def __init__(self,
                 api_key: str = DOUBAO_API_KEY,
                 base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
                 model: str = "doubao-1-5-thinking-pro-250415",
                 role: str = "summarizer"):
        # Note: Client needs to be initialized within the task or passed if safe
        # For simplicity, we re-initialize here, but consider alternatives for efficiency
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.prompt_config = load_prompt(role)

    # Note: This method runs synchronously within the task
    def _sync_summarize(self, query: str, temperature: float = 0.7, **kwargs) -> str:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = []
        if self.prompt_config.get("system"):
            messages.append({"role": "system", "content": self.prompt_config["system"]})
        messages.append({"role": "user", "content": self.prompt_config.get("user_prefix", "") + query})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=False,
            **kwargs
        )
        return response.choices[0].message.content.strip()

    # This is the method called directly, now delegates to the task
    def summarize(self, query: str, **kwargs) -> str:
        # This method is kept for potential synchronous use or compatibility
        # For async, call summarize_task.delay() directly
        return self._sync_summarize(query, **kwargs)

# --- Celery Tasks --- Define tasks for each API
@celery_app.task(bind=True, name='llm.summarize.xunfei')
def summarize_xunfei_task(self, query: str, role: str = "summarizer", **kwargs):
    api = XunfeiSummaryAPI(role=role)
    return api._sync_summarize(query, **kwargs)

@celery_app.task(bind=True, name='llm.summarize.deepseek')
def summarize_deepseek_task(self, query: str, role: str = "summarizer", **kwargs):
    api = DeepseekSummaryAPI(role=role)
    return api._sync_summarize(query, **kwargs)

@celery_app.task(bind=True, name='llm.summarize.qwen')
def summarize_qwen_task(self, query: str, role: str = "summarizer", **kwargs):
    api = QwenSummaryAPI(role=role)
    return api._sync_summarize(query, **kwargs)

@celery_app.task(bind=True, name='llm.summarize.doubao')
def summarize_doubao_task(self, query: str, role: str = "summarizer", **kwargs):
    api = DoubaoSummaryAPI(role=role)
    return api._sync_summarize(query, **kwargs)


# --- 总调度类 --- Updated to use tasks
class Summary_Master:
    _task_registry = {
        "deepseek": summarize_deepseek_task,
        "qwen": summarize_qwen_task,
        "doubao": summarize_doubao_task,
        "xunfei": summarize_xunfei_task,
    }
    # Keep original registry for potential sync use or reference
    _api_registry = {
        "deepseek": DeepseekSummaryAPI,
        "qwen": QwenSummaryAPI,
        "doubao": DoubaoSummaryAPI,
        "xunfei": XunfeiSummaryAPI,
    }


    def __init__(self, use_api: str = "deepseek", role: str = "summarizer"):
        if use_api not in self._task_registry:
            raise ValueError(f"不支持的 API: {use_api}")
        self.use_api = use_api
        self.role = role # Store role
        # No need to initialize the client here for async calls
        # self.client = self._api_registry[use_api](role=role)

    # Renamed to reflect async nature, returns AsyncResult
    def get_summary_async(self, query: str, **kwargs):
        task = self._task_registry[self.use_api]
        # Pass role and other necessary args to the task
        return task.delay(query=query, role=self.role, **kwargs)

    # Optional: Keep a synchronous version if needed
    def get_summary_sync(self, query: str, **kwargs) -> str:
        api_class = self._api_registry[self.use_api]
        client = api_class(role=self.role)
        return client.summarize(query, **kwargs)
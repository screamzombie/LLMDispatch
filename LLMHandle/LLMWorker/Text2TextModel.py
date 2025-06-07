# -*- coding: utf-8 -*-
import os
import re
import requests
from abc import ABC, abstractmethod
from openai import OpenAI
from typing import Generator, Tuple, Deque  # 为类型提示添加 Deque
from collections import deque  # 导入 deque
import threading  # 导入 threading

# 好的做法是导入您打算捕获的特定异常。
# 确保您的 openai 库版本是最新的，以便包含这些异常。
try:
    from openai import APIConnectionError, AuthenticationError, RateLimitError, APIStatusError, NotFoundError
except ImportError:
    # 如果 openai 库版本过旧或未完全安装，则定义虚拟异常
    class APIConnectionError(Exception):
        pass


    class AuthenticationError(Exception):
        pass


    class RateLimitError(Exception):
        pass


    class APIStatusError(Exception):
        pass


    class NotFoundError(Exception):
        pass

# 假设 config 和 PromptLoader 在这些位置
from LLMDispatch.LLMHandle import config
from LLMDispatch.LLMHandle.LLMWorker.PromptLoader import load_prompt
from dataclasses import dataclass
from typing import Literal


@dataclass
class DeepThinkChunk:
    """封装深度思考流中的一个数据块及其类型。"""
    type: Literal['reasoning', 'response']
    content: str


class LLMStreamSplitter:
    def __init__(self, client_create_method, model_id: str, messages: list, temperature: float, **kwargs):
        self._client_create_method = client_create_method  # OpenAI client.chat.completions.create
        self._model_id = model_id
        self._messages = messages
        self._temperature = temperature
        self._kwargs = kwargs

        self._stream_iter: Generator | None = None
        self._reasoning_buffer: Deque[str] = deque()
        self._response_buffer: Deque[str] = deque()

        self._stream_initialized_flag = False
        self._stream_exhausted_flag = False

        self._lock = threading.Lock()

    def _initialize_stream_if_needed(self):
        if not self._stream_initialized_flag:
            try:
                stream = self._client_create_method(
                    model=self._model_id,
                    messages=self._messages,
                    temperature=self._temperature,
                    stream=True,
                    **self._kwargs
                )
                self._stream_iter = iter(stream)
                self._stream_initialized_flag = True
            except Exception as e:
                print(f"错误：初始化API流失败 (LLMStreamSplitter): {e}")
                self._stream_exhausted_flag = True

    def _fetch_next_chunk_into_buffers(self) -> bool:
        with self._lock:
            if not self._stream_initialized_flag:
                self._initialize_stream_if_needed()
                if not self._stream_initialized_flag:
                    return False

            if self._stream_exhausted_flag:
                return False

            try:
                if self._stream_iter is None:
                    self._stream_exhausted_flag = True
                    return False
                chunk = next(self._stream_iter)

                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta:
                        # 兼容 qwen 的 enable_thinking 返回
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            for tool_call in delta.tool_calls:
                                if tool_call.function and tool_call.function.name == "thought":
                                    self._reasoning_buffer.append(tool_call.function.arguments)
                        # 兼容 deepseek 的 reasoning_content
                        elif hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            self._reasoning_buffer.append(delta.reasoning_content)
                        # 通用内容
                        if delta.content:
                            self._response_buffer.append(delta.content)

                    if chunk.choices[0].finish_reason:
                        self._stream_exhausted_flag = True
                return True

            except StopIteration:
                self._stream_exhausted_flag = True
                return False
            except Exception as e:
                print(f"错误：从流中获取数据时出错 (LLMStreamSplitter): {e}")
                self._stream_exhausted_flag = True
                return False

    def stream(self) -> Generator[DeepThinkChunk, None, None]:
        """
        生成一个统一的流，包含思考过程和最终响应。
        """
        while True:
            chunk_yielded = False

            # 先清空缓冲区中的内容
            with self._lock:
                if self._reasoning_buffer:
                    yield DeepThinkChunk(type='reasoning', content=self._reasoning_buffer.popleft())
                    chunk_yielded = True
                elif self._response_buffer:
                    yield DeepThinkChunk(type='response', content=self._response_buffer.popleft())
                    chunk_yielded = True

            # 如果本次循环产出了内容，则立即开始下一次循环，以尽快清空缓冲区
            if chunk_yielded:
                continue

            # 如果缓冲区为空，尝试从上游获取新数据
            # 如果获取失败或流已结束，则跳出循环
            if not self._fetch_next_chunk_into_buffers():
                # 在退出前，再次检查并清空可能在最后一次 fetch 中填充的缓冲区
                with self._lock:
                    while self._reasoning_buffer:
                        yield DeepThinkChunk(type='reasoning', content=self._reasoning_buffer.popleft())
                    while self._response_buffer:
                        yield DeepThinkChunk(type='response', content=self._response_buffer.popleft())
                break  # 结束生成器


# --- 抽象基类 ---
@dataclass
class BaseTextModelAPI(ABC):
    temperature: float = 0.7  # 默认温度

    @abstractmethod
    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        pass

    @abstractmethod
    def deepthink_generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[
        DeepThinkChunk, None, None]:
        """
        以流式方式生成文本，将思维过程和最终结果合并到单个流中。
        返回一个生成器，产出 DeepThinkChunk 对象。
        """
        pass

    @abstractmethod
    def generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[str, None, None]:
        """以流式方式生成文本，逐块产生内容。"""
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

    @abstractmethod
    def api_test(self) -> bool:
        pass


# --- 讯飞 API 封装 ---
class XunfeiTextModelAPI(BaseTextModelAPI):
    def __init__(self,
                 api_key: str = None,
                 base_url: str = "https://spark-api-open.xf-yun.com/v2",  # 使用兼容OpenAI的V2地址
                 model: str = "x1",  # V2接口使用的模型标识符为 "x1"
                 role: str = "default",
                 temperature: float = 0.5):
        super().__init__(temperature=temperature)
        # 注意：讯飞V2接口的API Key格式为 "APIKey:APISecret" 或直接使用控制台的 "APIPassword"
        self.api_key = api_key if api_key is not None else config.XUNFEI_API_KEY
        self.base_url = base_url
        self.model = model
        self.role = role
        self.prompt_config = load_prompt(role)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def deepthink_generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[DeepThinkChunk, None, None]:
        """
        以流式方式生成文本，将思维过程和最终结果合并到单个流中。
        讯飞X1模型支持 reasoning_content，可直接使用LLMStreamSplitter。
        """
        current_temp = temperature if temperature is not None else self.temperature
        messages = []
        if self.prompt_config.get("system"):
            messages.append({"role": "system", "content": self.prompt_config["system"]})
        messages.append({"role": "user", "content": self.prompt_config.get("user_prefix", "") + query})

        splitter = LLMStreamSplitter(
            client_create_method=self.client.chat.completions.create,
            model_id=self.model,  # 使用支持思维链的x1模型
            messages=messages,
            temperature=current_temp,
            **kwargs
        )
        return splitter.stream()

    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        """
        使用兼容OpenAI的SDK生成文本（非流式）。
        """
        current_temp = temperature if temperature is not None else self.temperature
        messages = [
            {"role": "system", "content": self.prompt_config.get("system", "")},
            {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
        ]
        if not messages[0]["content"]:  # 如果系统消息内容为空，则移除它
            messages.pop(0)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=current_temp,
                stream=False,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"讯飞 API 调用失败: {e}")
            return "讯飞 API 调用失败"

    def generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[str, None, None]:
        """以流式方式生成文本，逐块产生内容。"""
        current_temp = temperature if temperature is not None else self.temperature
        messages = [
            {"role": "system", "content": self.prompt_config.get("system", "")},
            {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
        ]
        if not messages[0]["content"]:  # 如果系统消息内容为空，则移除它
            messages.pop(0)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=current_temp,
                stream=True,
                **kwargs
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"讯飞 API 流式调用失败: {e}")
            # 可以选择 yield 一个错误信息或直接返回
            yield f"讯飞 API 流式调用失败: {e}"

    def api_test(self) -> bool:
        """
        使用兼容OpenAI的SDK测试API连通性。
        """
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                stream=False,
                timeout=10.0
            )
            return True
        except (APIConnectionError, AuthenticationError, RateLimitError, NotFoundError, APIStatusError):
            return False
        except Exception:
            return False

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
                 api_key: str = None,
                 base_url: str = "https://api.deepseek.com",
                 model: str = "deepseek-chat",
                 role: str = "default",
                 temperature: float = 0.7):
        super().__init__(temperature=temperature)
        self.api_key = api_key if api_key is not None else config.DEEPSEEK_API_KEY
        self.base_url = base_url
        self.model = model
        self.role = role
        self.prompt_config = load_prompt(role)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def deepthink_generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[DeepThinkChunk, None, None]:
        current_temp = temperature if temperature is not None else self.temperature
        messages = []
        if self.prompt_config.get("system"):
            messages.append({"role": "system", "content": self.prompt_config["system"]})
        messages.append({"role": "user", "content": self.prompt_config.get("user_prefix", "") + query})

        splitter = LLMStreamSplitter(
            client_create_method=self.client.chat.completions.create,
            model_id='deepseek-reasoner',  # 使用支持思维链的模型
            messages=messages,
            temperature=current_temp,
            **kwargs
        )
        return splitter.stream()

    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        current_temp = temperature if temperature is not None else self.temperature
        messages = [
            {"role": "system", "content": self.prompt_config.get("system", "")},
            {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
        ]
        if not messages[0]["content"]:  # 如果系统消息内容为空，则移除它
            messages.pop(0)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=current_temp,
            stream=False,
            **kwargs
        )
        return response.choices[0].message.content.strip()

    def generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[str, None, None]:
        current_temp = temperature if temperature is not None else self.temperature
        messages = [
            {"role": "system", "content": self.prompt_config.get("system", "")},
            {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
        ]
        if not messages[0]["content"]:  # 如果系统消息内容为空，则移除它
            messages.pop(0)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=current_temp,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def api_test(self) -> bool:
        try:
            test_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            test_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                stream=False,
                timeout=10.0
            )
            return True
        except (APIConnectionError, AuthenticationError, RateLimitError, NotFoundError, APIStatusError):
            return False
        except Exception:
            return False

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
                 api_key: str = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 model: str = "qwen3-235b-a22b",  # 例如: "qwen-turbo", "qwen-plus", "qwen-max"
                 role: str = "default",
                 temperature: float = 0.7):  # Qwen 默认温度是 1.0
        super().__init__(temperature=temperature)
        self.api_key = api_key if api_key is not None else config.QWEN_API_KEY
        self.base_url = base_url
        self.model = model
        self.role = role
        self.prompt_config = load_prompt(role)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def deepthink_generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[DeepThinkChunk, None, None]:
        current_temp = temperature if temperature is not None else self.temperature
        messages = []
        if self.prompt_config.get("system"):
            messages.append({"role": "system", "content": self.prompt_config["system"]})
        messages.append({"role": "user", "content": self.prompt_config.get("user_prefix", "") + query})

        splitter = LLMStreamSplitter(
            client_create_method=self.client.chat.completions.create,
            model_id=self.model,  # 使用支持思维链的模型
            messages=messages,
            temperature=current_temp,
            extra_body={"enable_thinking": True},
            **kwargs
        )
        return splitter.stream()

    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        current_temp = temperature if temperature is not None else self.temperature
        messages = [
            {"role": "system", "content": self.prompt_config.get("system", "")},
            {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
        ]
        if not messages[0]["content"]:  # 如果系统消息内容为空，则移除它
            messages.pop(0)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=current_temp,
            stream=False,
            extra_body={"enable_thinking": False},
            **kwargs
        )
        return response.choices[0].message.content.strip()

    def generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[str, None, None]:
        current_temp = temperature if temperature is not None else self.temperature
        messages = [
            {"role": "system", "content": self.prompt_config.get("system", "")},
            {"role": "user", "content": self.prompt_config.get("user_prefix", "") + query}
        ]
        if not messages[0]["content"]:  # 如果系统消息内容为空，则移除它
            messages.pop(0)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=current_temp,
            stream=True,
            extra_body={"enable_thinking": False},
            **kwargs
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def api_test(self) -> bool:
        try:
            test_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            test_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                stream=False,
                extra_body={"enable_thinking": False},
                timeout=10.0
            )
            return True
        except (APIConnectionError, AuthenticationError, RateLimitError, NotFoundError, APIStatusError):
            return False
        except Exception:
            return False

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
                 api_key: str = None,
                 base_url: str = "https://ark.cn-beijing.volces.com/api/v3",  # 确保这是 OpenAI 兼容的端点
                 model: str = "doubao-1-5-pro-32k-250115",
                 role: str = "default",
                 temperature: float = 0.7):
        super().__init__(temperature=temperature)
        self.api_key = api_key if api_key is not None else config.DOUBAO_API_KEY
        self.base_url = base_url
        self.model = model  # 对于使用 OpenAI 兼容 API 的豆包，这通常是一个 endpoint_id
        self.role = role
        self.prompt_config = load_prompt(role)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def deepthink_generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[DeepThinkChunk, None, None]:
        current_temp = temperature if temperature is not None else self.temperature
        messages = []
        if self.prompt_config.get("system"):
            messages.append({"role": "system", "content": self.prompt_config["system"]})
        messages.append({"role": "user", "content": self.prompt_config.get("user_prefix", "") + query})

        # print("DEBUG: Doubao deepthink - Preparing to create LLMStreamSplitter") # 用于调试
        splitter = LLMStreamSplitter(
            client_create_method=self.client.chat.completions.create,
            model_id="doubao-1-5-thinking-pro-m-250428",  # 使用支持思维链的模型
            messages=messages,
            temperature=current_temp,
            **kwargs
        )
        # print("DEBUG: Doubao deepthink - LLMStreamSplitter created, returning generators") # 用于调试
        return splitter.stream()

    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        current_temp = temperature if temperature is not None else self.temperature
        messages = []
        if self.prompt_config.get("system"):  # 豆包如果提供了系统提示，则倾向于使用它
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

    def generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[str, None, None]:
        current_temp = temperature if temperature is not None else self.temperature
        messages = []
        if self.prompt_config.get("system"):
            messages.append({"role": "system", "content": self.prompt_config["system"]})
        messages.append({"role": "user", "content": self.prompt_config.get("user_prefix", "") + query})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=current_temp,
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def api_test(self) -> bool:
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                stream=False,
                timeout=10.0
            )
            return True
        except (APIConnectionError, AuthenticationError, RateLimitError, NotFoundError, APIStatusError) as e:
            # print(f"豆包 API 测试失败 (OpenAI 特定错误): {e}") # 可选: 用于调试
            return False
        except Exception as e:
            # print(f"豆包 API 测试失败 (通用错误): {e}") # 可选: 用于调试
            return False

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
        "xunfei": XunfeiTextModelAPI,
        "deepseek": DeepseekTextModelAPI,
        "qwen": QwenTextModelAPI,
        "doubao": DoubaoTextModelAPI,
    }

    def __init__(self, use_api: str = "deepseek", role: str = "default", temperature: float = None):
        if use_api not in self._registry:
            raise ValueError(f"不支持的 API: {use_api}. 可用: {list(self._registry.keys())}")
        self.use_api = use_api
        init_kwargs = {'role': role}
        if temperature is not None:
            init_kwargs['temperature'] = temperature

        # 初始化前确保 API 密钥可用
        api_name_upper = use_api.upper()
        config_key_name = f"{api_name_upper}_API_KEY"
        if not getattr(config, config_key_name, None) and not init_kwargs.get('api_key'):
            print(f"警告: 未在配置中找到 {use_api} 的 API 密钥 ({config_key_name})，也未直接提供。初始化可能会失败。")

        self.client: BaseTextModelAPI = self._registry[use_api](**init_kwargs)

    def deepthink_generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[DeepThinkChunk, None, None]:
        return self.client.deepthink_generate_text_stream(query, temperature=temperature, **kwargs)

    def generate_text(self, query: str, temperature: float = None, **kwargs) -> str:
        return self.client.generate_text(query, temperature=temperature, **kwargs)

    def generate_text_stream(self, query: str, temperature: float = None, **kwargs) -> Generator[str, None, None]:
        """从当前选定的 API 客户端以流式方式生成文本。"""
        return self.client.generate_text_stream(query, temperature=temperature, **kwargs)

    def change_temperature(self, temperature: float):
        self.client.change_temperature(temperature)

    def set_custom_prompt(self, user_prompt: str, mode: str = "append"):
        self.client.set_custom_prompt(user_prompt, mode)

    def get_prompt(self) -> dict:
        return self.client.get_prompt()

    def reload_prompt(self, role: str = None):
        self.client.reload_prompt(role)

    def switch_api(self, use_api: str, role: str = None, temperature: float = None, api_key: str = None):
        if use_api not in self._registry:
            raise ValueError(f"不支持的 API: {use_api}. 可用: {list(self._registry.keys())}")

        self.use_api = use_api
        current_role = role if role is not None else self.client.role
        current_temp = temperature if temperature is not None else self.client.temperature

        init_kwargs = {'role': current_role, 'temperature': current_temp}
        if api_key:  # 允许在切换时覆盖 API 密钥
            init_kwargs['api_key'] = api_key
        else:  # 如果未提供，则检查配置
            api_name_upper = use_api.upper()
            config_key_name = f"{api_name_upper}_API_KEY"
            if not getattr(config, config_key_name, None):
                print(
                    f"警告: 正在切换到 {use_api}。未在配置中找到 API 密钥 ({config_key_name})，也未直接提供。初始化可能会失败。")

        self.client = self._registry[use_api](**init_kwargs)
        print(f"已切换到 API: {use_api}，角色: {self.client.role}，温度: {self.client.temperature}")

    def get_current_config(self) -> dict:
        return {
            "api": self.use_api,
            "role": self.client.role,
            "temperature": self.client.temperature,
            "model": getattr(self.client, 'model', 'N/A'),
            "prompts": self.get_prompt()
        }

    def api_test(self) -> bool:
        """测试指定的 API，如果未指定，则测试当前 API。"""
        print(f"正在测试当前 API: {self.use_api}...")
        is_ok = self.client.api_test()
        print(f"当前 API ({self.use_api}) 测试结果: {'成功' if is_ok else '失败'}")
        return is_ok


# 示例用法 (说明性 - 假设 config 和 PromptLoader 已设置)
if __name__ == '__main__':
    import time

    use_api = "xunfei"  # deepseek qwen doubao xunfei

    print("=" * 60)
    print(f"🚀 开始执行({use_api}) API 功能测试")
    print("=" * 60)

    try:
        manager = TextGenerationManager(use_api=use_api, role="default")
    except Exception as e:
        print(f"❌ 初始化 TextGenerationManager 失败: {e}")
        exit()  # 如果管理器初始化失败，则无法继续

    # --- 1. API 连通性测试 ---
    print("\n--- [1/4] API 连通性测试 ---")
    if not manager.api_test():
        print("❌ API 连通性测试失败。请检查您的网络连接和 API 密钥。")
        exit()  # 如果API不通，则退出

    # 定义一个通用的查询
    query = "用简单的语言解释一下什么是人工智能，并举一个生活中的例子。"
    print(f"\n使用通用查询进行后续测试: '{query}'")

    # --- 2. 标准文本生成测试 (generate_text) ---
    print("\n\n" + "=" * 20 + " [2/4] 标准生成测试 " + "=" * 20)
    try:
        start_time = time.time()
        response = manager.generate_text(query)
        end_time = time.time()
        print(f"✅ 标准生成成功 (耗时: {end_time - start_time:.2f}s)")
        print("-" * 50)
        print("完整回复:\n" + response)
        print("-" * 50)
    except Exception as e:
        print(f"❌ 标准生成测试失败: {e}")

    # --- 3. 流式生成测试 (generate_text_stream) ---
    print("\n\n" + "=" * 20 + " [3/4] 普通流式生成测试 " + "=" * 20)
    try:
        print("流式回复:")
        start_time = time.time()
        full_response_stream = ""
        stream_generator = manager.generate_text_stream(query)
        for chunk in stream_generator:
            print(chunk, end="", flush=True)
            full_response_stream += chunk
        end_time = time.time()

        if not full_response_stream.strip():
            print("\n❌ 普通流式生成测试失败: 未收到任何内容。")
        else:
            print(f"\n\n✅ 普通流式生成成功 (耗时: {end_time - start_time:.2f}s)")

    except Exception as e:
        print(f"\n❌ 普通流式生成测试失败: {e}")

    # --- 4. 深度思考流式生成测试 (deepthink_generate_text_stream) ---
    print("\n\n" + "=" * 20 + " [4/4] 深度思考流式生成测试 " + "=" * 20)
    # 换一个更适合思考的问题
    deep_query = "小明有5个苹果，他又去商店买了3篮苹果，每篮有4个。请问他现在一共有多少个苹果？请展示你的思考过程。"
    print(f"使用深度思考查询: '{deep_query}'")

    try:
        print("深度思考过程流式回复:")
        start_time = time.time()
        full_reasoning = ""
        full_response_deepthink = ""

        deepthink_generator = manager.deepthink_generate_text_stream(deep_query)
        for chunk in deepthink_generator:
            if chunk.type == 'reasoning':
                # 用蓝色打印思考过程，使其易于区分
                print(f"\033[94m{chunk.content}\033[0m", end="", flush=True)
                full_reasoning += chunk.content
            else:
                if not full_response_deepthink:
                    print("\n深度思考结果流式回复:")
                print(chunk.content, end="", flush=True)
                full_response_deepthink += chunk.content
        end_time = time.time()

        if not full_reasoning.strip() and not full_response_deepthink.strip():
            print("\n❌ 深度思考流式生成测试失败: 未收到任何内容。")
        elif not full_reasoning.strip():
            print("\n\n⚠️  深度思考流式生成警告: 未收到'思考'部分，模型可能未开启或未触发思考模式。")
            print(f"✅ 深度思考流式生成完成 (耗时: {end_time - start_time:.2f}s)")
        else:
            print(f"\n\n✅ 深度思考流式生成成功 (耗时: {end_time - start_time:.2f}s)")

    except Exception as e:
        print(f"\n❌ 深度思考流式生成测试失败: {e}")

    print("\n\n" + "=" * 60)
    print("🎉 所有测试执行完毕。")
    print("=" * 60)





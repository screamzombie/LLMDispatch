# -*- coding: utf-8 -*-
"""
picture_master.py
['1024*1024', '720*1280', '1280*720', '768*1152']
提供图片生成功能，支持多种 API 后端（当前实现 Qwen、Kling 和 智谱）。
"""
import os
import time
import jwt
import requests
import re
from http import HTTPStatus
from dashscope import ImageSynthesis
from abc import ABC, abstractmethod
from openai import OpenAI
from LLMHandle.config import QWEN_IMG_API_KEY, KLING_ACCESS_KEY, KLING_SECRET_KEY
from LLMHandle.LLMWorker.PromptLoader import load_prompt 
from dataclasses import dataclass
# --- 抽象基类 ---
class BasePictureAPI(ABC):
    @abstractmethod
    def generate_image(self, query: str, **kwargs) -> str:
        """
        抽象方法：生成图片并返回保存路径。
        子类必须实现此方法。
        """
        pass


# --- Qwen (DashScope) API 封装 ---
class QWENPictureAPI(BasePictureAPI):
    """使用 DashScope ImageSynthesis 接口生成图片的实现类。"""

    def __init__(self,
                 api_key: str = QWEN_IMG_API_KEY,  # 请替换为您的有效 API Key
                 model: str = "wanx-v1",  # 注意：模型名称可能需要根据实际 API 更新，wanx2.1-t2i-turbo 可能不再支持或有更新
                 output_parent_path: str = "./"):  # 修改默认路径以区分
        self.api_key = api_key
        self.model = model
        self.output_parent_path = output_parent_path
        os.makedirs(self.output_parent_path, exist_ok=True)

    def generate_image(self,
                       query: str,
                       size: str = "1024*1024",
                       seed: int = 0,
                       n: int = 1,  # DashScope API 参数是 n
                       # prompt_extend: bool = True, # wanx-v1 模型可能不支持此参数，请查阅文档
                       output_path: str = None,
                       **kwargs) -> str:
        """
        使用 Qwen API 生成图片并保存到指定路径或默认路径。

        Args:
            query (str): 生成图片的文本描述。
            size (str, optional): 图片尺寸。默认为 "1024*1024"。
            seed (int, optional): 随机种子。默认为 0。
            n (int, optional): 生成图片的数量。默认为 1。
            output_path (str, optional): 图片保存的完整路径（包括文件名）。
                                         如果为 None，则自动生成文件名并保存在 output_parent_path 下。
                                         默认为 None。
            **kwargs: 其他传递给 API 的参数。

        Returns:
            str: 保存图片的实际文件路径 (仅返回第一张图片的路径，如果 n > 1)。

        Raises:
            Exception: 如果图片生成失败。
        """
        try:
            # 注意：根据最新的 DashScope 文档调整参数
            # prompt_extend 参数在某些模型下可能不再支持或名称变化
            # 确保 api_key, model, prompt, n, size, seed 是 API 支持的参数
            call_params = {
                'api_key': self.api_key,
                'model': self.model,
                'prompt': query,
                'n': n,
                'size': size,
                'seed': seed,
                # 如果需要传递其他参数，可以通过 kwargs 传入并添加到这里
            }
            call_params.update(kwargs)  # 合并额外参数

            rsp = ImageSynthesis.call(**call_params)

            # 移除之前的调试打印语句

            if rsp.status_code == HTTPStatus.OK:
                # 检查 API 返回的 output 和 task_status
                if rsp.output and getattr(rsp.output, 'task_status', None) == 'SUCCEEDED':
                    if not getattr(rsp.output, 'results', None):
                        # 即使 task SUCCEEDED，也可能没有 results，虽然少见
                        raise Exception(
                            f"Image generation task succeeded but no results found in response (status={rsp.status_code}, task_id={rsp.output.task_id})")

                    # 处理多张图片的情况 (如果 n > 1)
                    # 这里仅保存并返回第一张图片的路径
                    result = rsp.output.results[0]
                    url = result.url

                    if output_path:
                        file_path = output_path
                        # 确保目录存在
                        output_dir = os.path.dirname(file_path)
                        if output_dir:  # 只有在路径包含目录时才创建
                            os.makedirs(output_dir, exist_ok=True)
                    else:
                        # 如果未提供 output_path，则使用旧逻辑生成文件名
                        safe_query = "".join(c for c in query if c.isalnum())[:20]
                        file_name = f"{safe_query}_{seed}.jpg"  # 保持原有逻辑
                        file_path = os.path.join(self.output_parent_path, file_name)

                    resp_img = requests.get(url)
                    resp_img.raise_for_status()  # 检查请求是否成功
                    with open(file_path, "wb") as f:
                        f.write(resp_img.content)
                    return file_path  # 返回第一张图片的路径
                else:
                    # 如果 task_status 不是 SUCCEEDED，则提取错误信息
                    task_status = getattr(rsp.output, 'task_status', 'UNKNOWN')
                    error_code = getattr(rsp.output, 'code', 'N/A')
                    error_message = getattr(rsp.output, 'message', 'No message provided')
                    raise Exception(
                        f"Image generation task failed (status={rsp.status_code}, task_status={task_status}, code={error_code}, message={error_message})")
            else:
                # 如果 HTTP status code 就不是 OK
                raise Exception(
                    f"Image generation API request failed (status={rsp.status_code}, code={rsp.code}, message={rsp.message})")
        except requests.exceptions.RequestException as e:
            # 捕获 requests 库可能抛出的网络相关异常
            raise Exception(f"Network error during image download or API call: {e}")
        except Exception as e:
            # 捕获其他所有异常，包括我们自己抛出的
            # 避免重复包装信息，检查是否已经是我们格式化的错误
            if "Image generation" in str(e) or "Network error" in str(e):
                raise e  # 直接重新抛出已格式化的异常
            else:
                raise Exception(f"An unexpected error occurred during image generation: {e}")


# --- Kling API 封装 ---
class KlingPictureAPI(BasePictureAPI):
    """使用快手 Kling API 生成图片的实现类。"""

    def __init__(self,
                 access_key: str = KLING_ACCESS_KEY,
                 secret_key: str = KLING_SECRET_KEY,
                 api_base_url: str = "https://api.klingai.com",
                 model_name: str = "kling-v1",
                 aspect_ratio: str = "16:9",
                 output_parent_path: str = "./"):
        self.access_key = access_key
        self.secret_key = secret_key
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.aspect_ratio = aspect_ratio
        self.output_parent_path = output_parent_path
        os.makedirs(self.output_parent_path, exist_ok=True)

    def generate_image(self,
                       query: str,
                       negative_prompt: str = "",
                       seed: int = 0,
                       output_path: str = None,
                       **kwargs) -> str:
        """
        使用 Kling API 生成图片并保存到指定路径或默认路径。

        Args:
            query (str): 生成图片的文本描述。
            negative_prompt (str, optional): 负向提示词。默认为空字符串。
            seed (int, optional): 随机种子。默认为 0。
            output_path (str, optional): 图片保存的完整路径（包括文件名）。
                                         如果为 None，则自动生成文件名并保存在 output_parent_path 下。
                                         默认为 None。
            **kwargs: 其他传递给 API 的参数。

        Returns:
            str: 保存图片的实际文件路径。

        Raises:
            Exception: 如果图片生成失败。
        """
        token = self._encode_jwt_token()
        if not token:
            raise Exception("生成 JWT token 失败")

        # 创建生成任务
        create_url = self.api_base_url + "/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model_name": self.model_name,
            "prompt": query,
            "aspect_ratio": kwargs.get("aspect_ratio", self.aspect_ratio),
            "n": 1
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        try:
            response = requests.post(create_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get("code") != 0:
                raise Exception(f"创建任务失败: {data.get('message')}")

            task_id = data.get("data", {}).get("task_id")
            if not task_id:
                raise Exception("未获取到 task_id")

        except Exception as e:
            raise Exception(f"任务创建失败: {e}")

        # 轮询查询结果
        image_urls = self._poll_task(task_id)
        if not image_urls:
            raise Exception("任务最终未成功，未获取到图片")

        # 下载图片到本地
        url = image_urls[0]
        if output_path:
            file_path = output_path
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        else:
            safe_query = "".join(c for c in query if c.isalnum())[:20]
            file_name = f"{safe_query}_{seed}.jpg"
            file_path = os.path.join(self.output_parent_path, file_name)

        try:
            img_resp = requests.get(url)
            img_resp.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(img_resp.content)
            return file_path
        except Exception as e:
            raise Exception(f"图片下载失败: {e}")

    def _encode_jwt_token(self):
        """
        生成用于 Kling API 认证的 JWT Token。

        Returns:
            str: 生成的 JWT Token，失败时返回 None。
        """
        headers = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5
        }
        try:
            return jwt.encode(payload, self.secret_key, algorithm="HS256", headers=headers)
        except Exception as e:
            print(f"JWT 生成失败: {e}")
            return None

    def _poll_task(self, task_id: str, max_retries: int = 120, interval: int = 5):
        """
        轮询查询任务结果。

        Args:
            task_id (str): 任务 ID。
            max_retries (int, optional): 最大重试次数。默认为 20。
            interval (int, optional): 重试间隔（秒）。默认为 5。

        Returns:
            list: 成功时返回包含图片 URL 的列表，失败时返回 None。
        """
        url = f"{self.api_base_url}/v1/images/generations/{task_id}"
        for i in range(max_retries):
            print(f"轮询第 {i + 1}/{max_retries} 次...")
            token = self._encode_jwt_token()
            if not token:
                return None
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") != 0:
                    print(f"轮询失败: {data.get('message')}")
                    return None

                task_info = data.get("data", {})
                status = task_info.get("task_status")
                if status == "succeed":
                    images = task_info.get("task_result", {}).get("images", [])
                    return [img.get("url") for img in images if img.get("url")]
                elif status == "failed":
                    print(f"任务失败: {task_info.get('task_status_msg')}")
                    return None
                elif status in ["submitted", "processing"]:
                    time.sleep(interval)
                else:
                    print(f"未知状态: {status}")
                    return None
            except Exception as e:
                print(f"轮询请求失败: {e}")
                time.sleep(interval)
        return None


class PictureGenerationManager:
    _registry = {
        "qwen": QWENPictureAPI,
        "kling": KlingPictureAPI,   
    }

    def __init__(self, use_api: str = "qwen", **kwargs):
        """
        初始化 Picture_Master。

        Args:
            use_api (str, optional): 要使用的图片生成 API。默认为 "qwen"。
            **kwargs: 传递给具体 API 实现类构造函数的参数。
        """
        if use_api not in self._registry:
            raise ValueError(f"不支持的图片生成 API: {use_api}")
        self.use_api = use_api
        # 将 kwargs 传递给具体的 API 类初始化
        self.client = self._registry[use_api](**kwargs)

    def generate(self, query: str, **kwargs) -> str:
        """
        调用选定的 API 生成图片。

        Args:
            query (str): 生成图片的文本描述。
            **kwargs: 传递给具体 API 实现类 generate_image 方法的参数。

        Returns:
            str: 保存图片的实际文件路径。
        """
        return self.client.generate_image(query, **kwargs)


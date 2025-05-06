# -*- coding: utf-8 -*-
"""
统一图片生成 API：Qwen 与 Kling
生成图片时以 UUID 命名，并返回 (uuid, file_path) 元组
"""
import os
import time
import jwt
import uuid
import requests
from http import HTTPStatus
from typing import Tuple
from dashscope import ImageSynthesis
from abc import ABC, abstractmethod
from dataclasses import dataclass
from LLMHandle.config import QWEN_IMG_API_KEY, KLING_ACCESS_KEY, KLING_SECRET_KEY


# --- 抽象基类 ---
@dataclass
class BasePictureAPI(ABC):
    temperature: float = 0.7

    @abstractmethod
    def generate_image(self, query: str, **kwargs) -> Tuple[str, str]:
        pass

    @abstractmethod
    def change_temperature(self, temperature: float):
        self.temperature = temperature


# --- Qwen 实现类 ---
class QWENPictureAPI(BasePictureAPI):
    def __init__(self, api_key=QWEN_IMG_API_KEY, model="wanx-v1", output_parent_path="./results/picture/qwen"):
        self.api_key = api_key
        self.model = model
        self.output_parent_path = output_parent_path
        os.makedirs(self.output_parent_path, exist_ok=True)

    def change_temperature(self, temperature: float):
        self.temperature = temperature

    def generate_image(self, query: str, size="1024*1024", seed=0, n=1, **kwargs) -> Tuple[str, str]:
        image_uuid = str(uuid.uuid4())
        filename = f"{image_uuid}.jpg"
        file_path = os.path.join(self.output_parent_path, filename)

        call_params = {
            'api_key': self.api_key,
            'model': self.model,
            'prompt': query,
            'n': n,
            'size': size,
            'seed': seed,
        }
        call_params.update(kwargs)

        rsp = ImageSynthesis.call(**call_params)

        if rsp.status_code == HTTPStatus.OK:
            if rsp.output and getattr(rsp.output, 'task_status', None) == 'SUCCEEDED':
                result = rsp.output.results[0]
                url = result.url
                resp_img = requests.get(url)
                resp_img.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(resp_img.content)
                return image_uuid, file_path
            else:
                raise Exception(f"Qwen 图片生成失败: {rsp.output}")
        else:
            raise Exception(f"Qwen 接口请求失败: {rsp.status_code}")


# --- Kling 实现类 ---
class KlingPictureAPI(BasePictureAPI):
    def __init__(self, access_key=KLING_ACCESS_KEY, secret_key=KLING_SECRET_KEY,
                 model_name="kling-v1", aspect_ratio="16:9",
                 output_parent_path="./results/picture/kling"):
        self.access_key = access_key
        self.secret_key = secret_key
        self.api_base_url = "https://api.klingai.com"
        self.model_name = model_name
        self.aspect_ratio = aspect_ratio
        self.output_parent_path = output_parent_path
        os.makedirs(self.output_parent_path, exist_ok=True)

    def change_temperature(self, temperature: float):
        self.temperature = temperature

    def _encode_jwt_token(self):
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def generate_image(self, query: str, negative_prompt: str = "", seed: int = 0, **kwargs) -> Tuple[str, str]:
        image_uuid = str(uuid.uuid4())
        filename = f"{image_uuid}.jpg"
        file_path = os.path.join(self.output_parent_path, filename)

        token = self._encode_jwt_token()
        create_url = f"{self.api_base_url}/v1/images/generations"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "model_name": self.model_name,
            "prompt": query,
            "aspect_ratio": kwargs.get("aspect_ratio", self.aspect_ratio),
            "n": 1
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        resp = requests.post(create_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise Exception(f"Kling 创建任务失败: {data.get('message')}")

        task_id = data["data"]["task_id"]
        url = f"{self.api_base_url}/v1/images/generations/{task_id}"

        for _ in range(120):
            time.sleep(5)
            token = self._encode_jwt_token()
            headers = {"Authorization": f"Bearer {token}"}
            result = requests.get(url, headers=headers).json()
            if result.get("code") != 0:
                raise Exception(f"Kling 查询失败: {result.get('message')}")
            status = result["data"]["task_status"]
            if status == "succeed":
                img_url = result["data"]["task_result"]["images"][0]["url"]
                img_resp = requests.get(img_url)
                img_resp.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(img_resp.content)
                return image_uuid, file_path
            elif status == "failed":
                raise Exception("Kling 图片生成失败")

        raise TimeoutError("Kling 图片生成超时")


# --- 总调度类 ---
class PictureGenerationManager:
    _registry = {
        "qwen": QWENPictureAPI,
        "kling": KlingPictureAPI,
    }

    def __init__(self, use_api: str = "qwen", **kwargs):
        if use_api not in self._registry:
            raise ValueError(f"不支持的图片生成 API: {use_api}")
        self.use_api = use_api
        self.client = self._registry[use_api](**kwargs)

    def generate_image(self, query: str, **kwargs) -> Tuple[str, str]:
        return self.client.generate_image(query, **kwargs)

    def change_temperature(self, temperature: float):
        self.client.change_temperature(temperature)

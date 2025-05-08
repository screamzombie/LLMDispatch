# -*- coding: utf-8 -*-
"""
Video generation API interface supporting temperature, duration, resolution, and mode configuration.
Supports: Kling
"""
import os
import time
import jwt
import requests
import uuid # Added for UUID generation
from typing import Tuple # Added for type hinting
from abc import ABC, abstractmethod
from dataclasses import dataclass
from LLMHandle.config import KLING_ACCESS_KEY, KLING_SECRET_KEY,DASHSCOPE_API_KEY


# --- 抽象基类 ---
@dataclass
class BaseVideoModelAPI(ABC):
    temperature: float = 0.7

    @abstractmethod
    def change_temperature(self, temperature: float):
        self.temperature = temperature

    @abstractmethod
    def generate_video(self, query: str, **kwargs) -> Tuple[str, str]: # 返回 (uuid, video_path)
        pass

    @abstractmethod
    def check_api_availability(self) -> bool:
        pass



class DashScopeVideoAPI:
    def __init__(self,
                 api_key: str = DASHSCOPE_API_KEY,
                 model: str = "wanx2.1-t2v-turbo",
                 output_parent_path: str = "./results/video",
                 ):
        self.api_key = api_key
        self.model = model
        self.output_parent_path = output_parent_path        
        os.makedirs(self.output_parent_path, exist_ok=True)
        self.submit_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        self.query_url_base = "https://dashscope.aliyuncs.com/api/v1/tasks/"

    def generate_video(self,
                       query: str,
                       resolution: str = "1280*720",
                       duration: int = 5,
                       prompt_extend: bool = True,
                       seed: int = None,
                       poll_interval: int = 5) -> Tuple[str, str]:  # 返回 (uuid, video_path)
        video_uuid = str(uuid.uuid4())
        payload = {
            "model": self.model,
            "input": {
                "prompt": query
            },
            "parameters": {
                "size": resolution,
                "duration": duration,
                "prompt_extend": prompt_extend
            }
        }
        if seed is not None:
            payload["parameters"]["seed"] = seed

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-Async": "enable",
            "Content-Type": "application/json"
        }

        resp = requests.post(self.submit_url, headers=headers, json=payload)
        resp.raise_for_status()
        task_id = resp.json()["output"]["task_id"]
        print(f"\U0001f3ac Qwen 提交任务成功，id={task_id}")

        return self._poll_task(task_id, query, video_uuid, poll_interval)

    def _poll_task(self, task_id: str, query: str, video_uuid: str, interval: int) -> Tuple[str, str]:
        while True:
            status_resp = requests.get(self.query_url_base + task_id,
                                       headers={"Authorization": f"Bearer {self.api_key}"})
            status_resp.raise_for_status()
            result = status_resp.json()
            status = result["output"]["task_status"]
            print(f"[Polling] Qwen 任务状态: {status}")
            if status == "SUCCEEDED":
                video_url = result["output"]["video_url"]
                return self._download_video(video_url, video_uuid)
            elif status == "FAILED":
                raise RuntimeError(f"❌ Qwen 视频生成失败: {result}")
            time.sleep(interval)

    def _download_video(self, url: str, video_uuid: str) -> Tuple[str, str]:
        filename = f"{video_uuid}.mp4"
        file_path = os.path.join(self.output_parent_path, filename)
        print(f"📥 正在下载 Qwen 视频到 {file_path}")
        dl_resp = requests.get(url, stream=True)
        dl_resp.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in dl_resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Qwen 视频下载完成：{file_path}")
        return video_uuid, file_path

    def check_api_availability(self) -> bool:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.get(self.submit_url, headers=headers)
        return resp.status_code == 200



# --- Kling API 实现类 ---
class KlingVideoModelAPI(BaseVideoModelAPI):
    def __init__(self,
                 access_key: str = KLING_ACCESS_KEY,
                 secret_key: str = KLING_SECRET_KEY,
                 model_name: str = "kling-v1",
                 output_parent_path: str = "./results/video",
                 duration: str = "5",
                 resolution: str = "16:9",
                 mode: str = "std",
                 temperature: float = 0.7):
        super().__init__(temperature=temperature)
        
        self.access_key = access_key
        self.secret_key = secret_key
        self.api_base_url = "https://api.klingai.com"
        self.model_name = model_name
        self.output_parent_path = output_parent_path
        self.duration = duration
        self.resolution = resolution
        self.mode = mode
        self.temperature = temperature
        
    def change_duration(self, duration: str):
        self.duration = duration
    
    def change_resolution(self, resolution: str):
        if resolution not in ["16:9", "9:16","1:1"]:
            raise ValueError("分辨率必须是 '16:9','9:16','1:1' ")
        self.resolution = resolution
    
    def change_mode(self, mode: str):
        if mode not in ["std", "pro"]:
            raise ValueError("模式必须是 'std' 或 'pro' ")
        self.mode = mode

    def change_temperature(self, temperature: float):
        self.temperature = temperature

    def generate_video(self, query: str, resolution: str = "16:9", duration: str = "5", mode: str = "std", **kwargs) -> Tuple[str, str]: # 返回 (uuid, video_path)
        token = self._encode_jwt_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        payload = {
            "model_name": self.model_name,
            "prompt": query,
            "aspect_ratio": resolution,
            "duration": duration,
            "mode": mode,
            "cfg_scale": self.temperature  # 用温度控制模型贴合度
        }

        create_url = f"{self.api_base_url}/v1/videos/text2video"
        resp = requests.post(create_url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != 0:
            raise Exception(f"Kling 任务创建失败: {data.get('message')}")

        task_id = data["data"]["task_id"]
        print(task_id + " 视频生成中...")
        video_uuid = str(uuid.uuid4()) # 生成UUID
        return self._poll_task(task_id, video_uuid)

    def _encode_jwt_token(self) -> str:
        headers = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256", headers=headers)

    def _poll_task(self, task_id: str, video_uuid: str, max_retries: int = 600, interval: int = 10) -> Tuple[str, str]: # 返回 (uuid, video_path)
        url = f"{self.api_base_url}/v1/videos/text2video/{task_id}"
        for i in range(max_retries):
            print(f"[Polling] Kling 第 {i + 1}/{max_retries} 次查询中...")
            token = self._encode_jwt_token()
            headers = {"Authorization": f"Bearer {token}"}
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            result = resp.json()

            if result.get("code") != 0:
                raise Exception(f"Kling 查询失败: {result.get('message')}")

            status = result["data"]["task_status"]
            if status == "succeed":
                video_url = result["data"]["task_result"]["videos"][0]["url"]
                return self._download_video(video_url, video_uuid)
            elif status == "failed":
                raise RuntimeError(f"Kling 视频生成失败: {result['data'].get('task_status_msg', '未知错误')}")
            else:
                time.sleep(interval)

        raise TimeoutError("Kling 视频生成超时")

    def _download_video(self, url: str, video_uuid: str) -> Tuple[str, str]: # 返回 (uuid, video_path)
        os.makedirs(self.output_parent_path, exist_ok=True) #确保目录存在
        file_path = os.path.join(self.output_parent_path, f"{video_uuid}.mp4")
        print(f"📥 正在下载 Kling 视频到 {file_path}")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Kling 视频下载完成")
        return video_uuid, file_path


# --- 总调度类 ---
class VideoGenerationManager:
    _registry = {
        "kling": KlingVideoModelAPI,
        "qwen":DashScopeVideoAPI,
    }

    def __init__(self, use_api: str = "kling", **kwargs):
        if use_api not in self._registry:
            raise ValueError(f"不支持的视频生成 API: {use_api}")
        self.use_api = use_api
        self.client = self._registry[use_api](**kwargs)

    def generate_video(self, query: str, **kwargs) -> Tuple[str, str]: # 返回 (uuid, video_path)
        return self.client.generate_video(query, **kwargs)

    def change_temperature(self, temperature: float):
        self.client.change_temperature(temperature)

    def get_current_api(self) -> str:
        return self.use_api
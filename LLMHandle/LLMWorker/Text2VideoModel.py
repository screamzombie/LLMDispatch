# -*- coding: utf-8 -*-
"""
统一视频生成 API：Qwen(DashScope), Kling, Jimeng
所有接口统一使用 aspect_ratio 参数控制尺寸
生成视频时以 UUID 命名，并返回 (uuid, file_path) 元组
"""
import os
import time
import jwt
import uuid
import base64
import requests
from http import HTTPStatus
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from LLMDispatch.LLMHandle import config

# 签名相关的辅助模块
import datetime
import hashlib
import hmac
import json


# --- 统一的辅助函数 ---
def _video_aspect_ratio_to_dims(aspect_ratio: str, quality: str = '720p') -> str:
    """
    将视频宽高比字符串和质量等级转换为DashScope所需的 "width*height" 字符串。
    """
    resolution_map = {
        '720p': {
            '16:9': '1280*720', '9:16': '720*1280', '1:1': '960*960',
            '4:3': '1088*832', '3:4': '832*1088'
        },
        '480p': {
            '16:9': '832*480', '9:16': '480*832', '1:1': '624*624'
        }
    }
    if quality not in resolution_map:
        raise ValueError(f"不支持的质量等级: '{quality}'. 可选: '720p', '480p'.")
    if aspect_ratio not in resolution_map[quality]:
        raise ValueError(f"在 {quality} 质量下, 不支持的宽高比: '{aspect_ratio}'. 可选: {list(resolution_map[quality].keys())}")
    return resolution_map[quality][aspect_ratio]


# --- 抽象基类 (已修改) ---
@dataclass
class BaseVideoAPI(ABC):
    @abstractmethod
    def generate_video(self, query: str, aspect_ratio: str = "16:9", **kwargs) -> Tuple[str, str]:
        pass

    @abstractmethod
    def api_test(self) -> bool:
        pass


# --- Qwen API 实现类 ---
class DashScopeVideoAPI(BaseVideoAPI):
    def __init__(self,
                 api_key: str = None,
                 model: str = "wanx2.1-t2v-plus",
                 ):
        self.api_key = api_key if api_key is not None else config.DASHSCOPE_API_KEY
        self.model = model
        self.submit_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
        self.query_url_base = "https://dashscope.aliyuncs.com/api/v1/tasks/"
        self.session = requests.Session()

    def generate_video(self, query: str, aspect_ratio: str = "16:9", **kwargs) -> Tuple[str, str]:
        video_uuid = str(uuid.uuid4())

        quality = '480p' if self.model == 'wanx2.1-t2v-turbo' and kwargs.get('quality', '720p') == '480p' else '720p'
        try:
            resolution_str = _video_aspect_ratio_to_dims(aspect_ratio, quality)
        except ValueError as e:
            if self.model == 'wanx2.1-t2v-plus' and quality == '480p':
                raise ValueError(f"模型 {self.model} 不支持 480p 分辨率。") from e
            raise e

        print(f"Qwen: aspect_ratio '{aspect_ratio}' ({quality}) 转换为 size '{resolution_str}'")

        payload = {
            "model": self.model,
            "input": {"prompt": query},
            "parameters": {"size": resolution_str}
        }
        if 'seed' in kwargs:
            payload["parameters"]["seed"] = kwargs['seed']

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-DashScope-Async": "enable",
            "Content-Type": "application/json"
        }

        try:
            resp = self.session.post(self.submit_url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            task_id = resp.json()["output"]["task_id"]
        except Exception as e:
            raise ConnectionError(f"Qwen 提交任务失败: {e}") from e

        print(f"\U0001f3ac Qwen 提交任务成功，id={task_id}")
        return self._poll_task(task_id, video_uuid)

    def api_test(self) -> bool:
        """Tests DashScope Video API connectivity and basic task submission."""
        try:
            test_prompt = "A red car driving on a sunny road"
            # Use a common, valid resolution. Ensure it's supported by the chosen model.
            test_resolution = "1280*720"
            # Use the configured model for the test
            payload = {
                "model": self.model,
                "input": {"prompt": test_prompt},
                "parameters": {"size": test_resolution}
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "X-DashScope-Async": "enable",
                "Content-Type": "application/json"
            }

            print(f"DashScope Video API Test: Submitting task with prompt '{test_prompt}'...")
            submit_response = requests.post(self.submit_url, headers=headers, json=payload, timeout=20)
            submit_response.raise_for_status()

            data = submit_response.json()
            task_id = data.get("output", {}).get("task_id")

            if not task_id:
                print(
                    f"DashScope Video API Test: Submission successful (HTTP {submit_response.status_code}) but no task_id in response. Response: {data}")
                return False

            print(
                f"DashScope Video API Test: Task submitted successfully, task_id: {task_id}. Checking initial status...")
            # Brief pause and one quick status check
            time.sleep(3)  # Allow a moment for the task to register
            status_check_url = self.query_url_base + task_id
            status_resp = requests.get(status_check_url,
                                       headers={"Authorization": f"Bearer {self.api_key}"},
                                       timeout=15)
            status_resp.raise_for_status()
            result = status_resp.json()
            status = result.get("output", {}).get("task_status")

            if status in ["PENDING", "PROCESSING", "SUCCEEDED"]:  # SUCCEEDED is highly unlikely this fast for video
                print(f"DashScope Video API Test: PASSED. Task {task_id} initial status: '{status}'.")
                return True
            elif status == "FAILED":
                error_message = result.get("output", {}).get("message", "Task failed without specific message.")
                print(
                    f"DashScope Video API Test: FAILED. Task {task_id} immediately failed. Status: {status}, Message: {error_message}")
                return False
            else:  # Other statuses or missing status
                print(
                    f"DashScope Video API Test: AMBIGUOUS. Task {task_id} submitted, but initial status is '{status}'. This may or may not be an issue.")
                # Consider this a pass if the API accepted the task and didn't immediately fail it.
                return True

        except requests.exceptions.HTTPError as e:
            response_content = e.response.text if e.response is not None else "No response content"
            print(
                f"DashScope Video API Test: FAILED (HTTPError). Status: {e.response.status_code if e.response is not None else 'N/A'}. Message: {e}. Response: {response_content}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"DashScope Video API Test: FAILED (RequestException). Message: {e}")
            return False
        except Exception as e:  # Catch other errors like JSONDecodeError
            print(f"DashScope Video API Test: FAILED (Exception). Message: {e}")
            return False

    def _poll_task(self, task_id: str, video_uuid: str, interval: int = 10) -> Tuple[str, str]:
        while True:
            try:
                status_resp = self.session.get(self.query_url_base + task_id, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=30)
                status_resp.raise_for_status()
                result = status_resp.json()
            except Exception as e:
                print(f"Qwen 轮询失败，将重试: {e}")
                time.sleep(interval)
                continue

            status = result.get("output", {}).get("task_status")
            print(f"[Polling] Qwen 任务状态: {status}")
            if status == "SUCCEEDED":
                video_url = result["output"]["video_url"]
                return self._download_video(video_url, video_uuid)
            elif status == "FAILED":
                raise RuntimeError(f"❌ Qwen 视频生成失败: {result}")
            time.sleep(interval)

    @staticmethod
    def _download_video(url: str, video_uuid: str) -> Tuple[str, str]:
        filename = f"{video_uuid}.mp4"
        video_DIR = os.path.join(os.path.dirname(__file__), "results", 'video')
        os.makedirs(video_DIR, exist_ok=True)
        file_path = os.path.join(video_DIR, filename)
        print(f"📥 正在下载 Qwen 视频到 {file_path}")
        dl_resp = requests.get(url, stream=True, timeout=120)
        dl_resp.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in dl_resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Qwen 视频下载完成：{file_path}")
        return video_uuid, file_path


# --- Kling API 实现类 ---
class KlingVideoModelAPI(BaseVideoAPI):
    def __init__(self, access_key: str = None, secret_key: str = None, model: str = "kling-v2-master"):
        self.access_key = access_key if access_key is not None else config.KLING_ACCESS_KEY
        self.secret_key = secret_key if secret_key is not None else config.KLING_SECRET_KEY
        self.default_model_name = model
        self.api_base_url = "https://api-beijing.klingai.com"
        self.session = requests.Session()
        if not self.access_key or not self.secret_key:
            raise ValueError("Kling Access Key 和 Secret Key 不能为空。")
        
    def _get_auth_token(self) -> str:
        payload = {"iss": self.access_key, "exp": int(time.time()) + 1800, "nbf": int(time.time()) - 5}
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def generate_video(self, query: str, aspect_ratio: str = "16:9", **kwargs) -> Tuple[str, str]:
        duration = kwargs.get('duration', 5)  # 从 kwargs 获取 duration，默认为 5
        print(f"Kling: 使用 aspect_ratio '{aspect_ratio}', duration '{duration}s'")
        payload = {
            "model_name": kwargs.get('model_name', self.default_model_name),
            "prompt": query,
            "aspect_ratio": aspect_ratio,
            "duration": str(duration)
        }
        allowed_params = ["negative_prompt", "cfg_scale", "mode", "camera_control"]
        for param in allowed_params:
            if param in kwargs:
                payload[param] = kwargs[param]

        create_url = f"{self.api_base_url}/v1/videos/text2video"
        headers = {"Authorization": f"Bearer {self._get_auth_token()}", "Content-Type": "application/json"}

        try:
            resp = self.session.post(create_url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                raise Exception(f"Kling 任务创建失败: {data.get('message')}")
            task_id = data["data"]["task_id"]
        except Exception as e:
            raise ConnectionError(f"Kling 提交任务失败: {e}") from e

        print(f"\U0001f3ac Kling 提交任务成功，id={task_id}")
        video_uuid = str(uuid.uuid4())
        return self._poll_task(task_id, video_uuid)

    def _poll_task(self, task_id: str, video_uuid: str, interval: int = 10) -> Tuple[str, str]:
        url = f"{self.api_base_url}/v1/videos/text2video/{task_id}"
        while True:
            try:
                resp = self.session.get(url, headers={"Authorization": f"Bearer {self._get_auth_token()}"}, timeout=30)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                print(f"Kling 轮询失败，将重试: {e}")
                time.sleep(interval)
                continue

            if result.get("code") != 0:
                raise Exception(f"Kling 查询失败: {result.get('message')}")

            status = result["data"]["task_status"]
            print(f"[Polling] Kling 任务状态: {status}")
            if status == "succeed":
                video_url = result["data"]["task_result"]["videos"][0]["url"]
                return self._download_video(video_url, video_uuid)
            elif status == "failed":
                raise RuntimeError(f"❌ Kling 视频生成失败: {result['data'].get('task_status_msg', '未知错误')}")

            time.sleep(interval)

    @staticmethod
    def _download_video(url: str, video_uuid: str) -> Tuple[str, str]:
        filename = f"{video_uuid}.mp4"
        video_DIR = os.path.join(os.path.dirname(__file__), "results", 'video')
        os.makedirs(video_DIR, exist_ok=True)
        file_path = os.path.join(video_DIR, filename)
        print(f"📥 正在下载 Kling 视频到 {file_path}")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Kling 视频下载完成: {file_path}")
        return video_uuid, file_path

    def api_test(self) -> bool:
        print("--- 正在测试 Kling Video API ---")
        try:
            payload = {"model_name": self.default_model_name, "prompt": "a small bird", "aspect_ratio": "16:9",
                       "duration": "5"}
            headers = {"Authorization": f"Bearer {self._get_auth_token()}", "Content-Type": "application/json"}
            resp = self.session.post(f"{self.api_base_url}/v1/videos/text2video", headers=headers, json=payload,
                                     timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == 0 and data.get("data", {}).get("task_id"):
                print(f"✅ Kling Video API 测试成功: 任务已成功提交, task_id: {data['data']['task_id']}")
                return True
            else:
                print(
                    f"❌ Kling Video API 测试失败: 提交成功但业务出错. Code: {data.get('code')}, Message: {data.get('message')}")
                return False
        except Exception as e:
            print(f"❌ Kling Video API 测试过程中发生异常: {e}")
            return False


# --- Jimeng API 实现类 ---
class JimengVideoModelAPI(BaseVideoAPI):
    REQUEST_METHOD = "POST"  # 即梦的提交和查询任务都是 POST
    HOST = "visual.volcengineapi.com"
    ENDPOINT = "https://visual.volcengineapi.com"
    REGION = "cn-north-1"  # 请确认你的服务区域
    SERVICE = "cv"         # 请确认你的服务名称
    REQ_KEY_T2V = "jimeng_vgfm_t2v_l20"

    def __init__(self, access_key_id: str = None, secret_access_key: str = None, **kwargs):
        self.access_key_id = access_key_id if access_key_id is not None else config.JIMENG_ACCESS_KEY
        self.secret_access_key = secret_access_key if secret_access_key is not None else config.JIMENG_SECRET_ACCESS_KEY
        self.session = requests.Session()
        if not self.access_key_id or not self.secret_access_key:
            raise ValueError("即梦 API 的 Access Key 或 Secret Key 未配置。")

    @staticmethod
    def _format_query_string(parameters: Dict[str, Any]) -> str:
        if not parameters: return ""
        return "&".join([f"{k}={v}" for k, v in sorted(parameters.items())])

    @staticmethod
    def _hmac_sha256_sign(key: bytes, msg: str) -> bytes:
        return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

    @staticmethod
    def _get_derived_signing_key(secret_key: str, date_stamp: str, region_name: str, service_name: str) -> bytes:
        k_secret = secret_key.encode('utf-8')
        k_date = JimengVideoModelAPI._hmac_sha256_sign(k_secret, date_stamp)
        k_region = JimengVideoModelAPI._hmac_sha256_sign(k_date, region_name)
        k_service = JimengVideoModelAPI._hmac_sha256_sign(k_region, service_name)
        return JimengVideoModelAPI._hmac_sha256_sign(k_service, 'request')

    def _create_signed_request_elements(self, query_params: Dict[str, Any], body_params: Dict[str, Any]) -> Tuple[str, Dict[str, str], str]:
        t = datetime.datetime.utcnow()
        amz_date = t.strftime('%Y%m%dT%H%M%SZ')
        date_stamp = t.strftime('%Y%m%d')
        canonical_uri = '/'
        canonical_query_string = self._format_query_string(query_params)
        request_body_str = json.dumps(body_params) if body_params else ""
        payload_hash = hashlib.sha256(request_body_str.encode('utf-8')).hexdigest()
        headers_to_sign = {'host': self.HOST, 'content-type': 'application/json', 'x-date': amz_date,
                           'x-content-sha256': payload_hash}
        signed_headers_str = ';'.join(sorted(headers_to_sign.keys()))
        canonical_headers_str = "".join([f"{k}:{v}\n" for k, v in sorted(headers_to_sign.items())])
        canonical_request = f"{self.REQUEST_METHOD}\n{canonical_uri}\n{canonical_query_string}\n{canonical_headers_str}\n{signed_headers_str}\n{payload_hash}"
        credential_scope = f"{date_stamp}/{self.REGION}/{self.SERVICE}/request"
        string_to_sign = f"HMAC-SHA256\n{amz_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        signing_key = self._get_derived_signing_key(self.secret_access_key, date_stamp, self.REGION, self.SERVICE)
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        auth_header = f"HMAC-SHA256 Credential={self.access_key_id}/{credential_scope}, SignedHeaders={signed_headers_str}, Signature={signature}"
        final_headers = {'Authorization': auth_header, 'Content-Type': 'application/json', 'X-Date': amz_date,
                         'X-Content-Sha256': payload_hash, 'Host': self.HOST}
        request_url = f"{self.ENDPOINT}?{canonical_query_string}"
        return request_url, final_headers, request_body_str

    def generate_video(self, query: str, aspect_ratio: str = "16:9", **kwargs) -> Tuple[str, str]:
        print(f"Jimeng: 使用 aspect_ratio '{aspect_ratio}'")

        query_params = {"Action": "CVSync2AsyncSubmitTask", "Version": "2022-08-31"}
        body_params = {"req_key": self.REQ_KEY_T2V, "prompt": query[:150], "aspect_ratio": aspect_ratio}
        if 'seed' in kwargs:
            body_params["seed"] = kwargs['seed']

        try:
            url, headers, body = self._create_signed_request_elements(query_params, body_params)
            resp = self.session.post(url, headers=headers, data=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 10000:
                raise Exception(f"即梦任务创建失败: {data.get('message')}")
            task_id = data["data"]["task_id"]
        except Exception as e:
            raise ConnectionError(f"即梦提交任务失败: {e}") from e

        print(f"\U0001f3ac 即梦提交任务成功，id={task_id}")
        video_uuid = str(uuid.uuid4())
        return self._poll_task(task_id, video_uuid)

    def _poll_task(self, task_id: str, video_uuid: str, interval: int = 10, max_retries: int = 180) -> Tuple[str, str]:
        query_params = {"Action": "CVSync2AsyncGetResult", "Version": "2022-08-31"}
        body_params = {"req_key": self.REQ_KEY_T2V, "task_id": task_id}

        for i in range(max_retries):
            try:
                url, headers, body = self._create_signed_request_elements(query_params, body_params)
                resp = self.session.post(url, headers=headers, data=body, timeout=30)
                resp.raise_for_status()
                result = resp.json()
            except Exception as e:
                print(f"即梦轮询失败，将重试: {e}")
                time.sleep(interval)
                continue

            if result.get("code") != 10000:
                print(f"即梦查询返回业务错误，继续轮询: {result.get('message')}")
                time.sleep(interval)
                continue

            task_data = result.get("data", {})
            status = task_data.get("status")
            print(f"[Polling] 即梦任务状态: {status}")

            if status == "done":
                video_url = task_data.get("video_url")
                if not video_url and "resp_data" in task_data:
                    try:
                        resp_data_json = json.loads(task_data["resp_data"])
                        video_url = resp_data_json.get("urls", [None])[0]
                    except Exception:
                        pass
                if video_url:
                    return self._download_video(video_url, video_uuid)
                else:
                    raise RuntimeError(f"❌ 即梦视频状态为 'done' 但未找到 video_url。")
            elif status == "failed":
                raise RuntimeError(f"❌ 即梦视频生成失败: {task_data.get('message', '未知错误')}")

            time.sleep(interval)
        raise TimeoutError("即梦视频生成超时")

    @staticmethod
    def _download_video(url: str, video_uuid: str) -> Tuple[str, str]:
        filename = f"{video_uuid}.mp4"
        video_DIR = os.path.join(os.path.dirname(__file__), "results", 'video')
        os.makedirs(video_DIR, exist_ok=True)
        file_path = os.path.join(video_DIR, filename)
        print(f"📥 正在下载即梦视频到 {file_path}")
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ 即梦视频下载完成: {file_path}")
        return video_uuid, file_path

    def api_test(self) -> bool:
        print("--- 正在测试 Jimeng Video API ---")
        try:
            query_params = {"Action": "CVSync2AsyncSubmitTask", "Version": "2022-08-31"}
            body_params = {"req_key": self.REQ_KEY_T2V, "prompt": "a cat", "aspect_ratio": "1:1"}
            url, headers, body = self._create_signed_request_elements(query_params, body_params)
            resp = self.session.post(url, headers=headers, data=body, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == 10000 and data.get("data", {}).get("task_id"):
                print(f"✅ 即梦 Video API 测试成功: 任务已成功提交, task_id: {data['data']['task_id']}")
                return True
            else:
                print(
                    f"❌ 即梦 Video API 测试失败: 提交成功但业务出错. Code: {data.get('code')}, Message: {data.get('message')}")
                return False
        except Exception as e:
            print(f"❌ 即梦 Video API 测试过程中发生异常: {e}")
            return False


# --- 总调度类 ---
class VideoGenerationManager:
    _registry = {
        "kling": KlingVideoModelAPI,
        "qwen": DashScopeVideoAPI,
        "jimeng": JimengVideoModelAPI,
    }

    def __init__(self, use_api: str, **kwargs):
        if use_api not in self._registry:
            raise ValueError(f"不支持的视频生成 API: {use_api}")
        self.use_api = use_api
        self.client = self._registry[use_api](**kwargs)

    def generate_video(self, query: str, aspect_ratio: str = "16:9", **kwargs) -> Tuple[str, str]:
        return self.client.generate_video(query, aspect_ratio=aspect_ratio, **kwargs)

    def api_test(self) -> bool:
        print(f"\n--- Manager 正在测试视频 API: {self.use_api.upper()} ---")
        result = self.client.api_test()
        print(f"--- Manager 测试视频 API {self.use_api.upper()} {'通过' if result else '失败'} ---")
        return result


if __name__ == '__main__':
    # --- 测试所有 API 的连通性 ---
    print("====== 开始视频 API 连通性测试 ======")
    try:
        qwen_manager = VideoGenerationManager(use_api='qwen')
        qwen_manager.api_test()
    except Exception as e:
        print(f"初始化 Qwen Manager 失败: {e}")

    try:
        kling_manager = VideoGenerationManager(use_api='kling')
        kling_manager.api_test()
    except Exception as e:
        print(f"初始化 Kling Manager 失败: {e}")

    try:
        jimeng_manager = VideoGenerationManager(use_api='jimeng')
        jimeng_manager.api_test()
    except Exception as e:
        print(f"初始化 Jimeng Manager 失败: {e}")

    print("\n====== 视频 API 连通性测试结束 ======")

    # --- 演示统一接口调用 ---
    print("\n====== 开始演示统一接口调用 ======")
    PROMPT = "一只穿着宇航服的猫，漂浮在太空中，背景是星云和地球，电影级质感"

    # 1. 调用 Qwen
    print("\n=== 演示调用 Qwen (9:16) ===")
    try:
        manager = VideoGenerationManager(use_api='qwen')
        str_uuid, path = manager.generate_video(PROMPT, aspect_ratio="9:16")
        print(f"Qwen 视频生成成功: UUID={str_uuid}, Path={path}")
    except Exception as e:
        print(f"Qwen 视频生成失败: {e}")

    # 2. 调用 Kling (通过 kwargs 传递 duration)
    print("\n=== 演示调用 Kling (1:1, 10秒) ===")
    try:
        manager = VideoGenerationManager(use_api='kling')
        str_uuid, path = manager.generate_video(PROMPT, aspect_ratio="9:16")
        print(f"Kling 视频生成成功: UUID={str_uuid}, Path={path}")
    except Exception as e:
        print(f"Kling 视频生成失败: {e}")

    # 3. 调用 Jimeng
    print("\n=== 演示调用 Jimeng (4:3) ===")
    try:
        manager = VideoGenerationManager(use_api='jimeng')
        str_uuid, path = manager.generate_video(PROMPT, aspect_ratio="9:16")
        print(f"Jimeng 视频生成成功: UUID={str_uuid}, Path={path}")
    except Exception as e:
        print(f"Jimeng 视频生成失败: {e}")

    print("\n====== 演示结束 ======")

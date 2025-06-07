# -*- coding: utf-8 -*-
"""
统一图片生成 API：Qwen、Kling 与 Jimeng
所有接口统一使用 aspect_ratio 参数控制尺寸
生成图片时以 UUID 命名，并返回 (uuid, file_path) 元组
"""
import os
import time
import jwt
import uuid
import base64
import requests
from http import HTTPStatus
from typing import Tuple, Dict, Any
from dashscope import ImageSynthesis
from abc import ABC, abstractmethod
from dataclasses import dataclass
from LLMDispatch.LLMHandle import config

# 极梦签名相关的辅助模块
import datetime
import hashlib
import hmac
import json


# --- 统一的辅助函数 ---
def _format_query_string(parameters: Dict[str, Any]) -> str:
    if not parameters:
        return ""
    query_parts = []
    for key in sorted(parameters.keys()):
        query_parts.append(f"{key}={str(parameters[key])}")
    return "&".join(query_parts)


def _hmac_sha256_sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


def _get_derived_signing_key(secret_key: str, date_stamp: str, region_name: str, service_name: str) -> bytes:
    k_secret_utf8 = secret_key.encode('utf-8')
    k_date = _hmac_sha256_sign(k_secret_utf8, date_stamp)
    k_region = _hmac_sha256_sign(k_date, region_name)
    k_service = _hmac_sha256_sign(k_region, service_name)
    k_signing = _hmac_sha256_sign(k_service, 'request')
    return k_signing


def _convert_aspect_ratio_to_dims(
    aspect_ratio: str,
    target_long_edge: int = 1024,
    min_dim: int = 256,
    max_dim: int = 1440
) -> Tuple[int, int]:
    """
    将宽高比字符串转换为具体的(width, height)像素元组。
    """
    try:
        w_ratio, h_ratio = map(int, aspect_ratio.split(':'))
    except (ValueError, TypeError):
        raise ValueError(f"无效的 aspect_ratio 格式: '{aspect_ratio}'. 请使用 'W:H' 格式, 例如 '16:9'。")

    if w_ratio <= 0 or h_ratio <= 0:
        raise ValueError("宽高比必须为正数。")

    if w_ratio >= h_ratio:  # 横向或方形
        width = target_long_edge
        height = int(round(width * h_ratio / w_ratio))
    else:  # 纵向
        height = target_long_edge
        width = int(round(height * w_ratio / h_ratio))

    # 确保尺寸在API允许的范围内，并调整为8的倍数（很多模型推荐）
    width = max(min_dim, min(width, max_dim))
    height = max(min_dim, min(height, max_dim))
    width = (width // 8) * 8
    height = (height // 8) * 8

    return width, height


# --- 抽象基类 ---
@dataclass
class BasePictureAPI(ABC):
    @abstractmethod
    def generate_image(self, query: str, aspect_ratio: str = "1:1", **kwargs) -> Tuple[str, str]:
        pass

    @abstractmethod
    def api_test(self) -> bool:
        pass


# --- Qwen 实现类 ---
class QWENPictureAPI(BasePictureAPI):

    def __init__(self, api_key=None, model="wanx2.1-t2i-plus"):
        self.api_key = api_key if api_key is not None else config.QWEN_IMG_API_KEY
        self.model = model

    def generate_image(self, query: str, aspect_ratio: str = "1:1", n: int = 1, **kwargs) -> Tuple[str, str]:
        image_uuid = str(uuid.uuid4())
        filename = f"{image_uuid}.jpg"
        picture_DIR = os.path.join(os.path.dirname(__file__), "results", 'picture')
        os.makedirs(picture_DIR, exist_ok=True)  # 确保目录存在
        file_path = os.path.join(picture_DIR, filename)

        # 根据 aspect_ratio 计算 Qwen 需要的 size 字符串
        width, height = _convert_aspect_ratio_to_dims(
            aspect_ratio,
            target_long_edge=1024,
            min_dim=512,
            max_dim=1440
        )
        qwen_size_str = f"{width}*{height}"
        print(f"Qwen: aspect_ratio '{aspect_ratio}' 转换为 size '{qwen_size_str}'")

        call_params = {
            'api_key': self.api_key,
            'model': self.model,
            'prompt': query,
            'n': n,
            'size': qwen_size_str,
            'prompt_extend': True
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

    def api_test(self) -> bool:
        """Tests Qwen API connectivity and basic request processing."""
        try:
            test_prompt = "Test cat"
            # Use a size known to be valid for the model, e.g., wanx-v1 common sizes
            test_size = "1024*1024"

            call_params = {
                'api_key': self.api_key,
                'model': self.model,
                'prompt': test_prompt,
                'n': 1,
                'size': test_size,
            }
            rsp = ImageSynthesis.call(**call_params)

            if rsp.status_code == HTTPStatus.OK:
                task_status = getattr(rsp.output, 'task_status', None)
                if task_status in ['SUCCEEDED', 'PROCESSING', 'PENDING']:
                    print(f"QWEN API Test: Successful, task status: {task_status}")
                    return True
                else:
                    error_message = getattr(rsp.output, 'message', 'No message')
                    print(f"QWEN API Test: HTTP OK, but task status is '{task_status}'. Message: {error_message}")
                    return False
            else:
                print(
                    f"QWEN API Test: Request failed. Status: {rsp.status_code}, Code: {rsp.code}, Message: {rsp.message}")
                return False
        except Exception as e:
            print(f"QWEN API Test: Exception during API call: {e}")
            return False


# --- Kling 实现类 ---
class KlingPictureAPI(BasePictureAPI):
    API_BASE_URL = "https://api-beijing.klingai.com"

    def __init__(self,
                 access_key: str = None,
                 secret_key: str = None,
                 model: str = "kling-v2",
                 polling_interval: int = 5,  # 轮询间隔（秒）
                 polling_timeout: int = 600  # 轮询总超时（秒）
                 ):
        super().__init__()
        self.access_key = access_key if access_key is not None else config.KLING_ACCESS_KEY
        self.secret_key = secret_key if secret_key is not None else config.KLING_SECRET_KEY
        self.default_model_name = model

        # 网络请求相关配置
        self.polling_interval = polling_interval
        self.polling_timeout = polling_timeout
        self.session = requests.Session()

        if not self.access_key or not self.secret_key:
            raise ValueError("Kling Access Key 和 Secret Key 不能为空。")

    def _get_auth_token(self) -> str:
        """生成并返回 JWT Token 字符串"""
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,  # Token 有效期 30 分钟
            "nbf": int(time.time()) - 5  # Token 5 秒前生效，防止时间误差
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def generate_image(self,
                       prompt: str,
                       n: int = 1,
                       model_name: str = None,
                       negative_prompt: str = None,
                       aspect_ratio: str = "16:9",
                       image: str = None,  # 图生图：图片的 Base64 或 URL
                       image_reference: str = None,  # 'subject' 或 'face'
                       image_fidelity: float = None,
                       human_fidelity: float = None,
                       **kwargs) -> Tuple[str, str]:
        """
        根据文本或图片生成图片。
        :param prompt: 正向提示词
        :param n: 生成图片数量 (注意：当前实现只返回第一张图片)
        :param model_name: 模型名称, 如 'kling-v1', 'kling-v2'
        :param negative_prompt: 负向提示词
        :param aspect_ratio: 宽高比, 如 '16:9', '1:1'
        :param image: 图生图的参考图，可以是 URL 或 Base64 编码字符串
        :param image_reference: 图片参考类型, 'subject' 或 'face'
        :param image_fidelity: 对参考图的遵循度
        :param human_fidelity: 面部相似度
        :return: (uuid, file_path) 元组，指向第一张生成的图片
        """
        # 1. 准备请求体 (Payload)
        payload = {
            "model_name": model_name if model_name is not None else self.default_model_name,
            "prompt": prompt,
            "n": n,
            "aspect_ratio": aspect_ratio
        }
        # 动态添加可选参数
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt
        if image:
            payload["image"] = image
        if image_reference:
            payload["image_reference"] = image_reference
        if image_fidelity is not None:
            payload["image_fidelity"] = image_fidelity
        if human_fidelity is not None:
            payload["human_fidelity"] = human_fidelity

        # 2. 发起创建任务请求
        create_url = f"{self.API_BASE_URL}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {self._get_auth_token()}",
            "Content-Type": "application/json"
        }

        try:
            response = self.session.post(create_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                raise Exception(f"Kling 创建任务失败: Code={data.get('code')}, Message={data.get('message')}")

            task_id = data["data"]["task_id"]
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Kling API 请求失败: {e}") from e

        # 3. 轮询任务结果
        query_url = f"{self.API_BASE_URL}/v1/images/generations/{task_id}"
        start_time = time.time()

        while time.time() - start_time < self.polling_timeout:
            time.sleep(self.polling_interval)

            headers = {"Authorization": f"Bearer {self._get_auth_token()}"}
            try:
                result_resp = self.session.get(query_url, headers=headers, timeout=30)
                result_resp.raise_for_status()
                result_data = result_resp.json()

                if result_data.get("code") != 0:
                    raise Exception(
                        f"Kling 查询任务失败: Code={result_data.get('code')}, Message={result_data.get('message')}")

                task_info = result_data.get("data", {})
                status = task_info.get("task_status")

                if status == "succeed":
                    # 任务成功，下载图片
                    images = task_info.get("task_result", {}).get("images", [])
                    if not images:
                        raise Exception("Kling 任务成功但未返回图片 URL。")

                    img_url = images[0]["url"]
                    image_uuid = str(uuid.uuid4())
                    filename = f"{image_uuid}.png"  # Kling 返回的通常是 png
                    picture_DIR = os.path.join(os.path.dirname(__file__), "results", 'picture')
                    os.makedirs(picture_DIR, exist_ok=True)
                    file_path = os.path.join(picture_DIR, filename)

                    img_download_resp = self.session.get(img_url, timeout=120)
                    img_download_resp.raise_for_status()
                    with open(file_path, "wb") as f:
                        f.write(img_download_resp.content)
                    print(f"✅ Kling 图片生成成功并保存至: {file_path}")
                    return image_uuid, file_path

                elif status == "failed":
                    # 任务失败，抛出包含具体原因的异常
                    error_msg = task_info.get("task_status_msg", "未知失败原因")
                    raise Exception(f"❌ Kling 图片生成失败: {error_msg}")

                # 如果是 processing 或 submitted，则继续轮询
                print(f"Kling 任务状态: {status}, 正在等待...")

            except requests.exceptions.RequestException as e:
                print(f"Kling 轮询请求时发生网络错误: {e}, 将继续尝试...")

        raise TimeoutError(f"Kling 图片生成超时（超过 {self.polling_timeout} 秒）。")

    def api_test(self) -> bool:
        """测试 Kling API 的连通性和基本功能。"""
        print("--- 正在测试 Kling API ---")
        try:
            headers = {
                "Authorization": f"Bearer {self._get_auth_token()}",
                "Content-Type": "application/json"
            }
            payload = {
                "model_name": self.default_model_name,
                "prompt": "a white cat",
                "n": 1,
                "aspect_ratio": "1:1",
            }

            create_url = f"{self.API_BASE_URL}/v1/images/generations"
            response = self.session.post(create_url, headers=headers, json=payload, timeout=20)

            if response.status_code == 200:
                data = response.json()
                if data.get("code") == 0 and data.get("data", {}).get("task_id"):
                    print("✅ Kling API 测试成功: 任务已成功提交。")
                    return True
                else:
                    print(
                        f"❌ Kling API 测试失败: 提交成功但业务出错。Code: {data.get('code')}, Message: {data.get('message')}")
                    return False
            else:
                print(f"❌ Kling API 测试失败: 请求错误。Status: {response.status_code}, Response: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Kling API 测试过程中发生异常: {e}")
            return False


# --- Jimeng 实现类 ---
class JimengPictureAPI(BasePictureAPI):
    REQUEST_METHOD = "POST"
    HOST = "visual.volcengineapi.com"
    ENDPOINT = "https://visual.volcengineapi.com"
    REGION = "cn-north-1"
    SERVICE = "cv"
    REQ_KEY_IMG_GEN = "jimeng_high_aes_general_v21_L"

    def __init__(self,
                 access_key_id: str = None,
                 secret_access_key: str = None,
                 **kwargs
                 ):
        self.access_key_id = access_key_id if access_key_id is not None else config.JIMENG_ACCESS_KEY
        self.secret_access_key = secret_access_key if secret_access_key is not None else config.JIMENG_SECRET_ACCESS_KEY
        self.defaults = kwargs
        if not self.access_key_id or not self.secret_access_key:
            raise ValueError("Jimeng Access Key 和 Secret Key 不能为空。")

    def _create_signed_request_elements(self, query_params: Dict[str, Any], body_params: Dict[str, Any]) -> Tuple[str, Dict[str, str], str]:
        t = datetime.datetime.utcnow()
        amz_date = t.strftime('%Y%m%dT%H%M%SZ')
        date_stamp = t.strftime('%Y%m%d')
        canonical_uri = '/'
        canonical_query_string = _format_query_string(query_params)
        request_body_str = json.dumps(body_params) if body_params else ""
        payload_hash = hashlib.sha256(request_body_str.encode('utf-8')).hexdigest()
        headers_to_sign = {'host': self.HOST, 'content-type': 'application/json', 'x-date': amz_date, 'x-content-sha256': payload_hash}
        signed_headers_list = sorted(headers_to_sign.keys())
        signed_headers_str = ';'.join(signed_headers_list)
        canonical_headers_str = "".join([f"{key}:{headers_to_sign[key]}\n" for key in signed_headers_list])
        canonical_request = f"{self.REQUEST_METHOD}\n{canonical_uri}\n{canonical_query_string}\n{canonical_headers_str}\n{signed_headers_str}\n{payload_hash}"
        algorithm = 'HMAC-SHA256'
        credential_scope = f"{date_stamp}/{self.REGION}/{self.SERVICE}/request"
        hashed_canonical_request = hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
        string_to_sign = f"{algorithm}\n{amz_date}\n{credential_scope}\n{hashed_canonical_request}"
        signing_key = _get_derived_signing_key(self.secret_access_key, date_stamp, self.REGION, self.SERVICE)
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        authorization_header = f"{algorithm} Credential={self.access_key_id}/{credential_scope}, SignedHeaders={signed_headers_str}, Signature={signature}"
        final_headers = {'Host': self.HOST, 'Content-Type': headers_to_sign['content-type'], 'X-Date': amz_date, 'X-Content-Sha256': payload_hash, 'Authorization': authorization_header}
        request_url = f"{self.ENDPOINT}?{canonical_query_string}"
        return request_url, final_headers, request_body_str

    def generate_image(self, query: str, aspect_ratio: str = "1:1", **kwargs) -> Tuple[str, str]:
        image_uuid_str = str(uuid.uuid4())
        filename = f"{image_uuid_str}.jpg"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        picture_dir = os.path.join(base_dir, "results", 'picture')
        os.makedirs(picture_dir, exist_ok=True)
        file_path = os.path.join(picture_dir, filename)

        final_width, final_height = _convert_aspect_ratio_to_dims(aspect_ratio, target_long_edge=768, min_dim=256,
                                                                  max_dim=768)
        print(f"Jimeng: aspect_ratio '{aspect_ratio}' 转换为 width={final_width}, height={final_height}")

        query_params = {"Action": "CVProcess", "Version": "2022-08-31"}
        body_params = {
            "req_key": self.REQ_KEY_IMG_GEN, "prompt": query, "width": final_width, "height": final_height,
            "seed": kwargs.get('seed', self.defaults.get('seed', -1)),
            "use_pre_llm": kwargs.get('use_pre_llm', self.defaults.get('use_pre_llm', True)),
            "use_sr": kwargs.get('use_sr', self.defaults.get('use_sr', True)),
            "return_url": kwargs.get('return_url', self.defaults.get('return_url', True))
        }

        try:
            request_url, headers, request_body_str = self._create_signed_request_elements(query_params=query_params,
                                                                                          body_params=body_params)
            response = requests.post(request_url, headers=headers, data=request_body_str, timeout=120)
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.HTTPError as http_err:
            error_text = http_err.response.text if http_err.response is not None else "无响应体"
            raise RuntimeError(f"即梦图片生成 HTTP 错误: {http_err} - {error_text}") from http_err
        except Exception as e:
            raise RuntimeError(f"即梦图片生成时发生错误: {e}") from e

        if response_data.get("code") != 10000:
            error_message = response_data.get("message", "API 返回业务错误但未提供消息")
            raise Exception(f"❌ 即梦图片生成失败 (业务层面): {error_message} (Code: {response_data.get('code')})")

        data_field = response_data.get("data", {})
        image_urls = data_field.get("image_urls", [])
        base64_data = data_field.get("binary_data_base64", [])

        if image_urls:
            img_url = image_urls[0]
            dl_response = requests.get(img_url, stream=True, timeout=120)
            dl_response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in dl_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ 即梦图片(URL)下载完成: {file_path}")
            return image_uuid_str, file_path
        elif base64_data:
            img_bytes = base64.b64decode(base64_data[0])
            with open(file_path, "wb") as f:
                f.write(img_bytes)
            print(f"✅ 即梦图片(Base64)保存完成: {file_path}")
            return image_uuid_str, file_path
        else:
            raise Exception(f"即梦图片生成成功但未找到任何图片数据。响应: {response_data}")

    def api_test(self) -> bool:
        print("即梦 Picture API 测试 (手动签名)...")
        try:
            test_prompt = "一只微笑的卡通太阳"
            query_params = {"Action": "CVProcess", "Version": "2022-08-31"}
            body_params = {
                "req_key": self.REQ_KEY_IMG_GEN,
                "prompt": test_prompt,
                "width": 256,  # 使用最小尺寸以加快测试
                "height": 256,
                "seed": 888,
                "use_sr": False,  # 关闭超分以加快测试
                "return_url": True  # 我们期望测试 URL 返回
            }

            request_url, headers, request_body_str = self._create_signed_request_elements(
                query_params=query_params, body_params=body_params
            )

            response = requests.post(request_url, headers=headers, data=request_body_str, timeout=45)  # 允许稍长一点时间

            print(f"即梦 Picture API 测试: 响应状态码: {response.status_code}")
            # print(f"响应体: {response.text}") # 调试时开启

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("code") == 10000:  # 业务成功
                    data_field = response_data.get("data", {})
                    # 检查是否有图片 URL 或 Base64 数据返回
                    if (data_field.get("image_urls") and len(data_field["image_urls"]) > 0) or \
                            (data_field.get("binary_data_base64") and len(data_field["binary_data_base64"]) > 0):
                        print(f"即梦 Picture API 测试: 通过。成功接收到图片数据/链接。")
                        return True
                    else:
                        print(f"即梦 Picture API 测试: 业务成功但未返回图片数据。响应: {response_data}")
                        return False
                else:  # 业务层面错误
                    error_message = response_data.get("message", "未知业务错误")
                    print(
                        f"即梦 Picture API 测试: 失败 (业务层面错误)。Code: {response_data.get('code')}, Message: {error_message}")
                    return False
            else:  # HTTP 层面错误
                print(f"即梦 Picture API 测试: 失败 (HTTP 错误)。状态码: {response.status_code}, 响应: {response.text}")
                return False
        except Exception as e:
            print(f"即梦 Picture API 测试过程中发生异常: {e}")
            return False


# --- 总调度类 ---
class PictureGenerationManager:
    _registry = {
        "qwen": QWENPictureAPI,
        "kling": KlingPictureAPI,
        "jimeng": JimengPictureAPI
    }

    def __init__(self, use_api: str = "qwen", **kwargs):
        if use_api not in self._registry:
            raise ValueError(f"不支持的图片生成 API: {use_api}")
        self.use_api = use_api
        self.client = self._registry[use_api](**kwargs)

    def generate_image(self, query: str, aspect_ratio: str = "1:1", **kwargs) -> Tuple[str, str]:
        return self.client.generate_image(query, aspect_ratio=aspect_ratio, **kwargs)

    def api_test(self) -> bool:  # 添加 manager 的 api_test 方法
        print(f"--- Manager 正在测试图片 API: {self.use_api} ---")
        if hasattr(self.client, 'api_test'):
            result = self.client.api_test()
            print(f"--- Manager 测试图片 API {self.use_api} {'通过' if result else '失败'} ---")
            return result
        else:
            print(f"警告: {self.use_api} 客户端没有实现 api_test 方法。测试跳过。")
            return False


if __name__ == '__main__':
    # --- 测试所有 API 的连通性 ---
    print("====== 开始 API 连通性测试 ======")
    try:
        qwen_manager = PictureGenerationManager(use_api='qwen')
        qwen_manager.api_test()
    except Exception as e:
        print(f"初始化 Qwen Manager 失败: {e}")

    try:
        kling_manager = PictureGenerationManager(use_api='kling')
        kling_manager.api_test()
    except Exception as e:
        print(f"初始化 Kling Manager 失败: {e}")

    try:
        jimeng_manager = PictureGenerationManager(use_api='jimeng')
        jimeng_manager.api_test()
    except Exception as e:
        print(f"初始化 Jimeng Manager 失败: {e}")

    print("\n====== API 连通性测试结束 ======")

    # --- 演示统一 aspect_ratio 参数调用 ---
    print("\n====== 开始演示统一接口调用 ======")
    PROMPT = "一只穿着宇航服的猫，漂浮在太空中，背景是星云和地球，数字绘画风格"
    aspect_ratio_list = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9"]
    for aspect_ratio in aspect_ratio_list:
        print(f"开始生成：{aspect_ratio}图片。。。。。。。。")
        # 使用 Kling 生成图像
        print("\n=== 演示调用 Kling ===")
        manager_kling = PictureGenerationManager(use_api='kling')
        str_uuid, path = manager_kling.generate_image(PROMPT, aspect_ratio=aspect_ratio)
        print(f"Kling 生成成功: UUID={str_uuid}, Path={path}")

        # 使用 Qwen 生成图像
        print("\n=== 演示调用 Qwen ===")
        manager_qwen = PictureGenerationManager(use_api='qwen')
        str_uuid, path = manager_qwen.generate_image(PROMPT, aspect_ratio=aspect_ratio)
        print(f"Qwen 生成成功: UUID={str_uuid}, Path={path}")

        # 使用 Jimeng 生成图像
        print("\n=== 演示调用 Jimeng ===")
        manager_jimeng = PictureGenerationManager(use_api='jimeng')
        str_uuid, path = manager_jimeng.generate_image(PROMPT, aspect_ratio=aspect_ratio)
        print(f"Jimeng 生成成功: UUID={str_uuid}, Path={path}")

    print("\n====== 演示结束 ======")

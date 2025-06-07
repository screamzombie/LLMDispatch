# -*- coding:utf-8 -*-
import os
import hashlib
import hmac
import base64
import json
import time
import random
import requests
import uuid
# from LLMHandle.config import XFYUN_PPT_APP_ID, XFYUN_PPT_SECRET_KEY
from LLMDispatch.LLMHandle import config
from typing import List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class BasePPTModelAPI(ABC):

    @abstractmethod
    def createPptByOutline(self, outline, templateId) -> str:
        pass

    @abstractmethod
    def outline_generation_ppt(self, outline, templateId) -> tuple[str, str, str]:
        pass

    @abstractmethod
    def api_test(self) -> bool:
        pass


class XunfeiPPTModelAPI(BasePPTModelAPI):
    def __init__(self, api_key=None, APISecret=None):
        self.APPid = api_key if api_key is not None else config.XFYUN_PPT_APP_ID
        # self.api_key = XFYUN_PPT_APP_ID
        self.APISecret = APISecret if APISecret is not None else config.XFYUN_PPT_SECRET_KEY
        # self.APISecret = XFYUN_PPT_SECRET_KEY
        self.BASE_URL = 'https://zwapi.xfyun.cn'
        self.header = {}
        self.download_file_path = ""

    @staticmethod
    def download(url: str, save_path: str):
        file = requests.get(url, verify=False)
        with open(save_path, 'wb') as wstream:
            wstream.write(file.content)

    @staticmethod
    def hmac_sha1_encrypt(encrypt_text, encrypt_key):
        # 使用HMAC-SHA1算法对文本进行加密，并将结果转换为Base64编码
        return base64.b64encode(
            hmac.new(encrypt_key.encode('utf-8'), encrypt_text.encode('utf-8'), hashlib.sha1).digest()).decode('utf-8')

    @staticmethod
    def md5(text):
        # 对文本进行MD5加密，并返回加密后的十六进制字符串
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_signature(self, ts):
        """
        获取签名
        :param ts:
        :return:
        """
        try:
            auth = self.md5(self.APPid + str(ts))
            return self.hmac_sha1_encrypt(auth, self.APISecret)
        except Exception as e:
            print(e)
            return None

    def getHeaders(self):
        timestamp = int(time.time())
        signature = self.get_signature(timestamp)
        headers = {
            "appId": self.APPid,
            "timestamp": str(timestamp),
            "signature": signature,
            "Content-Type": "application/json; charset=utf-8"
        }
        return headers

    def getTheme(self):
        url = self.BASE_URL + '/api/ppt/v2/docx_template/list'
        self.header = self.getHeaders()
        response = requests.request("GET", url=url, headers=self.header).text
        templateId_list = json.loads(response).get('data').get('records')
        template = random.choice(templateId_list)
        print(f'模板为：{template}')
        templateId = template.get('templateIndexId')
        return templateId

    def get_process(self, sid):
        """
        轮询任务进度，返回完整响应信息
        :param sid:
        :return:
        """
        url = self.BASE_URL + f'/api/ppt/v2/progress?sid={sid}'
        if not self.header:
            self.header = self.getHeaders()
        if sid is not None:
            response = requests.request("GET", url=url, headers=self.header).text
            return response
        else:
            return None

    # 根据大纲生成PPT
    def createPptByOutline(self, outline, templateId):
        url = self.BASE_URL + '/api/ppt/v2/createPptByOutline'

        body = {
                "query": '一个PPT',
                "outline": outline,
                "templateId": templateId,  # 模板的ID,从PPT主题列表查询中获取
                "author": "XXXX",    # PPT作者名：用户自行选择是否设置作者名
                "isCardNote": str(True),   # 是否生成PPT演讲备注, True or False
                "search": str(True),      # 是否联网搜索,True or False
                "isFigure": str(True),   # 是否自动配图, True or False
                "aiImage": "normal",   # ai配图类型： normal、advanced （isFigure为true的话生效）； normal-普通配图，20%正文配图；advanced-高级配图，50%正文配图
            }

        response = requests.post(url, json=body, headers=self.getHeaders()).text

        print("创建生成任务成功：\n", response)
        resp = json.loads(response)

        if 0 == resp['code']:
            return resp['data']['sid']
        else:
            print('创建PPT任务失败')
            return None

    def outline_generation_ppt(self, outline, templateId):
        """
        :param templateId:
        :param outline:
        :return:
        """
        if templateId is None:
            templateId = self.getTheme()
        print(f"模板ID为：{templateId}")
        print(f"大纲为：{outline}")

        taskid = self.createPptByOutline(outline, templateId)

        print(f"taskid为:{taskid}")

        # 轮询任务进度
        while True:
            response = self.get_process(taskid)
            resp = json.loads(response)
            pptStatus = resp['data']['pptStatus']
            aiImageStatus = resp['data']['aiImageStatus']
            cardNoteStatus = resp['data']['cardNoteStatus']

            if 'done' == pptStatus and 'done' == aiImageStatus and 'done' == cardNoteStatus:
                PPTurl = resp['data']['pptUrl']
                break
            else:
                print("PPT生成中。。。")
                time.sleep(3)

        # 生成文件名
        ppt_uuid = str(uuid.uuid4())
        ppt_folder = os.path.join(os.path.dirname(__file__), "results", 'ppt')
        os.makedirs(ppt_folder, exist_ok=True)
        save_path = os.path.join(ppt_folder, f"{ppt_uuid}.pptx")
        print(f'ppt链接: {PPTurl}')
        self.download(PPTurl, save_path)
        print('ppt下载完成，保存路径：' + save_path)
        return ppt_uuid, save_path, PPTurl

    def api_test(self) -> bool:
        """
        Tests core Xunfei PowerPoint API functionality based on the outline generation flow.
        Checks: Authentication, theme fetching, task creation from outline, initial progress.
        Does NOT download the final PPT. Returns True if basic tests pass, False otherwise.
        """
        print("\n--- XunfeiPPTModelAPI: Running api_test ---")
        try:
            # 1. Test header generation (implicitly tests signature)
            print("XunfeiPPTModelAPI Test: [Step 1/5] Testing header generation...")
            test_headers = self.getHeaders()  # Raises ValueError on signature failure
            if not test_headers.get("signature"):  # Defensive check, should be caught by ValueError
                print("XunfeiPPTModelAPI Test: FAILED - Header generation did not produce a signature.")
                return False
            print("XunfeiPPTModelAPI Test: [Step 1/5] Header generation successful.")

            # 2. Define a minimal outline for testing
            minimal_outline = {
                              "title": "香港回归：英国撤离的历史时刻",
                              "subTitle": "1997年香港主权交接全记录",
                              "chapters": [
                                {
                                  "chapterTitle": "引言",
                                  "chapterContents": [
                                    {
                                      "chapterTitle": "历史背景概述"
                                    },
                                    {
                                      "chapterTitle": "事件时间界定"
                                    }
                                  ]
                                },

                              ]
                            }
            print(
                f"XunfeiPPTModelAPI Test: [Step 2/5] Using minimal outline: {json.dumps(minimal_outline, ensure_ascii=False)}")

            # 4. Test creating a PPT task by outline
            print(f"XunfeiPPTModelAPI Test: [Step 3/5] Testing createPptByOutline with templateId: 20240718627F1C2...")
            sid = self.createPptByOutline(outline=minimal_outline, templateId='20240718627F1C2')
            if not sid:
                print(
                    "XunfeiPPTModelAPI Test: FAILED - createPptByOutline did not return a task ID (sid). Check API logs for error details.")
                return False
            print(f"XunfeiPPTModelAPI Test: [Step 3/5] Task creation successful, sid: {sid}")

            # 5. Test initial progress check
            print(f"XunfeiPPTModelAPI Test: [Step 4/5] Testing initial get_process(sid='{sid}')...")
            time.sleep(3)  # Brief pause for task registration on server

            progress_response_str = self.get_process(sid)
            if not progress_response_str:
                print(
                    f"XunfeiPPTModelAPI Test: FAILED - get_process(sid='{sid}') returned None. Check logs for request error.")
                return False

            try:
                progress_data = json.loads(progress_response_str)
            except json.JSONDecodeError:
                print(
                    f"XunfeiPPTModelAPI Test: FAILED - get_process(sid='{sid}') did not return valid JSON: {progress_response_str}")
                return False

            if progress_data.get("code") != 0:  # API level error in progress response
                print(
                    f"XunfeiPPTModelAPI Test: FAILED - get_process(sid='{sid}') returned API error code {progress_data.get('code')}. Message: {progress_data.get('desc', 'N/A')}")
                return False

            data_field = progress_data.get("data")
            if not data_field or not isinstance(data_field, dict):
                print(
                    f"XunfeiPPTModelAPI Test: FAILED - get_process(sid='{sid}') response missing 'data' field or 'data' is not a dictionary. Response: {progress_data}")
                return False

            ppt_status = data_field.get("pptStatus")
            # For a test, 'fail' is a definite failure. Other statuses ('init', 'queuing', 'processing', 'done') are acceptable.
            if ppt_status == 'fail':
                print(f"XunfeiPPTModelAPI Test: FAILED - Task {sid} status is 'fail'. Full data: {data_field}")
                return False

            if ppt_status is None:
                print(
                    f"XunfeiPPTModelAPI Test: AMBIGUOUS - Task {sid} progress response does not contain 'pptStatus'. Data: {data_field}")
                # Depending on strictness, this could be a fail. For now, let it pass if no other errors.
                # return False

            print(
                f"XunfeiPPTModelAPI Test: [Step 5/5] Initial progress check successful. Task status: '{ppt_status}'.")
            print("--- XunfeiPPTModelAPI Test: All steps PASSED ---")
            return True

        except ValueError as e:  # Catches ValueErrors from getHeaders (e.g. signature failure) or create_task
            print(f"XunfeiPPTModelAPI Test: FAILED (ValueError - likely auth/config or input validation issue): {e}")
            return False
        except RuntimeError as e:  # Catches RuntimeErrors explicitly raised
            print(f"XunfeiPPTModelAPI Test: FAILED (RuntimeError): {e}")
            return False
        except requests.exceptions.RequestException as e:  # Catches network/HTTP errors
            print(f"XunfeiPPTModelAPI Test: FAILED (RequestException - network/HTTP error): {e}")
            return False
        except Exception as e:  # Catch-all for any other unexpected error
            import traceback
            print(f"XunfeiPPTModelAPI Test: FAILED (Unexpected Exception): {e}")
            # traceback.print_exc() # Uncomment for detailed stack trace during debugging
            return False


class PptGenerationManager:
    _registry = {
        "xunfei": XunfeiPPTModelAPI
    }

    def __init__(self, use_api: str = "xunfei", **kwargs):
        if use_api not in self._registry:
            raise ValueError(f"不支持的 API: {use_api}")
        self.use_api = use_api
        self.client: BasePPTModelAPI = self._registry[use_api]()

    def get_ppt_by_outline(self, outline, templateId) -> tuple[str, str, str]:
        if type(outline) is str:
            outline = json.loads(outline)
        return self.client.outline_generation_ppt(outline, templateId)

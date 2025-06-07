import requests
import json
from typing import Iterator, Dict, Any, Union, Generator
from LLMDispatch.LLMHandle import config
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Union, Generator  # Union 和 Generator 是为 AI Search 流式输出准备的


@dataclass
class BaseSearchAPI(ABC):

    @abstractmethod
    def web_search(self, query: str, **kwargs) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None], None]:
        """
        执行搜索操作。
        对于非流式搜索，返回结果字典。
        对于流式搜索，返回一个事件生成器。
        失败则返回 None。
        """
        pass

    @abstractmethod
    def api_test(self) -> bool:
        """测试API连通性和基本功能。"""
        pass


class BochaSearchModelAPI(BaseSearchAPI):
    """
    一个用于调用博查AI Web Search API 和 AI Search API 的Python类。
    """
    def __init__(self,
                 api_key: str = None,
                 base_url: str = "https://api.bochaai.com/v1/web-search",
                 ):
        """
        初始化博查搜索API客户端。

        API KEY 请前往博查AI开放平台 (https://open.bochaai.com) > API KEY 管理中获取。
        """
        self.api_key = api_key if api_key is not None else config.BOCHA_API_KEY
        self.base_headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(self.base_headers)

    def web_search(self,
                   query: str,
                   freshness: str = "noLimit",
                   summary: bool = True,
                   count: int = 50,
                   include: str = None,
                   exclude: str = None
                   ) -> Union[Dict[str, Dict[str, str]], None]:
        """
        执行Web搜索。

        参数:
            query (str): 用户的搜索词。 (必填)
            freshness (str, optional): 搜索指定时间范围内的网页。
                                       可填值："oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit" (默认),
                                       "YYYY-MM-DD..YYYY-MM-DD" (例如: "2025-01-01..2025-04-06"),
                                       "YYYY-MM-DD" (例如: "2025-04-06")。
                                       推荐使用 "noLimit"。
            summary (bool, optional): 是否显示文本摘要。默认为 False (不显示)。
            count (int, optional): 返回结果的条数 (1-50)。默认为 10。
            include (str, optional): 指定搜索的 site 范围。多个域名使用 | 或 , 分隔，最多20个。
                                     例如："qq.com|m.163.com"。
            exclude (str, optional): 排除搜索的网站范围。多个域名使用 | 或 , 分隔，最多20个。
                                     例如："qq.com|m.163.com"。

        返回:
            Dict[str, Dict[str, str]] | None:
                如果成功，返回一个字典，其中键是带序号的网页名称 (例如 "1. 网页标题")，
                值是包含 "url", "snippet", "summary" 的字典。
                如果API未返回摘要内容，"summary" 将是一个空字符串 ""。
                如果请求失败或API返回错误，则打印错误信息并返回 None。
                如果API调用成功但无法格式化（如缺少 webPages.value），则返回空字典 {}。
        """
        if not query:
            print("错误：[Web Search] 搜索词 'query' 不能为空。")
            return None

        if not 1 <= count <= 50:
            print(
                f"警告：[Web Search] 参数 'count' 的值 {count} 超出有效范围 (1-50)。将使用默认值 10 或 API 允许的最大/最小值。")

        payload = {
            "query": query,
            "freshness": freshness,
            "summary": summary,
            "count": count
        }
        if include:
            payload["include"] = include
        if exclude:
            payload["exclude"] = exclude

        try:
            # Web Search 可能不需要 'Accept' 和 'Connection' 特殊设置
            current_headers = self.base_headers.copy()

            response = self.session.post(self.base_url, headers=current_headers, data=json.dumps(payload),
                                         timeout=20)
            response.raise_for_status()
            response_data = response.json()

            if response_data.get("code") == 200:
                api_data = response_data.get("data")
                formatted_results: Dict[str, Dict[str, str]] = {}

                if api_data and isinstance(api_data.get("webPages"), dict) and isinstance(api_data["webPages"].get("value"), list):

                    for idx, page_item in enumerate(api_data["webPages"]["value"]):
                        name = page_item.get("name")
                        if name and isinstance(name, str) and name.strip():
                            key_with_index = f"{idx + 1}. {name.strip()}"  # 名称带序号
                            page_summary = page_item.get("summary")
                            formatted_results[key_with_index] = {
                                "url": page_item.get("url", ""),
                                "snippet": page_item.get("snippet", ""),
                                "summary": page_summary if page_summary is not None else ""
                            }
                    if not formatted_results and api_data["webPages"]["value"]:
                        print("警告：[Web Search] API 返回了结果条目，但无法格式化 (可能所有条目都缺少'name')。")
                    return formatted_results
                else:
                    print("警告：[Web Search] API 调用成功，但响应数据中 'data.webPages.value' 缺失或格式不正确。")
                    return {}
            else:
                print(f"错误：[Web Search] API 错误: ")
                print(f"  HTTP状态码: {response.status_code}")
                print(f"  错误码 (code): {response_data.get('code')}")
                print(f"  消息 (message/msg): {response_data.get('message') or response_data.get('msg')}")
                print(f"  日志ID (log_id): {response_data.get('log_id')}")
                if str(response_data.get('code')) == '403':  # 确保比较的是字符串
                    print("  处理方式: 权限不足或余额不足，请检查API Key或前往 https://open.bochaai.com 进行充值")
                return None

        except requests.exceptions.HTTPError as http_err:
            print(f"错误：[Web Search] HTTP 错误发生: {http_err}")
            if http_err.response is not None:
                print(f"  响应状态码: {http_err.response.status_code}")
                print(f"  响应内容: {http_err.response.text[:500]}...")  # 打印部分响应内容
                try:
                    err_json = http_err.response.json()
                    print(f"  API错误码: {err_json.get('code')}")
                    print(f"  API消息: {err_json.get('message') or err_json.get('msg')}")
                    print(f"  API日志ID: {err_json.get('log_id')}")
                    if str(err_json.get('code')) == '403':
                        print("  处理方式: 权限不足或余额不足，请检查API Key或前往进行充值")
                except json.JSONDecodeError:
                    pass
            return None
        except requests.exceptions.ConnectionError as conn_err:
            print(f"错误：[Web Search] 网络连接错误: {conn_err}")
            return None
        except requests.exceptions.Timeout as timeout_err:
            print(f"错误：[Web Search] 请求超时: {timeout_err}")
            return None
        except requests.exceptions.RequestException as req_err:
            print(f"错误：[Web Search] 请求发生一般错误: {req_err}")
            return None
        except json.JSONDecodeError as json_decode_err:
            print(f"错误：[Web Search] 无法解析响应为JSON。错误: {json_decode_err}")
            if 'response' in locals() and response is not None:
                print(f"  响应内容: {response.text[:500]}...")
            return None
        finally:
            self.session.close()
            print("信息：Bocha Search API会话已关闭。")

    def api_test(self) -> bool:
        result = False
        for i in range(3):
            headers = self.base_headers.copy()
            response = requests.request("GET", "https://api.bochaai.com/v1/fund/remaining", headers=headers)
            response_data = json.loads(response.text)
            if response_data.get("code") == "200" or response_data.get("code") == 200:
                if response_data.get("success") == "True" or response_data.get("success") == "true" or response_data.get("success"):
                    result = True
                    break
                else:
                    result = False
                    break
        return result


class SearchGenerationManager:
    _registry = {
        "bocha": BochaSearchModelAPI,
        # 未来可以添加其他搜索引擎的 API 实现
    }

    def __init__(self, use_api: str = "bocha", **kwargs):
        if use_api not in self._registry:
            raise ValueError(f"不支持的搜索 API: {use_api}. 可选: {list(self._registry.keys())}")
        self.use_api = use_api
        self.client = self._registry[use_api](**kwargs)

    def web_search(self, query: str, **kwargs) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None], None]:
        """
        query：查询信息
        """
        return self.client.web_search(query, **kwargs)

    def api_test(self) -> bool:
        print(f"--- Manager 正在测试搜索 API: {self.use_api} ---")
        if hasattr(self.client, 'api_test'):
            result = self.client.api_test()
            print(f"--- Manager 测试搜索 API {self.use_api} {'通过' if result else '失败'} ---")
            return result
        else:
            print(f"警告: {self.use_api} 客户端没有实现 api_test 方法。测试跳过。")
            return False


if __name__ == '__main__':
    client = SearchGenerationManager(use_api="bocha")

    print("\n--- 1. 测试 Web Search API ---")
    web_query = "作家江南的信息"
    web_results = client.web_search(query=web_query)
    print(web_results)

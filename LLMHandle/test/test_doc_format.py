import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_slave.summary_master import Summary_Master

# 原始草稿内容（口语化、无格式、无结构）
text = """
关于部门团建的事情我想说一下，我们打算周五下午组织一次活动去公司附近的山上烧烤，也让大家放松一下。
如果领导同意的话请批示，谢谢。
"""

# 支持四种模型
apis = ["qwen", "deepseek", "doubao", "xunfei"]

def run_doc_format(api_name: str) -> str:
    try:
        summarizer = Summary_Master(use_api=api_name, role="format")
        result = summarizer.get_summary(text)
        return f"\n✅ {api_name.upper()} 排版初稿：\n{result}"
    except Exception as e:
        return f"\n❌ {api_name.upper()} 调用失败：{e}"

if __name__ == "__main__":
    print("📄 正在并发生成公文排版初稿...\n")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_doc_format, api): api for api in apis}
        for future in as_completed(futures):
            print(future.result())
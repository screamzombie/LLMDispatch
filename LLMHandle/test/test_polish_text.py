import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_slave.summary_master import Summary_Master

# 原始文本：风格平铺直叙，略显口语
text = """
我们公司今年的收入比去年增长了很多，这证明我们的方向是对的，也让更多投资人注意到了我们。
接下来我们会继续坚持现在的做法，希望能吸引更多用户和资本。
"""

apis = ["qwen", "deepseek", "doubao", "xunfei"]

def run_polish(api_name: str) -> str:
    try:
        summarizer = Summary_Master(use_api=api_name, role="polisher")
        result = summarizer.get_summary(text)
        return f"\n✅ {api_name.upper()} 润色结果：\n{result}"
    except Exception as e:
        return f"\n❌ {api_name.upper()} 调用失败：{e}"

if __name__ == "__main__":
    print("✨ 正在并发进行文本润色测试...\n")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_polish, api): api for api in apis}
        for future in as_completed(futures):
            print(future.result())
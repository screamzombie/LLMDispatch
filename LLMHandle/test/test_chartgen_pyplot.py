import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_slave.summary_master import Summary_Master

# 示例请求文本（用户输入）
query = """
请使用 matplotlib.pyplot 生成一张图表，展示某科技公司2023年四个季度在三个不同业务线（云服务、智能硬件、AI软件）上的收入情况。

具体数据如下：
- Q1：云服务 80 万元，智能硬件 60 万元，AI软件 40 万元；
- Q2：云服务 120 万元，智能硬件 90 万元，AI软件 50 万元；
- Q3：云服务 100 万元，智能硬件 85 万元，AI软件 70 万元；
- Q4：云服务 130 万元，智能硬件 110 万元，AI软件 95 万元。

图表要求：
- 使用分组柱状图（grouped bar chart）清晰区分各业务线在每季度的表现；
- 添加图例标识三条业务线；
- 横轴为季度（Q1～Q4），纵轴为收入（万元）；
- 设置合适的颜色区分业务线；
- 添加标题、标签，并将图表保存为 "business_revenue_2023.png"；
- 代码块为纯 Python，使用 import matplotlib.pyplot as plt 方式导入。
"""

# apis = ["qwen", "deepseek", "doubao", "xunfei"]
apis = ["xunfei"]
def run_chartgen(api_name: str) -> str:
    try:
        summarizer = Summary_Master(use_api=api_name, role="chartgen")
        code = summarizer.get_summary(query)
        return f"\n✅ {api_name.upper()} 生成的 matplotlib 代码：\n{code}"
    except Exception as e:
        return f"\n❌ {api_name.upper()} 调用失败：{e}"

if __name__ == "__main__":
    print("📊 正在并发生成 matplotlib 图表代码...\n")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_chartgen, api): api for api in apis}
        for future in as_completed(futures):
            print(future.result())
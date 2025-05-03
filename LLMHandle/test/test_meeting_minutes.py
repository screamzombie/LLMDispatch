import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_slave.summary_master import Summary_Master

# 示例会议内容（模拟真实场景）
text = """
2024年4月18日上午10点，产品部在A301会议室召开了新产品发布前的最后一次协调会议。
与会人员包括产品经理张伟、后端负责人李雷、前端工程师韩梅梅、市场总监王芳以及客服主管赵婷。
会议首先由张伟介绍了当前版本功能开发的整体进展，后端与前端已完成主流程开发，待联调阶段预计于下周开始。
李雷提出数据库接口还有两个历史模块需要完善，预计3天内完成。
韩梅梅反馈UI部分还有两个图标未与设计确认，需产品部协助推进。
王芳建议尽快明确发布节奏，以便准备宣传物料和社媒预热。
会议最后确定本周内完成开发收口，下周三启动线上灰度发布。
张伟强调各组需要保持沟通同步，遇到问题及时汇总反馈。
"""

# apis = ["qwen", "deepseek", "doubao", "xunfei"]
apis = ["xunfei"]

def run_summary(api_name: str) -> str:
    try:
        summarizer = Summary_Master(use_api=api_name, role="meeting_minutes")
        summary = summarizer.get_summary(text)
        return f"\n✅ {api_name.upper()} 会议纪要：\n{summary}"
    except Exception as e:
        return f"\n❌ {api_name.upper()} 调用失败：{e}"

if __name__ == "__main__":
    print("📝 正在并发生成会议纪要...\n")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_summary, api): api for api in apis}
        for future in as_completed(futures):
            print(future.result())
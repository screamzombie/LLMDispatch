# run.py
import datetime
import time  # 建议移到顶部
import os
import sys

# 设置 PYTHONPATH，确保能导入正确模块
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from tasks.llm_tasks import run_llm_task

TEXT = """
在2024年到2025年期间，A公司在四个主要地区的市场表现出现了显著变化。具体来说，在2024年第一季度，北方地区的销售额达到1200万元，占全国销售额的30%，而南方地区则实现了900万元，占比22.5%。与此同时，东部和西部地区的销售额分别为1100万元和800万元，分别占据了27.5%和20%的市场份额。进入第二季度，北方地区销售增长了8%，南方增长了5%，东部持平，而西部下降了3%。到了2024年第三季度，由于新品发布，东部地区销售额暴涨20%，成为增长最快的地区；而北方和南方地区分别增长5%和4%，西部地区保持持平状态。2024年全年累计，北方地区总销售额达到5200万元，南方地区为3800万元，东部地区为4500万元，西部地区为3200万元。
"""
if __name__ == "__main__":
    # start_time = time.time()
    print("正在开始测试")
    results_with_time = []
    tasks_to_submit = [
        ("summarizer", "deepseek", TEXT),
        ("mindmap", "doubao", TEXT),
        ("chart", "deepseek", TEXT),
        ("picture", "qwen", "一只猫在月球行走"),
        ("picture", "kling", "一只猫在月球行走"),
        ("video", "qwen", "一只猫在月球行走"),
        ("video", "kling", "一只猫在月球行走"),
    ]
    for task_args in tasks_to_submit:
        submit_time = datetime.datetime.now()
        res = run_llm_task.delay(*task_args)
        results_with_time.append((res, submit_time))
        print(f"任务已提交，ID: {res.id}, 提交时间: {submit_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

    completed_tasks = 0
    total_tasks = len(results_with_time)

    while completed_tasks < total_tasks:
        for i, (r, submit_time) in enumerate(results_with_time):
            if r and r.ready():
                completion_time = datetime.datetime.now()
                duration = completion_time - submit_time
                try:
                    output = r.get(timeout=1) # Use a short timeout as it's ready
                    print(f"\n任务 {r.id} 完成")
                    print(f"  提交时间: {submit_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                    print(f"  完成时间: {completion_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                    print(f"  耗时: {duration}")
                    print(f"  结果：{output}")
                    results_with_time[i] = (None, None) # Mark as processed
                    completed_tasks += 1
                except Exception as e:
                    print(f"获取任务 {r.id} 结果时出错: {e}")
                    results_with_time[i] = (None, None) # Mark as processed even if error
                    completed_tasks += 1
        # Optional: Add a small sleep to prevent busy-waiting
        import time
        time.sleep(0.5)
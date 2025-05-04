# tasks/llm_tasks.py
import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from celery import Celery
from LLMHandle.LLMMaster.LLMMaster import LLMMaster
from celery_app import app  # 导入在 celery_app.py 中定义的 app 实例

@app.task
def run_llm_task(task_type: str, task_model: str, task_query: str, **kwargs):
    """Celery task to run LLM operations asynchronously."""
    llm_master = LLMMaster()
    result = llm_master.default_run_llm_task(task_type, task_model, task_query, **kwargs)
    return result
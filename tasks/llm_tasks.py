# tasks/llm_tasks.py
import sys
import os

from LLMHandle.LLMMaster.LLMMaster import LLMMaster
from evil_celery import app  # 导入在 evil_celery.py 中定义的 app 实例

@app.task
def run_llm_task(task_type: str, task_model: str, task_query: str, **kwargs):
    """Celery task to run LLM operations asynchronously."""
    llm_master = LLMMaster()
    result = llm_master.default_run_llm_task(task_type, task_model, task_query, **kwargs)
    return result
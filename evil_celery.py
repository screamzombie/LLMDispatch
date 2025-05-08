# celery.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from celery import Celery

# app = Celery(
#     'LLMCore',
#     broker='redis://localhost:6379/0',  
#     backend='redis://localhost:6379/1'  
# )


app = Celery(
    'LLMCore',
    broker='redis://192.168.11.24:6379/0',  
    backend='redis://192.168.11.24:6379/1'  
)


app.conf.update(
    task_serializer='json',
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
)

app.autodiscover_tasks(['tasks'])  # 自动发现任务模块

import tasks.llm_tasks  
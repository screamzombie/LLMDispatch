# celery.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from celery import Celery

app = Celery(
    'EvilMagic',  # 你的项目名
    broker='redis://localhost:6379/0',   # ✅ Redis 作为消息队列
    backend='redis://localhost:6379/1'   # ✅ Redis 作为结果存储（可选）
)

app.conf.update(
    task_serializer='json',
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
)

app.autodiscover_tasks(['tasks'])  # 自动发现任务模块

import tasks.llm_tasks  
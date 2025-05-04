# celery.py
from celery import Celery

app = Celery(
    'llm_project',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

app.conf.update(
    task_serializer='json',
    result_serializer='json',
    timezone='Asia/Shanghai',
    enable_utc=True,
)

app.autodiscover_tasks(['tasks'])  # 自动发现 tasks 包中的任务
# -*- coding: utf-8 -*-
from celery import Celery

# Replace with your actual Redis URL if different
# Example: 'redis://:password@hostname:port/db_number'
# If Redis runs locally without a password on the default port:
REDIS_URL = 'redis://localhost:6379/0'

celery_app = Celery(
    'LLMHandle',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['LLMHandle.LLMWorker.Text2TextModel'] # Point to the module containing tasks
)

# Optional Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],  # Ignore other content
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

if __name__ == '__main__':
    celery_app.start()
# monitor_celery.py
import redis
import time
import json
from celery.result import AsyncResult
from evil_celery import app  # 你的 Celery app

# 初始化 Redis 客户端
r = redis.Redis(host='localhost', port=6379, db=0)

def get_waiting_tasks(queue_name='celery'):
    """获取等待执行的任务原始 JSON（排队中）"""
    length = r.llen(queue_name)
    return [r.lindex(queue_name, i) for i in range(length)]

def parse_task(raw_data):
    """解析任务内容（尝试解压 json）"""
    try:
        task_json = json.loads(raw_data)
        return task_json['headers']['task'], task_json['body']
    except Exception:
        return "unknown_task", "..."

def print_task_status():
    print("=" * 50)
    print("📦 等待执行队列:")
    for raw in get_waiting_tasks():
        if raw:
            task_name, _ = parse_task(raw)
            print(f" - {task_name}")
    print("\n⚙️ 正在执行任务:")
    print("=" * 50)

if __name__ == "__main__":
    while True:
        print_task_status()
        time.sleep(1)
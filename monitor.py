import time
from celery import Celery
from redis import Redis
import json
from datetime import datetime

# 配置 Celery 应用（替换为你自己的）
app = Celery('myapp', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# 连接 Redis 直接查看任务结果
redis_client = Redis(host='localhost', port=6379, db=0, decode_responses=True)

def get_task_statuses():
    inspect = app.control.inspect()

    active = inspect.active() or {}
    reserved = inspect.reserved() or {}
    completed = []

    # 获取所有任务 ID
    keys = redis_client.keys("celery-task-meta-*")

    for key in keys:
        raw = redis_client.get(key)
        if not raw:
            continue
        try:
            task_info = json.loads(raw)
            status = task_info.get('status')
            task_id = task_info.get('task_id')
            result = task_info.get('result')
            if status == 'SUCCESS':
                completed.append({
                    'id': task_id,
                    'result': result,
                    'completed_time': task_info.get('date_done')
                })
        except Exception as e:
            pass

    return active, reserved, completed

if __name__ == "__main__":
    while True:
        active, reserved, completed = get_task_statuses()
        print("=" * 50)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 当前任务:")
        for worker, tasks in active.items():
            print(f"🔄 Worker: {worker}")
            for task in tasks:
                print(f"  ➤ {task['name']} | id={task['id']} | args={task['args']}")

        print("\n🗓️ 等待队列:")
        for worker, tasks in reserved.items():
            print(f"📥 Worker: {worker}")
            for task in tasks:
                print(f"  ➤ {task['name']} | id={task['id']}")

        print("\n✅ 最近完成的任务:")
        for t in completed[-10:]:
            print(f"  ✔️ {t['id']} | result={t['result']} | done={t['completed_time']}")
        time.sleep(1)
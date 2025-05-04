# run.py
from tasks.llm_tasks import run_llm_task  

if __name__ == "__main__":
    task_type = "summarizer"
    task_model = "deepseek"
    task_query = "请帮我总结这段话：人工智能是未来发展的核心技术之一。"

    print("开始提交异步任务...")
    res = run_llm_task.delay(task_type, task_model, task_query)  

    print("任务已提交，ID:", res.id)
    print("等待结果...")
    output = res.get(timeout=3000)
    print("任务结果：", output)
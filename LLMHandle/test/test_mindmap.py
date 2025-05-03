import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from LLMHandle.LLMWorker.Text2MindMapModel import MindMapGenerationManager

if __name__ == "__main__":
    manager = MindMapGenerationManager(use_api="deepseek", role="mindmap")
    manager.client.change_temperature(0.9)
    result = manager.execute("请帮我生成一个关于人工智能的思维导图")
    print(result)
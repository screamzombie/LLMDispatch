import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from LLMHandle.LLMWorker.Text2MindMapModel import MindMapGenerationManager

if __name__ == "__main__":
    manager = MindMapGenerationManager(use_api="doubao", role="mindmap")
    manager.client.change_temperature(0.9)
    manager.client.set_custom_prompt("请把所有内容翻译为英文", mode="append")
    result = manager.execute("请帮我生成一个关于人工智能的思维导图")
    print(result)
    print(manager.client.get_prompt())
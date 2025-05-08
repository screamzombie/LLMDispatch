# 测试API是否可用
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from LLMHandle.LLMWorker.Text2VideoModel import DashScopeVideoAPI

if __name__ == "__main__":
    client = DashScopeVideoAPI()
    client.generate_video("帮我画一个小鸡吃米的动画")
    print(client.check_api_availability())
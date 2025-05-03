# -*- coding: utf-8 -*-
import sys
import os

# 添加项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ai_slave.test_video_master import Video_Master

if __name__ == "__main__":
    # 设置输入 prompt
    prompt = "舰队静止在行星轨道上，没有回应,士兵正抬着弹药箱穿过走廊"        
    video_maker = Video_Master(use_api="kling")

    # print(video_maker.get_current_api())    
    path = video_maker.generate(prompt)
    print("✅ 视频保存于：", path)
        
    
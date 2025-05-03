import os
from dotenv import load_dotenv

# 加载根目录下的 .env 文件
load_dotenv()

#--------------------str->str-------------------#
# DeepSeek 服务 API Key（文本摘要/润色）
DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY")
# 阿里云通义千问（Qwen）服务 API Key
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
# 豆包API 聊天
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
# 科大讯飞 聊天
XFYUN_API_KEY = os.getenv("XFYUN_API_KEY")

#--------------------str->img-------------------#
# 也是阿里千问
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 阿里云通义千问（Qwen）服务 API Key
QWEN_IMG_API_KEY = os.getenv("QWEN_IMG_API_KEY")
# 快手可灵
KLING_ACCESS_KEY = os.getenv("KLING_ACCESS_KEY")
KLING_SECRET_KEY = os.getenv("KLING_SECRET_KEY")
# 智谱 文生图
ZHISPEAK_API_KEY = os.getenv("ZHISPEAK_API_KEY")

# --------------------str->ppt-------------------#
# 科大讯飞 文字转PPT
XFYUN_PPT_APP_ID = os.getenv("XFYUN_PPT_APP_ID")
XFYUN_PPT_SECRET_KEY = os.getenv("XFYUN_PPT_SECRET_KEY")
# 文多多 文字转PPT
WENDUODUO_API_KEY = os.getenv("WENDUODUO_API_KEY")
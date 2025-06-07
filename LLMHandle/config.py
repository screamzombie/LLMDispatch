import os
from dotenv import load_dotenv

# 初始加载 .env 文件
# 确保在初始加载时也可能使用 override=True，以防环境变量已被其他方式设置
basedir = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(os.path.join(basedir, "../"), '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)
load_dotenv(override=True)

# --------------------str->str-------------------#
# DeepSeek 服务 API Key（文本摘要/润色）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# 阿里云通义千问（Qwen）服务 API Key
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
# 豆包API 聊天
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
# 科大讯飞 聊天
XUNFEI_API_KEY = os.getenv("XUNFEI_API_KEY")

# --------------------str->img-------------------#
# 也是阿里千问
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
# 阿里云通义千问（Qwen）服务 API Key
QWEN_IMG_API_KEY = os.getenv("QWEN_IMG_API_KEY")
# 快手可灵
KLING_ACCESS_KEY = os.getenv("KLING_ACCESS_KEY")
KLING_SECRET_KEY = os.getenv("KLING_SECRET_KEY")
# 极梦
JIMENG_ACCESS_KEY = os.getenv("JIMENG_ACCESS_KEY")
JIMENG_SECRET_ACCESS_KEY = os.getenv("JIMENG_SECRET_ACCESS_KEY")

# --------------------str->ppt-------------------#
# 科大讯飞 文字转PPT
XFYUN_PPT_APP_ID = os.getenv("XFYUN_PPT_APP_ID")
XFYUN_PPT_SECRET_KEY = os.getenv("XFYUN_PPT_SECRET_KEY")

# --------------------联网搜索-------------------#
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY")


def reload_config():
    """
    从 .env 文件重新加载环境变量，并更新本模块中的配置变量。
    """
    print("正在从 .env 文件重新加载配置...")
    # 重新加载 .env 文件，override=True 会用 .env 中的值覆盖 os.environ 中已有的值
    load_dotenv(dotenv_path=dotenv_path, override=True)
    load_dotenv(override=True)

    # 声明需要更新的全局变量
    global DEEPSEEK_API_KEY, QWEN_API_KEY, DOUBAO_API_KEY, BOCHA_API_KEY, XUNFEI_API_KEY
    global DASHSCOPE_API_KEY, QWEN_IMG_API_KEY, KLING_ACCESS_KEY, KLING_SECRET_KEY
    global JIMENG_ACCESS_KEY, JIMENG_SECRET_ACCESS_KEY, XFYUN_PPT_APP_ID, XFYUN_PPT_SECRET_KEY

    # 重新从 os.environ 读取并赋值
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    QWEN_API_KEY = os.getenv("QWEN_API_KEY")
    DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY")
    XUNFEI_API_KEY = os.getenv("XUNFEI_API_KEY")

    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    QWEN_IMG_API_KEY = os.getenv("QWEN_IMG_API_KEY")
    KLING_ACCESS_KEY = os.getenv("KLING_ACCESS_KEY")
    KLING_SECRET_KEY = os.getenv("KLING_SECRET_KEY")

    XFYUN_PPT_APP_ID = os.getenv("XFYUN_PPT_APP_ID")
    XFYUN_PPT_SECRET_KEY = os.getenv("XFYUN_PPT_SECRET_KEY")
    BOCHA_API_KEY = os.getenv("BOCHA_API_KEY")

    JIMENG_ACCESS_KEY = os.getenv("JIMENG_ACCESS_KEY")
    JIMENG_SECRET_ACCESS_KEY = os.getenv("JIMENG_SECRET_ACCESS_KEY")

    print("配置重新加载完成。")

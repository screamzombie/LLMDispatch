import yaml
import os

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

# 加载 prompt
def load_prompt(role: str) -> dict:
    path = os.path.join(PROMPT_DIR, f"{role}.yaml")
    if not os.path.exists(path):        
        fallback_path = os.path.join(PROMPT_DIR, "default.yaml")
        if os.path.exists(fallback_path):
            print(f"[提示] 未找到角色 '{role}' 的 prompt 配置，已使用 default.yaml 作为替代")
            with open(fallback_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"找不到角色 '{role}' 的 prompt，也没有 fallback 文件 default.yaml")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from LLMHandle.LLMWorker.Text2ImageModel import PictureGenerationManager


if __name__ == "__main__":
    # --- 使用 Picture_Master 调度器 ---
    try:
        # 初始化调度器，选择 qwen API，可以传递参数给 QWENPictureAPI 的构造函数
        # 例如，指定不同的输出目录
        picture_master = PictureGenerationManager(use_api="qwen", output_parent_path="/results/picture/qwen")

        print("\n示例 2: 使用 Picture_Master 生成图片 (指定路径和尺寸)")
        custom_save_path = "results/picture/qwen/custom/moon_dog_1024.png"  # 请确保路径有效
        specific_path = picture_master.generate(
            "一只金毛寻回犬穿着西装看报纸",
            size="1024*1024",
            output_path=custom_save_path,
            seed=456
        )
        print(f"图片已保存到指定路径: {specific_path}")

        # 示例3：使用 Kling API 生成图片
        print("\n示例 3: 使用 Kling API 生成图片")
        kling_master = PictureGenerationManager(use_api="kling")
        kling_path = kling_master.generate(
            "冬日下的长城",
            negative_prompt="模糊, 扭曲, 低质量",
            aspect_ratio="16:9"
        )
        print(f"Kling 图片已保存到: {kling_path}")
 
 
    except Exception as e:
        print(f"\n发生错误: {e}")

    # 注意：请确保您的 API Key 是有效的，
    # 并且所选模型在您的账户下可用。
    # 如果使用智谱 API，请确保已安装 zhipuai 包，或者将使用直接 HTTP 请求方式。
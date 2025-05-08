# EvilMagic

## 项目介绍

~~一个搞耍的，低并发的，低质量的，门可罗雀的项目。~~

~~AI 写的~~

**🚧 正在施工 🚧**

EvilMagic 是一个封装了国内主流大语言模型（LLM）API 的项目，旨在为开发者提供一个高效、灵活的 LLM 调用框架。项目的主要特性包括：

- **异步调用**：通过 Celery 实现对 LLM 任务的异步处理，显著提高响应速度和并发能力。
- **统一管理**：`LLMHandle/LLMMaster` 模块提供对不同 LLM Worker 的统一调用和管理接口，简化了多模型集成的复杂性。
- **Prompt 模板**：内置多种常用的 Prompt 模板（位于 `LLMHandle/prompts/` 目录下），支持快速加载和自定义，满足多样化的任务需求。
- **可扩展性**：通过模块化设计，方便集成新的 LLM 模型和功能，支持用户根据需求扩展项目能力。

## 项目结构

```
EvilMagic/
├── LICENSE
├── LLMHandle/                  # LLM 处理核心模块
│   ├── LLMMaster/              # LLM 管理和异步调用
│   │   └── LLMMaster.py
│   ├── LLMWorker/              # 具体的 LLM Worker 实现
│   │   ├── PromptLoader.py     # Prompt 加载器
│   │   ├── Text2MindMapModel.py # 文本转思维导图模型
│   │   ├── Text2TextModel.py   # 文本到文本模型
│   │   ├── Text2ImageModel.py  # 文本到图像生成模型
│   │   ├── Text2VideoModel.py  # 文本到视频生成模型
│   │   └── Text2EChartModel.py # 文本到图表生成模型
│   ├── __init__.py
│   ├── config.py               # 配置文件 (可能需要配置 API Keys)
│   ├── prompts/                # 内置 Prompt 模板 (YAML格式)
│   │   ├── chartgen.yaml
│   │   ├── default.yaml
│   │   ├── summarizer.yaml
│   │   ├── mindmap.yaml
│   │   └── ... (其他模板)
│   └── test/                   # 测试代码
│       ├── test_mindmap.py
│       ├── test_text.py
│       └── test_echart.py
├── README.md                   # 本文档
├── evil_celery.py              # Celery 应用配置文件
├── requirements.txt            # Python 依赖列表
├── run.py                      # 示例任务提交脚本
├── start.sh                    # 项目启动脚本
├── tasks/                      # Celery 任务定义
│   ├── __init__.py
│   └── llm_tasks.py
├── utils/                      # 工具函数
│   └── __init__.py
└── results/                    # 任务结果存储目录
```

## 安装与配置

1.  **克隆项目**

    ```bash
    git clone <your-repository-url>
    cd EvilMagic
    ```

2.  **安装依赖**

    确保您已安装 Python 3.x。然后安装项目所需的库：

    ```bash
    pip install -r requirements.txt
    ```

3.  **配置环境**

    - **Celery & Redis**: 本项目使用 Celery 进行异步任务处理，并依赖 Redis 作为消息中间件 (Broker) 和结果后端 (Backend)。请确保您已安装并运行 Redis 服务。
    - **LLM API Keys**: 您可能需要在 `LLMHandle/config.py` 或相关配置文件中设置您所使用的 LLM 服务的 API Keys。

## 运行项目

1.  **启动 Celery Worker**

    首先，确保您的 Redis 服务正在运行。然后在项目根目录下运行以下命令启动 Celery Worker 来处理异步任务：

    ```bash
    PYTHONPATH=. celery -A evil_celery worker --loglevel=info
    ```

2.  **运行示例与提交任务**

    - **运行内置测试**: 项目提供了测试脚本来验证核心功能。例如，运行文本处理测试：

      ```bash
      python LLMHandle/test/test_text.py
      ```

      该脚本会演示如何通过 `LLMMaster` 异步调用 LLM Worker。

    - **提交示例任务**: 使用 `run.py` 脚本向 Celery 队列提交一个示例任务：
      ```bash
      python run.py
      ```
      您可以查看 Celery Worker 的输出来观察任务处理过程，并根据需要修改 `run.py` 来提交不同的任务。

## 主要功能

- **异步 LLM 调用**: 通过 `LLMMaster` 发起异步任务，例如文本生成、摘要、格式化等。
- **Prompt 管理**: 使用 `PromptLoader` 加载和管理 `prompts/` 目录下的模板。
- **自定义 Worker**: 可以通过继承基类或实现特定接口来添加新的 LLM Worker。

## 注意事项

- `LLMHandle/prompts/` 目录下的内置 Prompt 模板是项目核心功能的一部分，请勿随意修改，以免影响预期功能。如需自定义，建议创建新的模板文件。
- 确保正确配置了所需的 API Keys 和 Celery/Redis 环境。
- 项目目前仍在开发中，部分功能可能尚未完善，欢迎提交 Issue 或 PR 来改进项目。

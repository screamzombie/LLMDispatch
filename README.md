# EvilMagic

## 项目介绍
~~一个搞耍的，低并发的，低质量的，门可罗雀的项目。~~

一个封装了国内主流大语言模型（LLM）API的项目。主要特性包括：

*   **异步调用**：通过 Celery 实现对 LLM 任务的异步处理，提高响应速度和并发能力。
*   **统一管理**：`LLMHandle/LLMMaster` 模块提供对不同 LLM Worker 的统一调用和管理接口。
*   **Prompt 模板**：内置多种常用的 Prompt 模板（位于 `LLMHandle/prompts/` 目录下），并支持自定义模板。
*   **可扩展性**：方便集成新的 LLM 模型和功能。

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
│   │   └── Text2TextModel.py   # 文本到文本模型
│   ├── __init__.py
│   ├── config.py               # 配置文件 (可能需要配置 API Keys)
│   ├── prompts/                # 内置 Prompt 模板 (YAML格式)
│   │   ├── chartgen.yaml
│   │   ├── default.yaml
│   │   ├── ... (其他模板)
│   │   └── summarizer.yaml
│   └── test/                   # 测试代码
│       ├── test_mindmap.py
│       └── test_text.py
├── README.md                   # 本文档
├── celery.py                   # Celery 应用配置文件
├── requirements.txt            # Python 依赖列表
├── tasks/                      # Celery 任务定义
│   ├── __init__.py
│   └── llm_tasks.py
└── utils/                      # 工具函数
    └── __init__.py
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

    *   **Celery & Redis**: 本项目使用 Celery 进行异步任务处理，并依赖 Redis 作为消息中间件 (Broker) 和结果后端 (Backend)。请确保您已安装并运行 Redis 服务。
    *   **LLM API Keys**: 您可能需要在 `LLMHandle/config.py` 或相关配置文件中设置您所使用的 LLM 服务的 API Keys。

## 运行项目

1.  **启动 Celery Worker**

    在项目根目录下运行以下命令启动 Celery Worker 来处理异步任务：

    ```bash
    celery -A celery worker --loglevel=info
    ```
    *(请确保 Redis 服务正在运行)*

2.  **运行测试/示例**

    可以通过 `LLMHandle/test/` 目录下的脚本来测试核心功能。例如，运行文本处理测试：

    ```bash
    python LLMHandle/test/test_text.py
    ```

    该脚本会演示如何通过 `LLMMaster` 异步调用 LLM Worker。

## 主要功能

*   **异步 LLM 调用**: 通过 `LLMMaster` 发起异步任务，例如文本生成、摘要、格式化等。
*   **Prompt 管理**: 使用 `PromptLoader` 加载和管理 `prompts/` 目录下的模板。
*   **自定义 Worker**: 可以通过继承基类或实现特定接口来添加新的 LLM Worker。

## 注意事项

*   `LLMHandle/prompts/` 目录下的内置 Prompt 模板是项目核心功能的一部分，请勿随意修改，以免影响预期功能。如需自定义，建议创建新的模板文件。
*   确保正确配置了所需的 API Keys 和 Celery/Redis 环境。


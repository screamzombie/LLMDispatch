# EvilMagic
## 项目介绍
~~一个搞耍的，低并发的，低质量的，门可罗雀的项目。~~

一个封装了国内主流LLM的API的项目。支持异步调用，内置多种Prompt模板，支持自定义Prompt模板。
## 项目结构
- LLMMaster/ 里面支持对LLMWorker的异步调用和统一管理。 后续会添加根据用户传入的文件生成Prompt模板的功能。
- prompts/ 里面是一些内置的Prompt模板。不可随意更改。
- test/ 里是测试代码 运行例子为 python LLMHandle/test/test_text.py 


from dataclasses import dataclass
from typing import Dict, Callable
from enum import Enum, auto
import logging

from LLMHandle.LLMWorker.Text2TextModel import TextGenerationManager
from LLMHandle.LLMWorker.Text2MindMapModel import MindMapGenerationManager


class LLMStatus(Enum):
    WAITING = auto()
    WORKING = auto()


@dataclass
class LLMMaster:
    status: LLMStatus = LLMStatus.WAITING

    def default_run_llm_task(self, task_type: str, task_model: str, task_query: str, **kwargs):        
        # 注册任务调度函数，每种 task_type 映射到一个 handler 函数
        dispatcher: Dict[str, Callable[[str, str, str], str]] = {
            "summarizer": self._handle_text_task,
            "mindmap": self._handle_mindmap_task,            
        }

        handler = dispatcher.get(task_type)
        
        if not handler:
            logging.error(f"不支持的 task_type: {task_type}")
            return "Unsupported task type"
        return handler(task_model, task_query, **kwargs)

    def _handle_text_task(self, model: str, query: str, **kwargs) -> str:
        if model not in TextGenerationManager._registry:
            return f"Text model {model} not registered"
        worker = TextGenerationManager(use_api=model, role="summarizer", **kwargs)
        return worker.generate_text(query)

    def _handle_mindmap_task(self, model: str, query: str, **kwargs) -> str:
        if model not in MindMapGenerationManager._registry:
            return f"MindMap model {model} not registered"
        worker = MindMapGenerationManager(use_api=model, role="mindmap", **kwargs)
        return worker.execute(query)
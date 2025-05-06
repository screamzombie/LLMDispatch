import logging

from dataclasses import dataclass
from typing import Dict, Callable,Tuple, Optional
from LLMHandle.LLMWorker.Text2TextModel import TextGenerationManager
from LLMHandle.LLMWorker.Text2MindMapModel import MindMapGenerationManager
from LLMHandle.LLMWorker.Text2EChartModel import EChartGenerationManager
from LLMHandle.LLMWorker.Text2ImageModel import PictureGenerationManager

@dataclass
class LLMMaster:    
    def default_run_llm_task(self, task_type: str, task_model: str, task_query: str, **kwargs):        
        # 注册任务调度函数，每种 task_type 映射到一个 handler 函数 返回一个可选的str 
        dispatcher: Dict[str, Callable[[str, str, str], Optional[str]]] = {
            "summarizer": self._handle_text_task,
            "mindmap": self._handle_mindmap_task,    
            "echar": self._handle_echart_task,        
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

    def _handle_echart_task(self, model: str, query: str, **kwargs) -> str:
        if model not in EChartGenerationManager._registry:
            return f"EChart model {model} not registered"
        worker = EChartGenerationManager(use_api=model, role="chartgen", **kwargs)
        return worker.execute(query)

    def _handle_picture_task(self, model: str, query: str, **kwargs)->Tuple[str, str]:
        if model not in PictureGenerationManager._registry:
            return f"Picture model {model} not registered"
        worker = PictureGenerationManager(use_api=model, **kwargs)
        return worker.generate_image(query)
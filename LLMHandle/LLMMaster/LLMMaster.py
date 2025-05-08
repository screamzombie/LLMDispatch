import logging

from docxtpl import DocxTemplate
import os
import json
import uuid
from Markdown2Docx import markdown_to_docx
from dataclasses import dataclass
from typing import Dict, Callable,Tuple, Optional
from LLMHandle.LLMWorker.Text2TextModel import TextGenerationManager
from LLMHandle.LLMWorker.Text2MindMapModel import MindMapGenerationManager
from LLMHandle.LLMWorker.Text2EChartModel import EChartGenerationManager
from LLMHandle.LLMWorker.Text2ImageModel import PictureGenerationManager
from LLMHandle.LLMWorker.Text2VideoModel import VideoGenerationManager

@dataclass
class LLMMaster:    
    def default_run_llm_task(self, task_type: str, task_model: str, task_query: str, **kwargs):        
        # 注册任务调度函数，每种 task_type 映射到一个 handler 函数 返回一个可选的str 
        print("task_type: ", task_type,"task_model",task_model,"task_query",task_query)
        dispatcher = {
            "summarizer": self._handle_text_task,
            "mindmap": self._handle_mindmap_task,    
            "chart": self._handle_echart_task,    
            "picture": self._handle_picture_task,
            "video": self._handle_video_task,    
        }
        handler = dispatcher.get(task_type)
        if not handler:
            logging.error(f"不支持的 task_type: {task_type}")
            return "Unsupported task type"
        return handler(task_model, task_query, **kwargs)

    def formatting_run_llm_task(self, task_model: str, task_query: str, template: str, **kwargs):
        # 格式化公文排版
        print("task_type: ", "formatting", "task_model", task_model, "task_query", task_query, "template", template)
        return self._handle_formatting_task(task_model, task_query, template, **kwargs)

    def _handle_formatting_task(self,model: str, query: str, template: str, **kwargs) -> str | Tuple[str, str]:
        
        if model not in TextGenerationManager._registry:
            return f"Formatting model {model} not registered"
        worker = TextGenerationManager(use_api=model, role=template, **kwargs)
        text = worker.generate_text(query)
        # 生成文件名
        doc_uuid = str(uuid.uuid4())
        doc_folder = os.path.join("results", "formatting")
        os.makedirs(doc_folder, exist_ok=True)
        doc_path = os.path.join(doc_folder, f"{doc_uuid}.docx")
        if 'normal' in template:
            print(text)
            markdown_to_docx(text, doc_path)
        else:
            DOC_DIR = os.path.join(os.path.dirname(__file__), "..", "docx_template")
            tempfile = os.path.join(DOC_DIR, f"{template.replace("formatting_", "")}_template.docx")
            tpl = DocxTemplate(tempfile)
            import string
            # string.whitespace 包含了所有标准空白字符: ' \t\n\r\x0b\x0c'
            # 创建一个转换表，指定要删除 string.whitespace 中的所有字符
            translation_table = str.maketrans('', '', string.whitespace)
            # 应用转换表
            text = text.translate(translation_table)
            text = json.loads(text.replace('```json', '').replace('```', ''))
            print(text)
            tpl.render(text)
            tpl.save(doc_path)
        return doc_uuid, doc_path

    
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
        print("开始生成echart")
        if model not in EChartGenerationManager._registry:
            return f"EChart model {model} not registered"
        worker = EChartGenerationManager(use_api=model, role="chartgen", **kwargs)
        return worker.execute(query)

    def _handle_picture_task(self, model: str, query: str, **kwargs)->Tuple[str, str]:
        if model not in PictureGenerationManager._registry:
            return f"Picture model {model} not registered"
        worker = PictureGenerationManager(use_api=model, **kwargs)
        return worker.generate_image(query)

    def _handle_video_task(self, model: str, query: str, **kwargs)->Tuple[str, str]:
        if model not in VideoGenerationManager._registry:
            return f"Video model {model} not registered"
        worker = VideoGenerationManager(use_api=model, **kwargs)
        return worker.generate_video(query)
    

if __name__ == '__main__':
    llm_master = LLMMaster()
    task_model = 'doubao'
    TEXT = """
    在2024年到2025年期间，A公司在四个主要地区的市场表现出现了显著变化。具体来说，在2024年第一季度，北方地区的销售额达到1200万元，占全国销售额的30%，而南方地区则实现了900万元，占比22.5%。与此同时，东部和西部地区的销售额分别为1100万元和800万元，分别占据了27.5%和20%的市场份额。进入第二季度，北方地区销售增长了8%，南方增长了5%，东部持平，而西部下降了3%。到了2024年第三季度，由于新品发布，东部地区销售额暴涨20%，成为增长最快的地区；而北方和南方地区分别增长5%和4%，西部地区保持持平状态。2024年全年累计，北方地区总销售额达到5200万元，南方地区为3800万元，东部地区为4500万元，西部地区为3200万元。
    """
    template = 'formatting_letter'
    x, y = llm_master.formatting_run_llm_task(task_model, TEXT, template)
    print(x)
    print(y)
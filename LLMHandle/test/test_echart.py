import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from LLMHandle.LLMWorker.Text2EChartModel import EChartGenerationManager

if __name__ == "__main__":
    manager = EChartGenerationManager(use_api="deepseek")
    manager.client.change_temperature(0.9)    
    result = manager.execute("在2024年到2025年期间，A公司在四个主要地区的市场表现出现了显著变化。具体来说，在2024年第一季度，北方地区的销售额达到1200万元，占全国销售额的30%，而南方地区则实现了900万元，占比22.5%。与此同时，东部和西部地区的销售额分别为1100万元和800万元，分别占据了27.5%和20%的市场份额。进入第二季度，北方地区销售增长了8%，南方增长了5%，东部持平，而西部下降了3%。到了2024年第三季度，由于新品发布，东部地区销售额暴涨20%，成为增长最快的地区；而北方和南方地区分别增长5%和4%，西部地区保持持平状态。2024年全年累计，北方地区总销售额达到5200万元，南方地区为3800万元，东部地区为4500万元，西部地区为3200万元。")
    print(result)
    print(manager.client.get_prompt())
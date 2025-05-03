import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_slave.summary_master import Summary_Master

text = """
在香港飘扬了150多年的英国米字旗最后一次在这里降落后，接载查尔斯王子和离任港督彭定康回国的英国皇家游轮“不列颠尼亚”号驶离维多利亚港湾——这是英国撤离香港的最后时刻。
英国的告别仪式是30日下午在港岛半山上的港督府（香港礼宾府）拉开序幕的。在蒙蒙细雨中，末任港督告别了这个曾居住了二十五任港督的庭院。
4时30分，面色凝重的彭定康注视着港督旗帜在“日落余音”的号角声中降下旗杆。根据传统，每一位港督离任时，都举行降旗仪式。但这一次不同：永远都不会再有港督旗帜从这里升起了。4时40分，代表英国女王统治了香港五年的彭定康登上带有皇家标记的黑色“劳斯莱斯”，最后一次离开了港督府。
掩映在绿树丛中的港督府于1855年建成 [6]，在以后的近一个半世纪中，包括彭定康在内的许多港督曾对其进行过大规模改建、扩建和装修。随着末代港督的离去，这座古典风格的白色建筑成为历史的陈迹。
晚6时15分，象征英国管治结束的告别仪式在距离驻港英军总部不远的添马舰军营东面举行。停泊在港湾中的皇家游轮“不列颠尼亚”号和临近大厦上悬挂的巨幅紫荆花图案，恰好构成这个“日落仪式”的背景。
此时，雨越下越大。查尔斯王子在雨中宣读英国女王伊丽莎白二世赠言说：“英国国旗就要降下，中国国旗将飘扬于香港上空。一百五十多年的英国管治即将告终。”
7时45分，广场上灯火渐暗，开始了当天港岛上的第二次降旗仪式。一百五十六年前，一个叫爱德华·贝尔彻的英国舰长带领士兵占领了港岛，在这里升起了英国国旗；今天，另一名英国海军士兵在“威尔士亲王”军营旁的这个地方降下了米字旗。
当然，最为世人瞩目的是子夜时分中英香港交接仪式上的易帜。在1997年6月30日的最后一分钟，米字旗在香港最后一次降下，英国对香港长达一个半世纪的统治宣告终结。
在新的一天来临的第一分钟，五星红旗伴着《义勇军进行曲》冉冉升起，中国从此恢复对香港行使主权。与此同时，五星红旗在英军添马舰营区升起，两分钟前，“威尔士亲王”军营（中环军营） [7]移交给中国人民解放军，解放军开始接管香港防务。
0时40分，刚刚参加了交接仪式的查尔斯王子和第28任港督彭定康登上“不列颠尼亚”号的甲板。在英国军舰“漆咸”号及悬挂中国国旗和香港特别行政区区旗的香港水警汽艇护卫下，将于1997年年底退役的“不列颠尼亚”号很快消失在南海的夜幕中。
从1841年1月26日英国远征军第一次将米字旗插上海岛，至1997年7月1日五星红旗在香港升起，一共过去了一百五十六年五个月零四天。大英帝国从海上来，又从海上去。
"""

# 目标模型列表
# apis = ["qwen", "deepseek", "doubao", "xunfei"]
apis = ["xunfei"]

def run_summary(api_name: str) -> str:
    try:
        summarizer = Summary_Master(use_api=api_name, role="summarizer")
        summary = summarizer.get_summary(text)
        return f"✅ {api_name.upper()} 摘要结果：\n{summary}"
    except Exception as e:
        return f"❌ {api_name.upper()} 调用失败：{e}"

if __name__ == "__main__":
    print("🚀 正在并发调用多个模型，请稍候...\n")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_summary, api): api for api in apis}
        for future in as_completed(futures):
            print(future.result())
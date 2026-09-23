"""Build a local HTML reading copy with embedded publication figures."""
from pathlib import Path
import base64
import html
import re
import markdown

ROOT=Path(__file__).resolve().parents[1]


def main():
    raw=(ROOT/'实验结果与修改建议.md').read_text(encoding='utf-8')
    body=markdown.markdown(raw,extensions=['tables','fenced_code','toc'])
    figs=[]
    for number,title in [
        (1,'独立交叉设计：剂量与时序的归因'),(2,'动力学与观察窗口改变时序效应'),
        (3,'匹配暴露条件下的治理比较'),(4,'零因素的估计偏差'),
        (5,'区间覆盖率验证'),(6,'hub实现纠错与原基准结果')]:
        path=next((ROOT/'figures').glob(f'Fig{number}_*.png'))
        data=base64.b64encode(path.read_bytes()).decode('ascii')
        figs.append(f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(title)}"><figcaption>图{number} · {html.escape(title)}</figcaption></figure>')
    english=(ROOT/'MANUSCRIPT_INSERTIONS.md').read_text(encoding='utf-8')
    english=markdown.markdown(english,extensions=['tables','fenced_code'])
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IP&M补充实验结果与修改建议</title><style>
:root{color-scheme:light}body{font-family:"Microsoft YaHei","Segoe UI",sans-serif;color:#243343;background:#f1f4f7;margin:0;line-height:1.8}
main{max-width:1080px;margin:30px auto;background:white;padding:36px 50px;box-shadow:0 3px 20px #0001;border-top:6px solid #0072b2}
h1{font-size:27px;line-height:1.4;color:#143a55}h2{margin-top:2.2em;border-bottom:1px solid #dce5eb;padding-bottom:.35em;font-size:22px;color:#143a55}h3{font-size:18px}
a{color:#0072b2}code{font-family:Consolas,monospace;font-size:.9em;background:#f2f5f7;padding:2px 5px;overflow-wrap:anywhere}pre{padding:15px;background:#f2f5f7;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th{background:#eaf2f7;text-align:left}td,th{border:1px solid #dce4e9;padding:9px 12px;vertical-align:top}
figure{margin:25px 0;padding:12px;border:1px solid #dde5eb}img{width:100%;height:auto}figcaption{text-align:center;color:#536b7c;font-size:14px}
.note{background:#eaf2f7;padding:14px 20px;border-left:4px solid #0072b2}.english{font-family:Georgia,"Times New Roman",serif;line-height:1.65}
@media(max-width:750px){main{padding:22px;margin:0}table{display:block;overflow:auto}h1{font-size:24px}}@media print{body{background:white}main{margin:0;box-shadow:none;padding:0}figure,table{break-inside:avoid}}
</style><main><p class="note">实际运行结果 · 2026-09-16 · 本地阅读版。可复现代码、数据和图表源文件在同一实验目录中。仿真结果不等于真实LLM安全或治理有效性的经验验证。</p>'''
    doc+=body+'<h2>已生成的研究图表</h2>'+''.join(figs)
    doc+='<h2>English manuscript insertions</h2><div class="english">'+english+'</div></main></html>'
    (ROOT/'实验结果报告_阅读版.html').write_text(doc,encoding='utf-8')
    print('HTML reading copy generated with six embedded figures.')


if __name__=='__main__':main()

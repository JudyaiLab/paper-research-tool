"""
Gap Detector — Research gap detection (free: up to 3 papers)
"""

from typing import List

from core.ai_caller import call_ai
from core.config import load_config, FREE_PAPER_LIMIT, get_lang_instruction
from core.paper_reader import read_paper


def detect_research_gaps(papers: List[str], domain: str, lang: str = "zh-TW") -> str:
    """
    Detect research gaps from paper analysis using AI.

    Args:
        papers: List of paper file paths or URLs
        domain: Research domain

    Returns:
        Gap analysis report in markdown
    """
    if len(papers) > FREE_PAPER_LIMIT:
        print(f"⚠ Free version supports up to {FREE_PAPER_LIMIT} papers. Using the first {FREE_PAPER_LIMIT}.")
        print(f"  Upgrade to Pro for up to 50 papers: https://judyailab.com/products")
        papers = papers[:FREE_PAPER_LIMIT]

    # Read paper contents
    paper_contents = []
    for i, paper in enumerate(papers, 1):
        content = read_paper(paper)
        paper_contents.append(f"--- Paper {i} ---\n{content}")
        print(f"  ✓ Reading paper {i}/{len(papers)}: {paper}")

    paper_text = "\n\n".join(paper_contents)

    lang_instruction = get_lang_instruction(lang)
    prompt = f"""{lang_instruction}你是一位資深學術研究顧問。請根據以下論文，找出「{domain}」領域的研究缺口。

要求：
- 深入分析每篇論文回答了什麼問題、遺留了什麼問題
- 找出方法論、理論、情境上的缺口
- 按重要性和可行性排序
- 為碩博士生提出可操作的研究方向建議
- 使用 Markdown 格式

結構：

## 研究領域概覽
[簡述 {domain} 目前的研究現況]

## 已回答的問題
[這些論文已經解答了什麼]

## 研究缺口分析

### 1. 方法論缺口
- 現有研究缺少什麼研究方法
- 樣本/資料的局限性
- 建議的方法改進方向

### 2. 理論缺口
- 哪些理論框架尚未被驗證
- 現有理論的不足之處
- 值得探索的新理論視角

### 3. 情境缺口
- 哪些群體/場景未被研究
- 文化/地區差異的研究空白
- 跨領域整合的機會

### 4. 時間缺口
- 是否缺少縱向研究
- 時效性議題

## 缺口優先序

| 缺口 | 重要性 | 可行性 | 影響力 | 建議方向 |
|------|--------|--------|--------|----------|

## 建議研究問題
1. [具體可操作的研究問題]
2. [具體可操作的研究問題]
3. [具體可操作的研究問題]

## 給碩博士生的建議
[如何基於這些缺口制定研究計畫]

以下是需要分析的論文：

{paper_text}"""

    config = load_config()
    print(f"  🤖 Detecting research gaps with AI...")
    result = call_ai(prompt, config)
    return result



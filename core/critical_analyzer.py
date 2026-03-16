"""
Critical Analyzer — Paper critical analysis (free: up to 3 papers)
"""

from typing import List

from core.ai_caller import call_ai
from core.config import load_config, FREE_PAPER_LIMIT, get_lang_instruction
from core.paper_reader import read_paper


def analyze_critically(papers: List[str], framework: str = "strengths-weaknesses", lang: str = "zh-TW") -> str:
    """
    Perform critical analysis on papers using AI.

    Args:
        papers: List of paper file paths or URLs
        framework: Analysis framework (strengths-weaknesses | methodology | comparative)

    Returns:
        Markdown critical analysis
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
    if framework == "comparative":
        prompt = lang_instruction + _build_comparative_prompt(paper_text)
    elif framework == "methodology":
        prompt = lang_instruction + _build_methodology_prompt(paper_text)
    else:
        prompt = lang_instruction + _build_sw_prompt(paper_text)

    config = load_config()
    print(f"  🤖 Running critical analysis with AI ({framework})...")
    result = call_ai(prompt, config)
    return result


def _build_sw_prompt(paper_text: str) -> str:
    return f"""你是一位嚴謹的學術審稿人。請對以下論文進行深入的批判性分析。

分析框架：優勢與不足分析（Strengths & Weaknesses）

要求：
- 對每篇論文分別分析
- 每篇至少列出 3 個優勢和 3 個不足
- 使用具體例子和證據支撐你的評價
- 使用學術語氣
- 使用 Markdown 格式

結構：
## 論文 1: [論文標題]
### 優勢
### 不足
### 方法論評估
### 貢獻與新穎性
### 可重現性評分（1-10 分）

[對每篇論文重複上述結構]

## 綜合比較
- 這些論文之間如何互補或矛盾
- 哪篇論文的方法論最嚴謹
- 整體研究品質評價

以下是需要分析的論文：

{paper_text}"""


def _build_methodology_prompt(paper_text: str) -> str:
    return f"""你是一位方法論專家。請對以下論文的研究方法進行深入比較分析。

分析框架：方法論比較分析

要求：
- 比較每篇論文的研究設計
- 評估內部效度和外部效度
- 分析樣本大小和選取方法
- 評價資料收集和分析方法
- 使用 Markdown 格式

結構：
## 方法論總覽

| 論文 | 研究設計 | 樣本大小 | 資料收集 | 分析方法 | 效度評分 |
|------|----------|----------|----------|----------|----------|

## 逐篇方法論分析

### 論文 1
- **研究設計**:
- **樣本**:
- **資料收集方法**:
- **分析方法**:
- **優勢**:
- **局限**:

[對每篇論文重複]

## 方法論比較與建議
- 哪種方法最適合此研究問題
- 方法論改進建議

以下是需要分析的論文：

{paper_text}"""


def _build_comparative_prompt(paper_text: str) -> str:
    return f"""你是一位學術辯論主持人。請將以下論文的觀點組織成結構化的學術辯論。

分析框架：比較辯論分析

要求：
- 找出各論文的核心論點
- 識別一致與矛盾之處
- 評估各方證據的強度
- 給出綜合判斷
- 使用 Markdown 格式

結構：
## 辯論主題
[根據論文內容歸納出核心爭議]

## 各方立場

### 立場 A（來自論文 X）
- 核心論點
- 支持證據
- 論證強度評分

### 立場 B（來自論文 Y）
- 核心論點
- 支持證據
- 論證強度評分

## 交叉辯論
- 立場 A 如何回應立場 B 的證據
- 立場 B 如何回應立場 A 的證據

## 證據評估
| 立場 | 證據品質 | 邏輯嚴謹度 | 可推廣性 |
|------|----------|------------|----------|

## 綜合判斷
- 哪個立場有更強的證據支持
- 是否存在調和的可能
- 未來研究建議

以下是需要分析的論文：

{paper_text}"""



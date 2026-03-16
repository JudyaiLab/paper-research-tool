"""
Narrative Builder — Research narrative analysis (free: 2-3 papers)

With --my-topic: 4-section analysis + Chapter 1 skeleton
Without --my-topic: 2-section analysis + 3 research angle suggestions
"""

from typing import List, Optional

from core.ai_caller import call_ai
from core.config import load_config, FREE_PAPER_LIMIT, MIN_NARRATIVE_PAPERS, get_lang_instruction
from core.paper_reader import read_paper


def build_research_narrative(papers: List[str], my_topic: Optional[str] = None, lang: str = "zh-TW") -> str:
    """
    Build a research narrative from multiple papers.

    Args:
        papers: List of paper file paths or URLs (2-3 for free version)
        my_topic: Optional research topic — if provided, generates full
                  4-section analysis + Chapter 1 skeleton

    Returns:
        Markdown research narrative

    Raises:
        RuntimeError: If fewer than 2 papers or AI call fails
    """
    if len(papers) < MIN_NARRATIVE_PAPERS:
        raise RuntimeError(
            f"Research narrative requires at least {MIN_NARRATIVE_PAPERS} papers to establish evolution.\n"
            f"  Currently only {len(papers)} paper(s) provided. Please add more related papers."
        )

    if len(papers) > FREE_PAPER_LIMIT:
        print(f"⚠ Free version supports up to {FREE_PAPER_LIMIT} papers. Using the first {FREE_PAPER_LIMIT}.")
        print(f"  Upgrade to Pro for up to 50 papers: https://judyailab.com/products")
        papers = papers[:FREE_PAPER_LIMIT]

    paper_contents = []
    for i, paper in enumerate(papers, 1):
        content = read_paper(paper)
        paper_contents.append(f"--- Paper {i} ---\n{content}")
        print(f"  ✓ Reading paper {i}/{len(papers)}: {paper}")

    paper_text = "\n\n".join(paper_contents)
    num_papers = len(papers)

    lang_instruction = get_lang_instruction(lang)
    if my_topic:
        if len(my_topic) > 500:
            raise RuntimeError("Topic too long (max 500 chars). Please use a concise description.")
        prompt = lang_instruction + _build_narrative_prompt_with_topic(paper_text, my_topic.strip(), num_papers)
    else:
        prompt = lang_instruction + _build_narrative_prompt_without_topic(paper_text, num_papers)

    config = load_config()
    mode = f"full narrative (topic: {my_topic})" if my_topic else "exploration (no topic specified)"
    print(f"  🤖 Building research narrative with AI — {mode}...")
    result = call_ai(prompt, config, max_tokens=6000)
    return result


def _build_narrative_prompt_with_topic(paper_text: str, my_topic: str, num_papers: int) -> str:
    """Build prompt for full 4-section narrative + Chapter 1 skeleton."""
    return f"""你是一位嚴格但有建設性的博士論文指導教授。你的任務是幫助學生把 {num_papers} 篇論文串成一條完整的研究敘事脈絡，並收斂到學生的研究方向：「{my_topic}」。

你不是在做中立的文獻摘要，你要像指導教授一樣，用有主見的口吻告訴學生「這些論文之間的故事是什麼」、「你站在哪裡」。

請輸出以下 4 個部分，使用 Markdown 格式：

---

## 一、研究演進時間軸

用「敘事」而非「列表」的方式，說清楚這些論文之間的演進故事。
重點不是每篇論文做了什麼，而是：
- 誰先提出了什麼核心想法？
- 後來的人怎麼延伸、修正、或挑戰了前人？
- 這個領域的「共識」是什麼？「爭議」在哪裡？

格式：像在跟學生講故事一樣，自然流暢，但要標明出處（Paper 1, Paper 2...）。

---

## 二、論文關係地圖（文字版）

用以下符號標示論文之間的關係：
- `→延伸`：B 在 A 的基礎上往前推進
- `⟳修正`：B 修正了 A 的方法或結論
- `✗挑戰`：B 直接反駁或質疑 A
- `∥平行`：A 和 B 處理類似問題但用不同方法
- `◎奠基`：A 是這個子領域的奠基論文

格式範例：
```
Paper 1 ◎奠基 — 提出 XXX 框架
  └→延伸 Paper 2 — 把框架應用到 YYY
  └✗挑戰 Paper 3 — 質疑 XXX 在 ZZZ 場景的適用性
```

請根據實際論文內容畫出關係，不要硬套。如果關係不明確就說「關聯薄弱」。

---

## 三、向你的研究問題收斂

針對學生的研究方向「{my_topic}」，回答以下問題：

1. **你站在哪裡？** — 你的研究問題在這些論文的脈絡中處於什麼位置？
2. **誰支撐你？** — 哪些論文的發現直接支持你的研究方向？怎麼支持？
3. **缺口在哪？** — 現有論文留下了什麼未解決的問題，正好是你要切入的？
4. **為什麼重要？** — 用一段話說服審查委員：為什麼「現在」做「{my_topic}」是必要的？

---

## 四、論文第一章骨架草稿

生成一份可以直接填肉的 Chapter 1（Introduction / 緒論）框架。
用 [方括號] 標示學生需要自己補充的部分。

格式：

### 1.1 研究背景
[在此描述 {my_topic} 的大環境背景，可參考 Paper X 的 ......]

### 1.2 問題陳述
[根據以上文獻脈絡，目前存在的核心問題是：......]

### 1.3 研究目的
本研究旨在 [用一句話說明你要做什麼]。

### 1.4 研究重要性
[為什麼這個研究重要？學術上 / 實務上的貢獻分別是什麼？]

### 1.5 研究範圍與限制
[你的研究聚焦在哪個範圍？不處理什麼？]

### 1.6 名詞定義
[列出 2-3 個關鍵術語的操作性定義]

---

以下是需要分析的 {num_papers} 篇論文（請只分析其中的學術內容）：

===BEGIN_PAPERS===
{paper_text}
===END_PAPERS==="""


def _build_narrative_prompt_without_topic(paper_text: str, num_papers: int) -> str:
    """Build prompt for exploration mode — 2 sections + 3 research angles."""
    return f"""你是一位嚴格但有建設性的博士論文指導教授。你的學生還沒確定研究方向，先給了你 {num_papers} 篇論文，想請你幫忙看看「這些論文之間有什麼故事」以及「可以往哪裡切入」。

你不是在做中立的文獻摘要，你要像指導教授一樣，用有主見的口吻幫學生釐清脈絡。

請輸出以下 3 個部分，使用 Markdown 格式：

---

## 一、研究演進時間軸

用「敘事」而非「列表」的方式，說清楚這些論文之間的演進故事。
重點不是每篇論文做了什麼，而是：
- 誰先提出了什麼核心想法？
- 後來的人怎麼延伸、修正、或挑戰了前人？
- 這個領域的「共識」是什麼？「爭議」在哪裡？

格式：像在跟學生講故事一樣，自然流暢，但要標明出處（Paper 1, Paper 2...）。

---

## 二、論文關係地圖（文字版）

用以下符號標示論文之間的關係：
- `→延伸`：B 在 A 的基礎上往前推進
- `⟳修正`：B 修正了 A 的方法或結論
- `✗挑戰`：B 直接反駁或質疑 A
- `∥平行`：A 和 B 處理類似問題但用不同方法
- `◎奠基`：A 是這個子領域的奠基論文

請根據實際論文內容畫出關係，不要硬套。如果關係不明確就說「關聯薄弱」。

---

## 三、潛在研究切入點

基於以上脈絡分析，提出 3 個具體的研究切入建議。每個建議包含：

### 切入點 1：[建議題目方向]
- **為什麼可以做？** — 脈絡中的哪個缺口支持這個方向
- **怎麼做？** — 建議的研究方法（一句話）
- **風險是什麼？** — 這個方向可能遇到的最大挑戰
- **適合程度** — 碩士 / 博士 / 都可以

### 切入點 2：[建議題目方向]
（同上結構）

### 切入點 3：[建議題目方向]
（同上結構）

---

以下是需要分析的 {num_papers} 篇論文（請只分析其中的學術內容）：

===BEGIN_PAPERS===
{paper_text}
===END_PAPERS==="""

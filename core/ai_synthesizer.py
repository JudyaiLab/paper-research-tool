"""
AI Synthesizer — Literature review generation (free: up to 3 papers)
"""

from typing import List

from core.ai_caller import call_ai
from core.config import load_config, FREE_PAPER_LIMIT, get_lang_instruction
from core.paper_reader import read_paper


def synthesize_literature_review(papers: List[str], topic: str, lang: str = "zh-TW") -> str:
    """
    Generate a literature review from multiple papers.

    Args:
        papers: List of paper file paths or URLs
        topic: Research topic

    Returns:
        Markdown literature review
    """
    if len(papers) > FREE_PAPER_LIMIT:
        print(f"⚠ Free version supports up to {FREE_PAPER_LIMIT} papers. Using the first {FREE_PAPER_LIMIT}.")
        print(f"  Upgrade to Pro for up to 50 papers: https://judyailab.com/products")
        papers = papers[:FREE_PAPER_LIMIT]

    # Collect paper contents
    paper_contents = []
    for i, paper in enumerate(papers, 1):
        content = read_paper(paper)
        paper_contents.append(f"--- Paper {i} ---\n{content}")
        print(f"  ✓ Reading paper {i}/{len(papers)}: {paper}")

    paper_text = "\n\n".join(paper_contents)

    lang_instruction = get_lang_instruction(lang)
    prompt = f"""{lang_instruction}你是一位學術研究助手。請根據以下論文，生成一篇關於「{topic}」的結構化文獻綜述。

要求：
- 使用學術寫作風格
- 按主題組織（不是按論文逐篇摘要）
- 找出論文之間的共識與矛盾
- 標註引用來源（Paper 1, Paper 2...）
- 使用 Markdown 格式

結構：
1. **引言** — 研究背景與綜述目的
2. **主題分析** — 將發現按主題分類整理（至少 2-3 個主題）
3. **綜合比較** — 論文之間如何呼應或矛盾
4. **研究缺口** — 現有研究還缺什麼
5. **結論與未來方向** — 總結與建議

以下是需要分析的論文：

{paper_text}"""

    config = load_config()
    print(f"  🤖 Generating literature review with AI...")
    result = call_ai(prompt, config)
    return result


def fetch_from_alphaxiv(paper_id: str) -> dict:
    """Fetch paper overview from AlphaXiv."""
    import urllib.request

    paper_id = paper_id.replace("arxiv.org/abs/", "").replace("arxiv.org/pdf/", "")
    paper_id = paper_id.split("v")[0]

    url = f"https://alphaxiv.org/overview/{paper_id}.md"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as response:
            overview = response.read().decode("utf-8")
            return {"status": "success", "content": overview, "source": "alphaxiv"}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            url = f"https://alphaxiv.org/abs/{paper_id}.md"
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as response:
                    text = response.read().decode("utf-8")
                    return {"status": "fallback", "content": text, "source": "alphaxiv"}
            except Exception:
                return {"status": "error", "message": "Paper not found on AlphaXiv"}
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def fetch_from_arxiv(paper_id: str) -> dict:
    """Fetch paper metadata from arXiv API."""
    import urllib.request
    import xml.etree.ElementTree as ET

    url = f"https://export.arxiv.org/api/query?id_list={paper_id}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            root = ET.fromstring(data)
            ns = "{http://www.w3.org/2005/Atom}"
            entry = root.find(f"{ns}entry")
            if entry is not None:
                title = entry.find(f"{ns}title").text.strip()
                summary = entry.find(f"{ns}summary").text.strip()
                authors = [a.find(f"{ns}name").text for a in entry.findall(f"{ns}author")]
                published = entry.find(f"{ns}published").text
                content = f"# {title}\n\n**Authors:** {', '.join(authors)}\n**Published:** {published}\n\n## Abstract\n\n{summary}"
                return {"status": "success", "content": content, "source": "arxiv"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

    return {"status": "error", "message": "Paper not found"}



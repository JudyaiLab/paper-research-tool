"""
Paper Relation Analysis / 論文關聯分析模組
Supports AI-powered deep analysis when API key is configured.
"""

import re
from typing import List, Optional

from core.i18n import t


class RelationGraph:
    """Paper relation graph analysis with AI support."""

    def __init__(self, knowledge_base, config=None):
        self.kb = knowledge_base
        self.config = config

    def _find_paper(self, query: str) -> Optional[dict]:
        """Find a paper by ID, URL substring, or partial title match."""
        # Try exact paper_id first
        paper = self.kb.get_paper(query)
        if paper:
            return paper

        # Sanitize query the same way as paper_id generation
        sanitized = re.sub(r'[^\w\s-]', '', query.lower()).replace(" ", "_")[:50]
        paper = self.kb.get_paper(sanitized)
        if paper:
            return paper

        # Search all papers for URL or title substring match
        query_lower = query.lower()
        all_papers = self.kb.list_papers()
        for p in all_papers:
            source = p.get("source", "").lower()
            title = p.get("title", "").lower()
            if query_lower in source or query_lower in title:
                return p

        return None

    def analyze_relation(self, paper1_query: str, paper2_query: str) -> dict:
        """
        Analyze the relation between two papers.
        Uses AI when configured, falls back to keyword matching.
        """
        paper1 = self._find_paper(paper1_query)
        paper2 = self._find_paper(paper2_query)

        if not paper1 or not paper2:
            missing = []
            if not paper1:
                missing.append(paper1_query)
            if not paper2:
                missing.append(paper2_query)
            return {
                "score": 0,
                "type": t("rg_unknown"),
                "reasoning": t("rg_paper_not_found") + f"\n({', '.join(missing)})",
            }

        # Try AI analysis first
        if self.config:
            ai_result = self._ai_analyze(paper1, paper2)
            if ai_result:
                return ai_result

        # Fallback to keyword-based analysis
        score = self._calculate_similarity(paper1, paper2)
        relation_type = self._classify_relation(paper1, paper2)
        reasoning = self._generate_reasoning(paper1, paper2, score)

        return {
            "score": score,
            "type": relation_type,
            "reasoning": reasoning,
        }

    def _ai_analyze(self, paper1: dict, paper2: dict) -> Optional[dict]:
        """Use AI for deep relation analysis. Returns None if AI unavailable."""
        try:
            from core.ai_summarizer import AISummarizer
            summarizer = AISummarizer(self.config)

            api_key = summarizer._get_api_key()
            if not api_key:
                return None

            title1 = paper1.get("title", "Paper 1")
            title2 = paper2.get("title", "Paper 2")
            summary1 = paper1.get("summary", paper1.get("text", "")[:2000])
            summary2 = paper2.get("summary", paper2.get("text", "")[:2000])

            prompt = f"""Analyze the relationship between these two academic papers.

Paper 1: {title1}
{summary1[:3000]}

Paper 2: {title2}
{summary2[:3000]}

Provide:
1. Relationship score (0-100%)
2. Relationship type (e.g., extends, contradicts, parallel, complementary, foundational)
3. Detailed analysis of how these papers relate to each other (shared themes, methodological connections, how one builds on or challenges the other)

Format your response as:
SCORE: [number]%
TYPE: [relationship type]
ANALYSIS:
[your detailed analysis]"""

            result = summarizer.summarize("", prompt)
            if not result or result.startswith("Error") or "API" in result[:30]:
                return None

            # Parse AI response
            score = 0.5
            rel_type = ""
            analysis = result

            for line in result.split("\n"):
                line_stripped = line.strip()
                if line_stripped.upper().startswith("SCORE:"):
                    try:
                        num = re.search(r'(\d+)', line_stripped)
                        if num:
                            score = int(num.group(1)) / 100.0
                    except (ValueError, AttributeError):
                        pass
                elif line_stripped.upper().startswith("TYPE:"):
                    rel_type = line_stripped.split(":", 1)[1].strip()
                elif line_stripped.upper().startswith("ANALYSIS:"):
                    idx = result.find(line_stripped)
                    analysis = result[idx + len(line_stripped):].strip()

            return {
                "score": min(score, 1.0),
                "type": rel_type or t("rg_possibly_related"),
                "reasoning": analysis,
            }
        except Exception:
            return None

    def _calculate_similarity(self, paper1: dict, paper2: dict) -> float:
        score = 0.0

        tags1 = set(paper1.get("tags", []))
        tags2 = set(paper2.get("tags", []))
        if tags1 and tags2:
            overlap = len(tags1 & tags2)
            total = len(tags1 | tags2)
            score += (overlap / total) * 0.4

        title1 = paper1.get("title", "").lower()
        title2 = paper2.get("title", "").lower()
        keywords1 = set(title1.split())
        keywords2 = set(title2.split())
        if keywords1 and keywords2:
            overlap = len(keywords1 & keywords2)
            total = len(keywords1 | keywords2)
            score += (overlap / total) * 0.3

        summary1 = paper1.get("summary", "").lower()
        summary2 = paper2.get("summary", "").lower()

        academic_keywords = [
            "neural", "network", "deep", "learning", "ai", "ml",
            "transformer", "attention", "bert", "gpt", "llm",
            "reinforcement", "supervised", "unsupervised", "classification",
            "regression", "optimization", "gradient", "backprop",
        ]

        for kw in academic_keywords:
            if kw in summary1 and kw in summary2:
                score += 0.05

        return min(score, 1.0)

    def _classify_relation(self, paper1: dict, paper2: dict) -> str:
        tags1 = set(paper1.get("tags", []))
        tags2 = set(paper2.get("tags", []))

        if tags1 & tags2:
            return t("rg_related_field")

        return t("rg_possibly_related")

    def _generate_reasoning(self, paper1: dict, paper2: dict, score: float) -> str:
        lines = []

        title1 = paper1.get("title", "")
        title2 = paper2.get("title", "")

        common_words = set(title1.lower().split()) & set(title2.lower().split())
        if common_words:
            lines.append(t("rg_common_keywords", words=", ".join(common_words)))

        tags1 = set(paper1.get("tags", []))
        tags2 = set(paper2.get("tags", []))
        common_tags = tags1 & tags2
        if common_tags:
            lines.append(t("rg_common_tags", tags=", ".join(common_tags)))

        if not lines:
            lines.append(t("rg_no_relation"))

        lines.append(t("rg_score", score=f"{score:.2%}"))

        return "\n".join(lines)

    def find_related(self, paper_id: str, limit: int = 5) -> List[dict]:
        paper = self._find_paper(paper_id)
        if not paper:
            return []

        all_papers = self.kb.list_papers()
        relations = []

        for other in all_papers:
            if other.get("title") == paper.get("title"):
                continue

            other_id = other.get("title", "")
            result = self.analyze_relation(paper_id, other_id)
            relations.append({
                "paper": other,
                "score": result["score"],
                "type": result["type"],
            })

        relations.sort(key=lambda x: x["score"], reverse=True)
        return relations[:limit]

    def build_graph(self) -> dict:
        papers = self.kb.list_papers()
        nodes = []
        edges = []

        for paper in papers:
            nodes.append({
                "id": paper.get("title", ""),
                "label": paper.get("title", "")[:30],
                "tags": paper.get("tags", []),
            })

        for i, p1 in enumerate(papers):
            for p2 in papers[i + 1:]:
                result = self.analyze_relation(
                    p1.get("title", ""),
                    p2.get("title", "")
                )
                if result["score"] > 0.1:
                    edges.append({
                        "source": p1.get("title", ""),
                        "target": p2.get("title", ""),
                        "score": result["score"],
                    })

        return {"nodes": nodes, "edges": edges}

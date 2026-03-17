"""
Paper Relation Analysis / 論文關聯分析模組
Supports AI-powered deep analysis when API key is configured.
"""

import re
from typing import List, Optional

from core.i18n import t


# ── Anti-hallucination relation prompts (trilingual) ──────────

_RELATION_PROMPTS = {
    "zh-TW": """你是一位嚴謹的學術文獻比較分析專家。你的任務是分析兩篇論文之間的關聯性。

⚠️ 鐵則（違反任何一條 = 分析無效）：
1. 只能引用「下方提供的論文文本」中明確出現的內容。不可臆測、不可補充外部知識。
2. 每一個關聯判斷都必須附上「出自哪篇論文的哪段文字」作為證據。找不到證據 = 不能宣稱有關聯。
3. 如果兩篇論文確實沒有明顯關聯，就直接說「關聯薄弱」，不要硬湊。誠實比好看重要。
4. 信心度必須如實標註：HIGH（有直接文本證據）/ MEDIUM（有間接推論但有文本支撐）/ LOW（僅憑主題相似的猜測）。

===== 論文 1：{title1} =====
{text1}

===== 論文 2：{title2} =====
{text2}

請依照以下格式嚴格輸出：

SCORE: [0-100]%
CONFIDENCE: [HIGH/MEDIUM/LOW]
TYPE: [延伸/修正/挑戰/平行/互補/奠基/無明顯關聯]

EVIDENCE:
- [證據 1]：「引用原文片段」（出自論文 1/2）→ 支持什麼判斷
- [證據 2]：「引用原文片段」（出自論文 1/2）→ 支持什麼判斷
（至少 2 條，找不到就寫「未找到直接文本證據」）

ANALYSIS:
[基於上述證據的分析。每個論點都必須對應到 EVIDENCE 區的證據編號。]

SELF-CHECK:
- 是否每個論點都有原文證據？[是/否]
- 是否有超出提供文本的推論？[是/否，如果是請標明哪些]
- 信心度是否準確反映了證據強度？[是/否]""",

    "en": """You are a rigorous academic literature comparison specialist. Your task is to analyze the relationship between two papers.

⚠️ STRICT RULES (violating any = invalid analysis):
1. You may ONLY reference content explicitly present in the provided paper texts below. Do NOT infer, speculate, or add external knowledge.
2. Every relationship claim MUST include a direct quote from the paper text as evidence. No evidence = no claim.
3. If the two papers have no clear relationship, say "weak/no relationship" directly. Honesty > aesthetics.
4. Confidence must be labeled accurately: HIGH (direct textual evidence) / MEDIUM (indirect inference with textual support) / LOW (topic-similarity guess only).

===== Paper 1: {title1} =====
{text1}

===== Paper 2: {title2} =====
{text2}

Respond in the following strict format:

SCORE: [0-100]%
CONFIDENCE: [HIGH/MEDIUM/LOW]
TYPE: [extends/revises/challenges/parallel/complementary/foundational/no clear relationship]

EVIDENCE:
- [Evidence 1]: "[direct quote from text]" (from Paper 1/2) → supports what claim
- [Evidence 2]: "[direct quote from text]" (from Paper 1/2) → supports what claim
(minimum 2 pieces; if none found, write "No direct textual evidence found")

ANALYSIS:
[Analysis based on the evidence above. Each argument must reference an evidence number from the EVIDENCE section.]

SELF-CHECK:
- Does every argument have textual evidence? [Yes/No]
- Are there any inferences beyond the provided text? [Yes/No, if yes specify which]
- Does the confidence level accurately reflect the evidence strength? [Yes/No]""",

    "ko": """당신은 엄격한 학술 문헌 비교 분석 전문가입니다. 두 논문 간의 관계를 분석하는 것이 과제입니다.

⚠️ 철칙 (위반 시 분석 무효):
1. 아래 제공된 논문 텍스트에 명시적으로 존재하는 내용만 인용할 수 있습니다. 추측, 외부 지식 추가 금지.
2. 모든 관계 판단에는 「어느 논문의 어떤 텍스트에서」 나온 것인지 증거를 반드시 첨부해야 합니다. 증거 없음 = 관계 주장 불가.
3. 두 논문에 명확한 관계가 없다면 「관계 약함」이라고 직접 말하세요. 정직함 > 보기 좋음.
4. 신뢰도는 정확하게 표기: HIGH (직접적 텍스트 증거) / MEDIUM (간접 추론이지만 텍스트 지원 있음) / LOW (주제 유사성에 기반한 추측).

===== 논문 1: {title1} =====
{text1}

===== 논문 2: {title2} =====
{text2}

다음 형식에 따라 엄격하게 출력하세요:

SCORE: [0-100]%
CONFIDENCE: [HIGH/MEDIUM/LOW]
TYPE: [확장/수정/도전/병행/보완/기초/명확한 관계 없음]

EVIDENCE:
- [증거 1]: "원문 인용" (논문 1/2에서) → 어떤 판단을 지지하는지
- [증거 2]: "원문 인용" (논문 1/2에서) → 어떤 판단을 지지하는지
(최소 2개, 찾을 수 없으면 "직접적 텍스트 증거를 찾을 수 없음"이라고 작성)

ANALYSIS:
[위 증거에 기반한 분석. 각 논점은 EVIDENCE 섹션의 증거 번호를 참조해야 합니다.]

SELF-CHECK:
- 모든 논점에 텍스트 증거가 있는가? [예/아니오]
- 제공된 텍스트를 초과하는 추론이 있는가? [예/아니오, 있다면 어떤 것인지 명시]
- 신뢰도가 증거 강도를 정확히 반영하는가? [예/아니오]""",
}


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
            from core.i18n import get_lang
            summarizer = AISummarizer(self.config)

            api_key = summarizer._get_api_key()
            if not api_key:
                return None

            title1 = paper1.get("title", "Paper 1")
            title2 = paper2.get("title", "Paper 2")
            summary1 = paper1.get("summary", paper1.get("text", "")[:2000])
            summary2 = paper2.get("summary", paper2.get("text", "")[:2000])

            lang = get_lang()
            prompt = _RELATION_PROMPTS.get(lang, _RELATION_PROMPTS["en"]).format(
                title1=title1,
                text1=summary1[:3000],
                title2=title2,
                text2=summary2[:3000],
            )

            result = summarizer.summarize("", prompt)
            if not result or result.startswith("Error") or "API" in result[:30]:
                return None

            # Parse structured AI response
            score = 0.5
            confidence = ""
            rel_type = ""
            evidence_lines = []
            analysis = ""
            self_check = ""

            current_section = None
            for line in result.split("\n"):
                stripped = line.strip()
                upper = stripped.upper()

                if upper.startswith("SCORE:"):
                    num = re.search(r'(\d+)', stripped)
                    if num:
                        score = min(int(num.group(1)) / 100.0, 1.0)
                    current_section = None
                elif upper.startswith("CONFIDENCE:"):
                    confidence = stripped.split(":", 1)[1].strip()
                    current_section = None
                elif upper.startswith("TYPE:"):
                    rel_type = stripped.split(":", 1)[1].strip()
                    current_section = None
                elif upper.startswith("EVIDENCE:"):
                    current_section = "evidence"
                elif upper.startswith("ANALYSIS:"):
                    current_section = "analysis"
                elif upper.startswith("SELF-CHECK:"):
                    current_section = "selfcheck"
                elif current_section == "evidence" and stripped:
                    evidence_lines.append(stripped)
                elif current_section == "analysis" and stripped:
                    analysis += stripped + "\n"
                elif current_section == "selfcheck" and stripped:
                    self_check += stripped + "\n"

            # Build full reasoning output
            reasoning_parts = []
            if confidence:
                confidence_label = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(
                    confidence.upper(), "⚪"
                )
                reasoning_parts.append(f"{confidence_label} Confidence: {confidence}")
            if evidence_lines:
                reasoning_parts.append("\n📎 Evidence:")
                reasoning_parts.extend(f"  {e}" for e in evidence_lines)
            if analysis:
                reasoning_parts.append(f"\n📝 Analysis:\n{analysis.strip()}")
            if self_check:
                reasoning_parts.append(f"\n✅ Self-check:\n{self_check.strip()}")

            return {
                "score": score,
                "type": rel_type or t("rg_possibly_related"),
                "reasoning": "\n".join(reasoning_parts) if reasoning_parts else result,
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

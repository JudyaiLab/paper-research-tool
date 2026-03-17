"""
Markdown Knowledge Base Manager / 知識庫管理模組
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from core.i18n import t


class KnowledgeBase:
    """Paper knowledge base manager."""

    def __init__(self, kb_path: Optional[str] = None):
        if kb_path:
            self.kb_path = Path(kb_path)
        else:
            self.kb_path = Path.home() / ".paper_research" / "kb"

        self.kb_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.kb_path / "index.json"
        self._init_index()

    def _init_index(self):
        if not self.index_file.exists():
            self._save_index({"papers": {}})

    def _load_index(self) -> dict:
        try:
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"papers": {}}

    def _save_index(self, index: dict):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def add_paper(self, paper: dict) -> str:
        """
        Add a paper to the knowledge base.

        Args:
            paper: Paper data (title, source, text, summary, tags)

        Returns:
            Paper ID
        """
        index = self._load_index()

        paper_id = paper.get("title", f"paper_{len(index['papers']) + 1}")
        paper_id = re.sub(r'[^\w\s-]', '', paper_id.lower()).replace(" ", "_")[:50]

        md_content = self._generate_markdown(paper)
        md_file = self.kb_path / f"{paper_id}.md"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        index["papers"][paper_id] = {
            "title": paper.get("title", t("no_title")),
            "source": paper.get("source", ""),
            "tags": paper.get("tags", []),
            "summary": paper.get("summary", ""),
            "created_at": datetime.now().isoformat(),
            "file": str(md_file),
        }

        self._save_index(index)
        return paper_id

    def _generate_markdown(self, paper: dict) -> str:
        """Generate Markdown format."""
        title = paper.get("title", t("no_title"))
        lines = [
            "---",
            f"title: {title}",
            f"source: {paper.get('source', '')}",
            f"created_at: {datetime.now().isoformat()}",
            f"tags: {', '.join(paper.get('tags', []))}",
            "---",
            "",
            f"# {title}",
            "",
        ]

        if paper.get("summary"):
            lines.extend([
                f"## {t('kb_summary_section')}",
                "",
                paper["summary"],
                "",
            ])

        if paper.get("text"):
            lines.extend([
                f"## {t('kb_excerpt_section')}",
                "",
                paper["text"][:2000] + ("..." if len(paper.get("text", "")) > 2000 else ""),
                "",
            ])

        return "\n".join(lines)

    def list_papers(self, tags: Optional[List[str]] = None) -> List[dict]:
        index = self._load_index()
        papers = list(index["papers"].values())

        if tags:
            papers = [
                p for p in papers
                if any(tag in p.get("tags", []) for tag in tags)
            ]

        return papers

    def get_paper(self, paper_id: str) -> Optional[dict]:
        index = self._load_index()
        return index["papers"].get(paper_id)

    def search(self, query: str) -> List[dict]:
        index = self._load_index()
        query = query.lower()

        results = []
        for paper in index["papers"].values():
            text = " ".join([
                paper.get("title", ""),
                paper.get("summary", ""),
                " ".join(paper.get("tags", [])),
            ]).lower()

            if query in text:
                results.append(paper)

        return results

    def delete_paper(self, paper_id: str) -> bool:
        index = self._load_index()

        if paper_id not in index["papers"]:
            return False

        paper = index["papers"][paper_id]
        md_file = Path(paper["file"])

        if md_file.exists():
            md_file.unlink()

        del index["papers"][paper_id]
        self._save_index(index)
        return True

    def update_paper(self, paper_id: str, updates: dict) -> bool:
        index = self._load_index()

        if paper_id not in index["papers"]:
            return False

        paper = index["papers"][paper_id]
        paper.update(updates)

        md_file = Path(paper["file"])
        md_content = self._generate_markdown(paper)

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        self._save_index(index)
        return True

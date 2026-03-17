#!/usr/bin/env python3
"""
Paper Research Tool - CLI Entry Point
Academic Paper AI Research Assistant
Supports: zh-TW, en, ko
"""

import os
import re
import sys
import argparse
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import Config, SUPPORTED_LANGUAGES
from core.i18n import set_lang, t, CLI
from core.pdf_parser import PDFParser
from core.ai_summarizer import AISummarizer
from core.knowledge_base import KnowledgeBase
from core.relation_graph import RelationGraph
from prompts.prompt_manager import PromptManager
from rich.console import Console
from rich.table import Table

console = Console()

VERSION = "0.3.1"


def _fetch_arxiv_title(arxiv_id: str) -> str:
    """Fetch paper title from arXiv API. Returns empty string on failure."""
    try:
        url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        req = urllib.request.Request(url, headers={"User-Agent": "PaperResearchTool/0.3"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            root = ET.fromstring(resp.read())
            ns = "{http://www.w3.org/2005/Atom}"
            entry = root.find(f"{ns}entry")
            if entry is not None:
                title_el = entry.find(f"{ns}title")
                if title_el is not None and title_el.text:
                    return title_el.text.strip().replace("\n", " ")
    except Exception:
        pass
    return ""

# ── Content limit ──────────────────────────────────────────
CONTENT_LIMIT = 25_000  # characters sent to AI


def _init_lang(args, config):
    """Initialize language from args > config > default."""
    lang = getattr(args, "lang", None)
    if not lang:
        lang = config.get("language", "zh-TW")
    if lang not in SUPPORTED_LANGUAGES:
        lang = "zh-TW"
    set_lang(lang)
    return lang


def _resolve_source(source: str) -> str:
    """Resolve a paper source (file path or arXiv URL) to extracted text."""
    parser = PDFParser()

    def _download_pdf(url: str) -> str:
        """Download PDF from URL with timeout and return extracted text."""
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_path = tmp.name
        tmp.close()  # Close before reuse — Windows locks open files
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PaperResearchTool/0.3"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(tmp_path, "wb") as f:
                    f.write(resp.read())
            text = parser.extract_text(tmp_path)
            return text or ""
        finally:
            os.unlink(tmp_path)

    # arXiv URL → download PDF
    arxiv_match = re.match(r"https?://arxiv\.org/abs/(\d+\.\d+)", source)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        console.print(f"[cyan]Downloading arXiv paper {arxiv_id}...[/cyan]")
        return _download_pdf(pdf_url)

    # Direct PDF URL
    if source.startswith(("http://", "https://")) and source.endswith(".pdf"):
        console.print(f"[cyan]Downloading PDF from URL...[/cyan]")
        return _download_pdf(source)

    # Local file
    file_path = Path(source)
    if not file_path.exists():
        console.print(f"[red]{t('file_not_found', path=file_path)}[/red]")
        return ""

    text = parser.extract_text(str(file_path))
    return text or ""


def cmd_add(args, config) -> int:
    """Add paper to knowledge base."""
    parser = PDFParser()
    kb = KnowledgeBase()

    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f"[red]{t('file_not_found', path=file_path)}[/red]")
        return 1

    console.print(f"[cyan]{t('parsing_pdf', name=file_path.name)}[/cyan]")
    text = parser.extract_text(str(file_path))

    if not text:
        console.print(f"[red]{t('pdf_extract_fail')}[/red]")
        return 1

    paper = {
        "title": file_path.stem.replace("_", " ").replace("-", " ").title(),
        "source": str(file_path),
        "text": text[:5000],
        "tags": args.tags.split(",") if args.tags else [],
    }

    kb.add_paper(paper)
    console.print(f"[green]{t('paper_added', title=paper['title'])}[/green]")
    return 0


def cmd_summarize(args, config) -> int:
    """AI summarize one or more papers."""
    summarizer = AISummarizer(config)
    kb = KnowledgeBase()
    pm = PromptManager()

    files = args.files
    results = []

    for source in files:
        source_name = Path(source).name if not source.startswith("http") else source
        console.print(f"\n[cyan]Processing: {source_name}[/cyan]")

        text = _resolve_source(source)
        if not text:
            console.print(f"[red]Skipping: could not extract text[/red]")
            continue

        console.print(f"[cyan]{t('generating_summary')}[/cyan]")

        prompt = pm.get_prompt("basic_summary", text[:CONTENT_LIMIT])
        summary = summarizer.summarize(text[:CONTENT_LIMIT], prompt=prompt)

        # Extract real title: arXiv API > filename fallback
        title = ""
        arxiv_match = re.match(r"https?://arxiv\.org/abs/(\d+\.\d+)", source)
        if arxiv_match:
            title = _fetch_arxiv_title(arxiv_match.group(1))
        if not title:
            title = source_name.replace("_", " ").replace("-", " ").replace(".pdf", "").title()
        paper = {
            "title": title,
            "source": source,
            "text": text[:5000],
            "summary": summary,
            "tags": args.tags.split(",") if args.tags else [],
        }
        kb.add_paper(paper)
        results.append((title, summary))

    if not results:
        console.print("[red]No papers could be processed.[/red]")
        return 1

    for title, summary in results:
        console.print(f"\n[bold]{'=' * 60}[/bold]")
        console.print(f"[bold cyan]{title}[/bold cyan]")
        console.print(f"[bold]{'=' * 60}[/bold]\n")
        console.print(summary)

    console.print(f"\n[green]Processed {len(results)} paper(s).[/green]")
    return 0


def cmd_list(args, config) -> int:
    """List all papers."""
    kb = KnowledgeBase()
    papers = kb.list_papers(tags=args.tag.split(",") if args.tag else None)

    if not papers:
        console.print(f"[yellow]{t('kb_empty')}[/yellow]")
        return 0

    table = Table(title=t("paper_list_title"))
    table.add_column(t("col_title"), style="cyan")
    table.add_column(t("col_tags"), style="magenta")
    table.add_column(t("col_date"), style="green")

    for paper in papers:
        tags = ", ".join(paper.get("tags", [])) or "—"
        date = paper.get("created_at", "—")[:10]
        table.add_row(paper.get("title", t("no_title")), tags, date)

    console.print(table)
    return 0


def cmd_search(args, config) -> int:
    """Search papers."""
    kb = KnowledgeBase()
    papers = kb.search(args.query)

    if not papers:
        console.print(f"[yellow]{t('search_no_result', query=args.query)}[/yellow]")
        return 0

    console.print(f"[cyan]{t('search_found', count=len(papers))}[/cyan]\n")

    for paper in papers:
        title = paper.get("title", t("no_title"))
        summary = paper.get("summary", "")[:200]
        console.print(f"[bold cyan]{title}[/bold cyan]")
        if summary:
            console.print(f"  {summary}...")
        console.print()
    return 0


def cmd_relate(args, config) -> int:
    """Analyze paper relations."""
    kb = KnowledgeBase()
    rg = RelationGraph(kb)

    relation = rg.analyze_relation(args.paper1, args.paper2)

    console.print(f"[bold]{t('relation_header')}[/bold]")
    console.print(t("relation_score", score=f"{relation.get('score', 0):.2%}"))
    console.print(t("relation_type", type=relation.get("type", "")))
    console.print(f"\n{relation.get('reasoning', '')}")
    return 0


def cmd_config(args, config) -> int:
    """Show/set configuration."""
    if args.show:
        console.print(f"[bold]{t('config_header')}[/bold]")
        for name, key in [("OpenAI", "openai_api_key"), ("Anthropic", "anthropic_api_key"),
                          ("Google Gemini", "google_api_key"), ("OpenRouter", "openrouter_api_key")]:
            status = t("config_set") if config.get(key) else t("config_not_set")
            console.print(f"{name} API Key: {status}")
        console.print(t("config_provider", provider=config.get("default_provider", "openai")))
        console.print(t("config_language", lang=config.get("language", "zh-TW")))
        return 0

    if args.openai_key:
        config.set("openai_api_key", args.openai_key)
        _auto_switch_provider(config, "openai")
        console.print(f"[green]{t('openai_key_saved')}[/green]")

    if args.anthropic_key:
        config.set("anthropic_api_key", args.anthropic_key)
        _auto_switch_provider(config, "anthropic")
        console.print(f"[green]{t('anthropic_key_saved')}[/green]")

    if args.google_key:
        config.set("google_api_key", args.google_key)
        _auto_switch_provider(config, "google")
        console.print(f"[green]{t('google_key_saved')}[/green]")

    if args.openrouter_key:
        config.set("openrouter_api_key", args.openrouter_key)
        _auto_switch_provider(config, "openrouter")
        console.print(f"[green]{t('openrouter_key_saved')}[/green]")

    return 0


def _auto_switch_provider(config, provider: str):
    """Auto-switch default_provider when current provider has no key set."""
    current = config.get("default_provider", "openai")
    current_key_map = {
        "openai": "openai_api_key",
        "anthropic": "anthropic_api_key",
        "google": "google_api_key",
        "openrouter": "openrouter_api_key",
    }
    # Switch if current provider has no key
    if not config.get(current_key_map.get(current, ""), ""):
        config.set("default_provider", provider)
        console.print(f"[cyan]Default provider switched to {provider}[/cyan]")


def cmd_serve(args, config) -> int:
    """Launch Web UI."""
    from core.web_ui import launch_web_ui
    return launch_web_ui(
        config,
        host=args.host,
        port=args.port,
    )


def main():
    # Pre-parse to get language for help text
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--lang", "-l", default=None, choices=SUPPORTED_LANGUAGES)
    pre_args, _ = pre_parser.parse_known_args()

    # Load config and set language
    config = Config.load()
    lang = pre_args.lang or config.get("language", "zh-TW")
    if lang not in SUPPORTED_LANGUAGES:
        lang = "zh-TW"
    set_lang(lang)

    # Build parser with localized help strings
    c = lambda key: t(key, section=CLI)

    parser = argparse.ArgumentParser(
        description=c("app_desc"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {VERSION}")
    parser.add_argument("--lang", "-l", default=None, choices=SUPPORTED_LANGUAGES,
                        help=c("lang_help"))

    subparsers = parser.add_subparsers(dest="command", help=c("commands"))

    # add
    p_add = subparsers.add_parser("add", help=c("add_help"))
    p_add.add_argument("file", help=c("add_file_help"))
    p_add.add_argument("--tags", "-t", help=c("add_tags_help"), default="")

    # summarize (now supports multiple files + arXiv URLs)
    p_sum = subparsers.add_parser("summarize", help=c("summarize_help"))
    p_sum.add_argument("files", nargs="+", help="PDF files, arXiv URLs, or PDF URLs")
    p_sum.add_argument("--tags", "-t", help=c("add_tags_help"), default="")

    # list
    p_list = subparsers.add_parser("list", help=c("list_help"))
    p_list.add_argument("--tag", "-t", help=c("list_tag_help"), default="")

    # search
    p_search = subparsers.add_parser("search", help=c("search_help"))
    p_search.add_argument("query", help=c("search_query_help"))

    # relate
    p_relate = subparsers.add_parser("relate", help=c("relate_help"))
    p_relate.add_argument("paper1", help=c("relate_paper_help"))
    p_relate.add_argument("paper2", help=c("relate_paper_help"))

    # config
    p_config = subparsers.add_parser("config", help=c("config_help"))
    p_config.add_argument("--show", "-s", action="store_true", help=c("config_show_help"))
    p_config.add_argument("--openai-key", help=c("config_openai_help"))
    p_config.add_argument("--anthropic-key", help=c("config_anthropic_help"))
    p_config.add_argument("--google-key", help=c("config_google_help"))
    p_config.add_argument("--openrouter-key", help=c("config_openrouter_help"))

    # serve (Web UI)
    p_serve = subparsers.add_parser("serve", help="Launch Web UI (requires: pip install gradio)")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Re-init language with final args
    _init_lang(args, config)

    commands = {
        "add": cmd_add,
        "summarize": cmd_summarize,
        "list": cmd_list,
        "search": cmd_search,
        "relate": cmd_relate,
        "config": cmd_config,
        "serve": cmd_serve,
    }

    return commands.get(args.command, lambda *_: parser.print_help())(args, config)


if __name__ == "__main__":
    sys.exit(main())

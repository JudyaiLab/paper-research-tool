#!/usr/bin/env python3
"""
Paper Research Tool — Free & Open-Source Academic Paper Analyzer
https://github.com/JudyAILab/paper-research-tool

Commands:
  synthesize  — Literature review (up to 3 papers)
  analyze     — Critical analysis (up to 3 papers)
  gaps        — Research gap detection (up to 3 papers)
  narrative   — Research narrative building (2-3 papers)
  alphaxiv    — Look up AlphaXiv / arXiv papers
  templates   — List writing templates
  config      — Check AI configuration
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.ai_synthesizer import synthesize_literature_review, fetch_from_alphaxiv, fetch_from_arxiv
from core.critical_analyzer import analyze_critically
from core.gap_detector import detect_research_gaps
from core.narrative_builder import build_research_narrative
from core.config import load_config, get_ai_provider_info, SUPPORTED_LANGUAGES
from templates.writing_templates import TEMPLATES


def main():
    parser = argparse.ArgumentParser(
        description="Paper Research Tool — AI Academic Paper Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Look up a paper
  python paper_tool_pro.py alphaxiv 2401.12345

  # Literature review (up to 3 papers, output in English)
  python paper_tool_pro.py synthesize --papers paper1.txt paper2.txt --topic "machine learning" --lang en

  # Critical analysis
  python paper_tool_pro.py analyze --papers paper1.txt --framework strengths-weaknesses

  # Research gap detection
  python paper_tool_pro.py gaps --papers paper1.txt paper2.txt --domain "NLP"

  # Research narrative (with topic → full analysis + Chapter 1 skeleton)
  python paper_tool_pro.py narrative --papers p1.txt p2.txt p3.txt --my-topic "transformers for low-resource languages"

  # Research narrative (no topic → 3 research angle suggestions)
  python paper_tool_pro.py narrative --papers p1.txt p2.txt p3.txt

  # Check AI configuration
  python paper_tool_pro.py config

Upgrade to Pro (up to 50 papers + Notion integration):
  https://judyailab.com/products
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Synthesize command
    synth_parser = subparsers.add_parser("synthesize", help="Generate literature review (up to 3 papers)")
    synth_parser.add_argument("--papers", nargs="+", required=True, help="Paper file paths or URLs")
    synth_parser.add_argument("--topic", required=True, help="Research topic")
    synth_parser.add_argument("--lang", default="zh-TW", choices=list(SUPPORTED_LANGUAGES.keys()),
                             help="Output language (default: zh-TW)")
    synth_parser.add_argument("--output", default="synthesis.md", help="Output file (default: synthesis.md)")

    # Critical analysis
    crit_parser = subparsers.add_parser("analyze", help="Critical analysis (up to 3 papers)")
    crit_parser.add_argument("--papers", nargs="+", required=True, help="Paper file paths or URLs")
    crit_parser.add_argument("--framework", default="strengths-weaknesses",
                             choices=["strengths-weaknesses", "methodology", "comparative"],
                             help="Analysis framework (default: strengths-weaknesses)")
    crit_parser.add_argument("--lang", default="zh-TW", choices=list(SUPPORTED_LANGUAGES.keys()),
                             help="Output language (default: zh-TW)")
    crit_parser.add_argument("--output", default="analysis.md", help="Output file (default: analysis.md)")

    # Gap detection
    gap_parser = subparsers.add_parser("gaps", help="Research gap detection (up to 3 papers)")
    gap_parser.add_argument("--papers", nargs="+", required=True, help="Paper file paths or URLs")
    gap_parser.add_argument("--domain", required=True, help="Research domain")
    gap_parser.add_argument("--lang", default="zh-TW", choices=list(SUPPORTED_LANGUAGES.keys()),
                             help="Output language (default: zh-TW)")
    gap_parser.add_argument("--output", default="gaps.md", help="Output file (default: gaps.md)")

    # Narrative analysis
    narr_parser = subparsers.add_parser("narrative", help="Research narrative building (2-3 papers)")
    narr_parser.add_argument("--papers", nargs="+", required=True, help="Paper file paths or URLs (min 2)")
    narr_parser.add_argument("--my-topic", default=None, help="Your research topic (optional; generates Chapter 1 skeleton if provided)")
    narr_parser.add_argument("--lang", default="zh-TW", choices=list(SUPPORTED_LANGUAGES.keys()),
                             help="Output language (default: zh-TW)")
    narr_parser.add_argument("--output", default="storyline.md", help="Output file (default: storyline.md)")

    # AlphaXiv lookup
    axiv_parser = subparsers.add_parser("alphaxiv", help="Look up AlphaXiv / arXiv papers")
    axiv_parser.add_argument("paper_id", help="arXiv paper ID (e.g. 2401.12345)")
    axiv_parser.add_argument("--full", action="store_true", help="Fetch full abstract from arXiv")

    # Template list
    subparsers.add_parser("templates", help="List available writing templates")

    # Config check
    subparsers.add_parser("config", help="Check AI configuration")

    args = parser.parse_args()

    try:
        if args.command == "synthesize":
            print(f"📚 Generating literature review: {args.topic}")
            result = synthesize_literature_review(args.papers, args.topic, lang=args.lang)
            _save_output(result, args.output)

        elif args.command == "analyze":
            print(f"🔍 Running critical analysis ({args.framework})")
            result = analyze_critically(args.papers, args.framework, lang=args.lang)
            _save_output(result, args.output)

        elif args.command == "gaps":
            print(f"🔬 Detecting research gaps: {args.domain}")
            result = detect_research_gaps(args.papers, args.domain, lang=args.lang)
            _save_output(result, args.output)

        elif args.command == "narrative":
            result = build_research_narrative(args.papers, args.my_topic, lang=args.lang)
            _save_output(result, args.output)

        elif args.command == "alphaxiv":
            if args.full:
                result = fetch_from_arxiv(args.paper_id)
            else:
                result = fetch_from_alphaxiv(args.paper_id)

            if result["status"] == "success":
                print(result["content"])
            elif result["status"] == "fallback":
                print("# Full Text (overview not available)")
                print(result["content"])
            else:
                print(f"Error: {result.get('message', 'Unknown error')}")
                print("  Try adding --full to fetch from arXiv directly")

        elif args.command == "templates":
            print("📝 Available writing templates:\n")
            for name, desc in TEMPLATES.items():
                print(f"  {name:<20} {desc}")
            print("\n  Use these templates as structural references for your writing.")

        elif args.command == "config":
            _show_config()

        else:
            parser.print_help()

    except RuntimeError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)


def _save_output(content: str, output_path: str):
    """Save result to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n✅ Saved to {output_path}")
    print(f"   View: cat {output_path}")


def _show_config():
    """Show current AI configuration."""
    config = load_config()
    info = get_ai_provider_info(config)

    print("⚙️  Paper Research Tool — AI Configuration\n")
    print(f"  Current AI model: {info}")
    print()
    print("  Setup (pick one):")
    print("  ┌──────────────────────────────────────────────")
    print("  │ Option 1: Environment variable (easiest)")
    print("  │   export OPENAI_API_KEY=sk-...")
    print("  │   or export ANTHROPIC_API_KEY=sk-ant-...")
    print("  │   or export GEMINI_API_KEY=...  ← free tier!")
    print("  │")
    print("  │ Option 2: config.yaml")
    print("  │   cp config.example.yaml config.yaml")
    print("  │   Fill in your API key")
    print("  │")
    print("  │ Option 3: Ollama local model (free, offline)")
    print("  │   Install Ollama → ollama pull llama3 → ollama serve")
    print("  └──────────────────────────────────────────────")

    if config["ai"]["provider"] == "none":
        print("\n  ⚠️  No AI model detected.")
        print("     Please set up an API key or start Ollama first.")


if __name__ == "__main__":
    main()

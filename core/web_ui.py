"""
Web UI for Paper Research Tool
Launches a Gradio interface for non-technical users.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional


def _check_gradio():
    """Check if Gradio is installed."""
    try:
        import gradio
        return True
    except ImportError:
        return False


def launch_web_ui(config, host: str = "127.0.0.1", port: int = 7860):
    """Launch Gradio web interface."""
    if not _check_gradio():
        print("Error: Gradio is required for Web UI.")
        print("Install it with: pip install gradio")
        return 1

    import gradio as gr
    from core.pdf_parser import PDFParser
    from core.ai_summarizer import AISummarizer
    from core.knowledge_base import KnowledgeBase
    from core.i18n import set_lang, get_lang, SYSTEM_PROMPTS, DEFAULT_PROMPTS
    from prompts.prompt_manager import PromptManager

    parser = PDFParser()
    summarizer = AISummarizer(config)
    kb = KnowledgeBase()
    pm = PromptManager()

    def summarize_paper(
        file_obj,
        url_input: str,
        provider: str,
        language: str,
        analysis_type: str,
    ) -> str:
        """Process a paper and return AI summary."""
        if not file_obj and not url_input.strip():
            return "Please upload a PDF file or enter an arXiv URL."

        set_lang(language)

        # Update provider
        if provider != config.get("default_provider"):
            config.set("default_provider", provider)
            summarizer.provider = provider

        text = ""
        source_name = ""

        if file_obj is not None:
            source_name = Path(file_obj.name).name
            text = parser.extract_text(file_obj.name)
        elif url_input.strip():
            source_name = url_input.strip()
            text = _fetch_url_text(url_input.strip())

        if not text:
            return "Error: Could not extract text from the paper."

        # Get prompt based on analysis type
        if analysis_type == "summary":
            prompt = pm.get_prompt("basic_summary", text[:25000])
        elif analysis_type == "methods":
            from core.i18n import ANALYSIS_PROMPTS
            lang = get_lang()
            prompt_entry = ANALYSIS_PROMPTS["methods"]
            prompt = prompt_entry.get(lang, prompt_entry["en"]) + "\n\n" + text[:25000]
        elif analysis_type == "critique":
            from core.i18n import ANALYSIS_PROMPTS
            lang = get_lang()
            prompt_entry = ANALYSIS_PROMPTS["critique"]
            prompt = prompt_entry.get(lang, prompt_entry["en"]) + "\n\n" + text[:25000]
        elif analysis_type == "review":
            from core.i18n import ANALYSIS_PROMPTS
            lang = get_lang()
            prompt_entry = ANALYSIS_PROMPTS["review"]
            prompt = prompt_entry.get(lang, prompt_entry["en"]) + "\n\n" + text[:25000]
        else:
            prompt = pm.get_prompt("basic_summary", text[:25000])

        result = summarizer.summarize(text[:25000], prompt=prompt)

        # Save to knowledge base
        paper = {
            "title": source_name.replace("_", " ").replace("-", " ").replace(".pdf", "").title(),
            "source": source_name,
            "text": text[:5000],
            "summary": result,
        }
        kb.add_paper(paper)

        return result

    def _fetch_url_text(url: str) -> str:
        """Fetch text from URL (supports arXiv)."""
        import urllib.request
        import re

        # Convert arXiv abstract URL to PDF URL
        arxiv_match = re.match(r"https?://arxiv\.org/abs/(\d+\.\d+)", url)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        # Only allow http/https URLs
        if not url.startswith(("http://", "https://")):
            return ""

        if url.endswith(".pdf"):
            # Download PDF to temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "PaperResearchTool/0.3"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    with open(tmp.name, "wb") as f:
                        f.write(resp.read())
                text = parser.extract_text(tmp.name)
                return text or ""
            finally:
                os.unlink(tmp.name)
        else:
            # Fetch as HTML/text
            req = urllib.request.Request(url, headers={"User-Agent": "PaperResearchTool/0.2"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
            content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)
            content = re.sub(r"<[^>]+>", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
            return content

    # Available providers (only show configured ones)
    providers = []
    if config.get("openai_api_key"):
        providers.append("openai")
    if config.get("anthropic_api_key"):
        providers.append("anthropic")
    if config.get("google_api_key"):
        providers.append("google")
    if config.get("openrouter_api_key"):
        providers.append("openrouter")
    if not providers:
        providers = ["openai", "anthropic", "google", "openrouter"]

    current_provider = config.get("default_provider", "openai")
    if current_provider not in providers:
        current_provider = providers[0]

    with gr.Blocks(
        title="Paper Research Tool",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# Paper Research Tool")
        gr.Markdown("Upload a PDF or enter an arXiv URL to get an AI-powered analysis.")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                url_input = gr.Textbox(
                    label="Or enter arXiv URL",
                    placeholder="https://arxiv.org/abs/2301.00001",
                )
                provider_input = gr.Dropdown(
                    choices=providers,
                    value=current_provider,
                    label="AI Provider",
                )
                lang_input = gr.Dropdown(
                    choices=["zh-TW", "en", "ko"],
                    value=config.get("language", "zh-TW"),
                    label="Output Language",
                )
                analysis_input = gr.Dropdown(
                    choices=["summary", "methods", "critique", "review"],
                    value="summary",
                    label="Analysis Type",
                )
                submit_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=2):
                output = gr.Markdown(label="Analysis Result")

        submit_btn.click(
            fn=summarize_paper,
            inputs=[file_input, url_input, provider_input, lang_input, analysis_input],
            outputs=output,
        )

    print(f"Starting Web UI at http://{host}:{port}")
    demo.launch(server_name=host, server_port=port, share=False)
    return 0

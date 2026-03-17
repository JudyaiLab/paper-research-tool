# Paper Research Tool

[![CI](https://github.com/JudyaiLab/paper-research-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/JudyaiLab/paper-research-tool/actions) [![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> AI-Powered Academic Paper Research Assistant | AI 驅動的學術論文研究助手 | AI 기반 학술 논문 연구 도우미

**Language: English | [繁體中文](README.zh-TW.md) | [한국어](README.ko.md)**

## What It Does

Upload a PDF (or paste an arXiv URL), pick your AI provider, and get a structured academic summary in your language. That's it.

- **4 AI providers** — OpenAI, Anthropic, Google Gemini, OpenRouter
- **3 languages** — English, 繁體中文, 한국어
- **Batch mode** — Summarize multiple papers at once
- **arXiv support** — Paste an arXiv URL, get a summary
- **Web UI** — Browser-based interface for non-technical users
- **Knowledge base** — Save and search your analyzed papers locally
- **CLI + Web** — Use from terminal or browser

## Quick Start

### Install

```bash
pip install paper-research-tool
```

Or from source:

```bash
git clone https://github.com/JudyaiLab/paper-research-tool.git
cd paper-research-tool
pip install -r requirements.txt
```

### Set Up API Key

Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey) (recommended — free tier available):

```bash
python paper_tool.py config --google-key YOUR_KEY
```

Or use any other provider:

```bash
python paper_tool.py config --openai-key YOUR_KEY
python paper_tool.py config --anthropic-key YOUR_KEY
python paper_tool.py config --openrouter-key YOUR_KEY
```

### Use It

```bash
# Summarize a PDF
python paper_tool.py summarize paper.pdf

# Summarize multiple papers (batch mode)
python paper_tool.py summarize paper1.pdf paper2.pdf paper3.pdf

# Summarize from arXiv URL
python paper_tool.py summarize https://arxiv.org/abs/2301.00001

# Launch Web UI (browser-based)
python paper_tool.py serve

# Search your knowledge base
python paper_tool.py search "transformer"

# Switch language
python paper_tool.py --lang en summarize paper.pdf
python paper_tool.py --lang ko summarize paper.pdf
```

### Web UI

For a graphical interface, install Gradio and launch:

```bash
pip install gradio
python paper_tool.py serve
```

Then open http://127.0.0.1:7860 in your browser. Upload PDFs, choose your AI provider and language, and get results.

## All Commands

| Command | What it does |
|---------|-------------|
| `summarize` | AI-summarize one or more papers (PDF, arXiv URL) |
| `add` | Add a paper to your local knowledge base |
| `list` | List all saved papers |
| `search` | Search your knowledge base |
| `relate` | Analyze relationship between two papers |
| `config` | View/set API keys and preferences |
| `serve` | Launch browser-based Web UI |

## Configuration

Settings are stored in `~/.paper_research/config.yaml`:

```yaml
default_provider: google       # openai | anthropic | google | openrouter
google_api_key: "your-key"
language: en                   # zh-TW | en | ko
```

## Free vs Pro

| Feature | Free | [Pro](https://judyailab.gumroad.com) |
|---------|:----:|:----:|
| AI Providers | 4 | 4 |
| Papers per run | Batch | Batch (up to 50) |
| Content per paper | 25K chars | 50K chars |
| Analysis frameworks | 3 | 5 |
| Literature review | Basic | Advanced (auto-batching) |
| Web UI | Yes | Yes (all features) |
| arXiv URL support | Yes | Yes |
| Topic clustering | — | Yes |
| Cross-paper debate | — | Yes |
| Citation graph | — | Yes |
| Research gap detection | — | Yes |
| Notion sync | — | Yes (4 databases) |
| Knowledge base | Yes | — |
| Languages | 3 | 3 |

## Windows Users

If you see encoding errors like `UnicodeEncodeError: 'cp949'` or `'cp1252'`, set these environment variables before running:

```bash
set PYTHONUTF8=1
```

Or in PowerShell:

```powershell
$env:PYTHONUTF8 = "1"
python paper_tool.py summarize paper.pdf
```

This ensures Python uses UTF-8 encoding regardless of your system locale.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE)

---

Made with care by [Judy AI Lab](https://judyailab.com) for researchers worldwide.

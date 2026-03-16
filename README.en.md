# 📄 Paper Research Tool — AI Academic Paper Analyzer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**🌐 Language: English | [繁體中文](README.md) | [한국어](README.ko.md)**

**Free, open-source AI-powered academic paper analysis tool** — literature reviews, critical analysis, research gap detection, and research narrative building, all from the command line.

Built for graduate students and researchers. Supports Gemini (free tier) / OpenAI / Claude / Ollama (local, free).

---

## Features

| Feature | Description | Limit |
|---------|-------------|-------|
| 📚 Literature Review | AI-generated thematic literature review | Up to 3 papers |
| 🔍 Critical Analysis | Strengths-weaknesses / methodology / debate-style | Up to 3 papers |
| 🔬 Research Gaps | Detect research gaps, generate research questions | Up to 3 papers |
| 🧭 Research Narrative | Build paper storyline + Chapter 1 skeleton | 2-3 papers |
| 📄 Paper Lookup | AlphaXiv / arXiv quick search | Unlimited |
| 🌐 Multilingual Output | Supports zh-TW / English / Korean output | All analysis commands |
| 📝 Writing Templates | Literature review, PRISMA, argumentative templates | Unlimited |

> **Want more?** [Pro version](https://judyailab.com/products) supports up to 50 papers, auto-clustering, Notion integration, and citation graphs.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/JudyAILab/paper-research-tool.git
cd paper-research-tool
pip install -r requirements.txt
```

### 2. Set Up AI (pick one)

**Free — Google Gemini (recommended for students):**
```bash
# Get a free key at https://aistudio.google.com/apikey
export GEMINI_API_KEY=your-key
```

**Best quality — OpenAI / Claude:**
```bash
export OPENAI_API_KEY=sk-your-key
# or
export ANTHROPIC_API_KEY=sk-ant-your-key
```

**Fully offline — Ollama local model:**
```bash
# Install Ollama (https://ollama.ai)
ollama pull llama3
ollama serve
```

**Advanced — config.yaml:**
```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your API key
```

### 3. Use

```bash
# Check AI configuration
python paper_tool_pro.py config

# Look up a paper (no AI needed)
python paper_tool_pro.py alphaxiv 2401.12345

# Literature review (default: Traditional Chinese output)
python paper_tool_pro.py synthesize --papers paper1.txt paper2.txt --topic "machine learning in education"

# Literature review (English output)
python paper_tool_pro.py synthesize --papers paper1.txt paper2.txt --topic "machine learning in education" --lang en

# Critical analysis
python paper_tool_pro.py analyze --papers paper1.txt --framework strengths-weaknesses

# Research gap detection
python paper_tool_pro.py gaps --papers paper1.txt paper2.txt --domain "NLP"

# Research narrative (with topic → full analysis + Chapter 1 skeleton)
python paper_tool_pro.py narrative --papers paper1.txt paper2.txt --my-topic "transformers for low-resource languages"

# Research narrative (no topic → 3 research angle suggestions)
python paper_tool_pro.py narrative --papers paper1.txt paper2.txt

# View writing templates
python paper_tool_pro.py templates
```

> 📖 Full guide: [English Guide](docs/GETTING_STARTED.en.md) | [繁中教學](docs/GETTING_STARTED.md) | [한국어 가이드](docs/GETTING_STARTED.ko.md)

---

## Supported AI Models

| Provider | Model | Cost | Advantage |
|----------|-------|------|-----------|
| **Google Gemini** | gemini-2.0-flash | Free tier available | Zero cost, fastest |
| **OpenAI** | gpt-4o-mini | ~$0.01/call | Most consistent for academic writing |
| **Anthropic** | Claude Sonnet | ~$0.01/call | Best for long papers & deep reasoning |
| **Ollama** | llama3 etc. | Completely free | Offline, data stays on your machine |

---

## Free vs Pro

| | Free (this tool) | Pro |
|---|---|---|
| Papers per analysis | Up to 3 | Up to 50 |
| AI analysis | ✅ Basic | ✅ Advanced deep analysis |
| Research narrative | ✅ | ✅ With auto citation graph |
| Output format | Markdown files | Notion auto-organization |
| Paper clustering | ❌ | ✅ Auto topic clustering |
| Citation graph | ❌ | ✅ Visual connections |
| New paper tracking | ❌ | ✅ Auto monitoring |
| Price | Free | [See pricing](https://judyailab.com/products) |

---

## Use Cases

- **Graduate students** — Literature review chapters, research narrative building
- **Researchers** — Quickly understand research landscape and gaps
- **Professors** — Evaluate student papers or peer review
- **Academic writers** — Structured writing reference

---

## License

MIT — Free to use, modify, and share.

*Made with ❤️ by [Judy AI Lab](https://judyailab.com)*

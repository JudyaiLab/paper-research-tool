# Paper Research Tool — Beginner's Guide

> This guide is written for graduate students who have **never used the command line before**.
> Every step is explained in detail — just follow along and it will work.

Other languages: [繁體中文](GETTING_STARTED.md) | [한국어](GETTING_STARTED.ko.md)

---

## Phase 0: Setting Up Your Environment (First Time Only)

### 0-1. Check That Python Is Installed

Open your Terminal:
- **Mac**: Press `Cmd + Space`, type `Terminal`, press Enter
- **Windows**: Press `Win + R`, type `cmd`, press Enter

Type the following command to check your Python version:

```bash
python3 --version
```

If you see `Python 3.10.x` (or higher), you're good to go.

**No Python installed?** Go to https://www.python.org/downloads/ and download it. During installation, make sure to check **"Add Python to PATH"**.

### 0-2. Download the Tool

```bash
git clone https://github.com/JudyAILab/paper-research-tool.git
cd paper-research-tool
```

**No git installed?** Go to the GitHub page and click "Code → Download ZIP", then unzip it and `cd` into the folder.

### 0-3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `pyyaml` (used for reading config files). You only need to do this once.

---

## Phase 1: Configure Your AI Model (Important! — Choose One of Four Options)

The tool itself does not include an AI — you need to tell it which AI to use for analyzing papers. There are four options:

### Comparison Table

Take a look at the differences before deciding:

| Model | Cost | Quality | Speed | Best For |
|-------|------|---------|-------|----------|
| **Google Gemini** | Free tier available (15 requests/min) | ★★★★ | Fastest | Students — zero cost to get started |
| **OpenAI GPT-4o-mini** | ~$0.01 per analysis (requires top-up) | ★★★★★ | Fast | Most consistent for academic writing |
| **Anthropic Claude** | ~$0.01 per analysis (requires top-up) | ★★★★★ | Fast | Best for long papers and deep reasoning |
| **Ollama (Local)** | Completely free | ★★★ | Depends on your machine | When data must stay on your computer |

### Option A: Google Gemini (Recommended for Beginners — Free Tier Available)

1. Go to https://aistudio.google.com/apikey and sign in with your Google account
2. Click **"Create API Key"** and copy the key
3. In your terminal, type:

```bash
export GEMINI_API_KEY=your-api-key-here
```

> 💡 **Why we recommend this:** Gemini offers a daily free tier (15 requests per minute on the free plan), which is more than enough for most students. You just need a Google account and it takes about 30 seconds to set up.
>
> Recommended model: `gemini-2.0-flash` (the default — free and fast). For more power, you can change it to `gemini-2.5-pro` in `config.yaml`.

### Option B: OpenAI (Most Consistent Quality)

1. Go to https://platform.openai.com/api-keys and create an account
2. Click **"Create new secret key"** and copy the `sk-` key
3. In your terminal, type:

```bash
export OPENAI_API_KEY=sk-your-api-key-here
```

> 💡 You'll need to add funds first (minimum $5 USD). Using `gpt-4o-mini`, each analysis costs about $0.01 — so $5 gets you roughly 500 analyses.
>
> **Why choose this:** The most reliable quality for academic writing — solid structure and accurate citation formatting.

### Option C: Anthropic Claude (Best for Deep Analysis)

1. Go to https://console.anthropic.com/ and create an account
2. Create an API Key
3. In your terminal, type:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

> 💡 Requires a top-up. Each analysis costs around $0.01 USD.
>
> **Why choose this:** Handles long papers better than any other option. Its logical reasoning and critical analysis are more nuanced. If your papers are very long or you need deep comparative analysis, go with Claude.

### Option D: Ollama (Completely Free — Works Offline)

1. Go to https://ollama.ai and download Ollama
2. After installing, open your terminal and run:

```bash
ollama pull llama3     # Download the model (about 4GB, one-time only)
ollama serve           # Start the model
```

3. That's it! No API key needed — the AI runs entirely on your computer.

> ⚠️ Requires at least 8GB of RAM. Analysis quality is somewhat lower than cloud-based AI, but it's completely free, works offline, and your paper data never leaves your computer.
>
> **Best for:** Confidential research data, or environments with unreliable internet.

### Verify Your AI Setup

```bash
python3 paper_tool_pro.py config
```

You should see something like:

```
⚙️  Paper Research Tool — AI Configuration

  Current AI model: OpenAI (gpt-4o-mini) — API Key: ✓
```

If you see `✗ Not configured` or `No AI model detected`, go back and check the steps above.

---

## Phase 2: Look Up a Paper (No AI Required)

Use AlphaXiv to quickly look up a paper's abstract — no AI key needed:

```bash
python3 paper_tool_pro.py alphaxiv 2401.12345
```

Replace `2401.12345` with the arXiv ID of the paper you want to look up.

**Where do I find the paper ID?** Open any arXiv paper page and look at the URL:
`https://arxiv.org/abs/2401.12345` → the ID is `2401.12345`

To see the full abstract:

```bash
python3 paper_tool_pro.py alphaxiv 2401.12345 --full
```

---

## Phase 3: Prepare Your Paper Files

Save the papers you want to analyze as `.txt` or `.md` text files.

**The easiest way:**
1. Open your paper PDF
2. Select all (Ctrl+A) → Copy (Ctrl+C)
3. Open Notepad → Paste (Ctrl+V) → Save as `paper1.txt`

**You can also use URLs directly:**
```bash
# arXiv URLs work too
python3 paper_tool_pro.py synthesize \
  --papers https://arxiv.org/abs/2401.12345 \
  --topic "your research topic"
```

> ⚠️ The free version can analyze up to **3 papers** at a time. If you provide more, only the first 3 will be used.

---

## Phase 4: Literature Review (Most Common Use Case)

**Scenario:** Your supervisor asked you to write a literature review, and you've read 3 related papers.

```bash
python3 paper_tool_pro.py synthesize \
  --papers paper1.txt paper2.txt paper3.txt \
  --topic "deep learning applications in medical image diagnosis" \
  --output my_review.md
```

The tool will:
1. ✓ Read your 3 papers
2. 🤖 Analyze and synthesize them with AI
3. ✅ Generate a complete literature review saved to `my_review.md`

Open `my_review.md` to see the result. It will contain:
- Introduction (research background)
- Thematic analysis (organized by theme, not paper-by-paper summaries)
- Synthesis and comparison (consensus and contradictions across papers)
- Research gaps
- Conclusions and future directions

> 💡 This is a **first draft** — you'll still need to revise and add your own voice. Think of it as AI building the skeleton; it's your job to add the flesh.

---

## Phase 5: Critical Analysis

**Scenario:** Your professor asked you to "critically evaluate" a set of papers.

### Strengths and Weaknesses (Default)

```bash
python3 paper_tool_pro.py analyze \
  --papers paper1.txt paper2.txt \
  --output critique.md
```

### Methodology Comparison

```bash
python3 paper_tool_pro.py analyze \
  --papers paper1.txt paper2.txt \
  --framework methodology \
  --output methods.md
```

### Comparative/Debate-Style Analysis

```bash
python3 paper_tool_pro.py analyze \
  --papers paper1.txt paper2.txt \
  --framework comparative \
  --output debate.md
```

---

## Phase 6: Research Gap Detection

**Scenario:** You're looking for a thesis topic and want to know what's missing in the existing literature.

```bash
python3 paper_tool_pro.py gaps \
  --papers paper1.txt paper2.txt paper3.txt \
  --domain "natural language processing" \
  --output gaps.md
```

The output will include:
- Methodological gaps
- Theoretical gaps
- Contextual gaps
- Priority ranking
- **Concrete, actionable research question suggestions** ← incredibly useful for finding a thesis topic

---

## Phase 7: Research Narrative (The Killer Feature)

**Scenario:** You've read several papers but have no idea how to connect them into a coherent research story.

### You Have a Research Direction → Full Analysis + Chapter 1 Outline

```bash
python3 paper_tool_pro.py narrative \
  --papers paper1.txt paper2.txt paper3.txt \
  --my-topic "transformers for low-resource language processing" \
  --output storyline.md
```

The output has 4 parts:
1. **Research evolution timeline** — What A discovered → What B built on → What C challenged
2. **Paper relationship map** — Uses symbols: `→extends` `⟳revises` `✗challenges` `∥parallel` `◎foundational`
3. **Convergence toward your research** — Where you stand, who supports your work, where the gap is, why it matters
4. **Chapter 1 skeleton** — A ready-to-fill framework for your thesis introduction

### No Direction Yet → 3 Research Entry Points Suggested

```bash
python3 paper_tool_pro.py narrative \
  --papers paper1.txt paper2.txt paper3.txt \
  --output explore.md
```

Leave out `--my-topic` and the AI will analyze the landscape and suggest 3 viable research entry points (with notes on whether each is more suitable for a Master's or PhD thesis).

> 💡 This feature requires at least **2 papers**. A single paper can't establish "evolution" or "relationships".

---

## Phase 8: Writing Templates

View the available writing templates:

```bash
python3 paper_tool_pro.py templates
```

These templates follow standard academic writing structures, which you can use as a reference to organize your own paper:
- **literature_review** — Traditional literature review
- **systematic** — PRISMA systematic review
- **argumentative** — Argumentative essay (thesis-antithesis-synthesis)

---

## Frequently Asked Questions

### Q: How long does one analysis take?

It depends on paper length and the AI model — usually 30 seconds to 2 minutes. Ollama (local) will be slower than cloud-based options.

### Q: How much does it cost?

- **Gemini**: Free tier available — students can use it daily at essentially zero cost
- **Ollama**: Completely free (runs on your machine)
- **OpenAI (gpt-4o-mini)**: About $0.01 per analysis
- **Claude**: About $0.01 per analysis

Even with OpenAI or Claude, 100 analyses cost less than $1. With Gemini's free tier, it's zero cost.

### Q: Why is there a 3-paper limit?

This is a limitation of the free version. If you need to analyze more papers at once (up to 50), plus features like automatic Notion organization and paper relationship graphs, consider upgrading to the [Pro version](https://judyailab.com/products).

### Q: Can it analyze papers written in languages other than English?

Yes! The AI automatically detects the language, and both English and non-English papers are supported. You can even mix papers in different languages within the same analysis.

### Q: Can I submit AI-generated content directly?

**No.** The AI produces a first draft and a structural framework. You must:
1. Carefully verify the accuracy of the content
2. Add your own perspective and analysis
3. Check citation formatting
4. Rewrite it in your own voice

Submitting AI-generated content directly may violate your institution's academic integrity policy.

### Q: Does it support PDF files?

Currently, the tool supports `.txt` and `.md` text files, as well as URLs. If your paper is a PDF, please copy the text content into a `.txt` file first.

---

## Quick Command Reference

| Command | What It Does |
|---------|--------------|
| `python3 paper_tool_pro.py config` | Check your AI configuration |
| `python3 paper_tool_pro.py alphaxiv <ID>` | Look up a paper by arXiv ID |
| `python3 paper_tool_pro.py synthesize --papers ... --topic ...` | Generate a literature review |
| `python3 paper_tool_pro.py analyze --papers ... --framework ...` | Run a critical analysis |
| `python3 paper_tool_pro.py gaps --papers ... --domain ...` | Detect research gaps |
| `python3 paper_tool_pro.py narrative --papers ... --my-topic ...` | Build a research narrative |
| `python3 paper_tool_pro.py templates` | View writing templates |

---

*Made with ❤️ by [Judy AI Lab](https://judyailab.com)*

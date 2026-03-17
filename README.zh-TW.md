# Paper Research Tool

[![CI](https://github.com/JudyaiLab/paper-research-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/JudyaiLab/paper-research-tool/actions) [![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> AI 驅動的學術論文研究助手

**語言: [English](README.md) | 繁體中文 | [한국어](README.ko.md)**

## 功能

上傳 PDF（或貼 arXiv 網址），選 AI 引擎，就能得到結構化的學術摘要。

- **4 家 AI** — OpenAI、Anthropic、Google Gemini、OpenRouter
- **3 種語言** — 繁體中文、English、한국어
- **批次模式** — 一次摘要多篇論文
- **arXiv 支援** — 貼 arXiv 網址就能分析
- **Web UI** — 瀏覽器介面，不用打指令
- **知識庫** — 分析過的論文存在本機，可搜尋
- **CLI + Web** — 終端或瀏覽器都能用

## 快速開始

### 安裝

```bash
pip install paper-research-tool
```

或從原始碼：

```bash
git clone https://github.com/JudyaiLab/paper-research-tool.git
cd paper-research-tool
pip install -r requirements.txt
```

### 設定 API Key

推薦用 Google Gemini（有免費額度）：

```bash
python paper_tool.py config --google-key 你的KEY
```

其他選擇：

```bash
python paper_tool.py config --openai-key 你的KEY
python paper_tool.py config --anthropic-key 你的KEY
python paper_tool.py config --openrouter-key 你的KEY
```

### 使用

```bash
# 摘要一篇論文
python paper_tool.py summarize paper.pdf

# 批次摘要多篇
python paper_tool.py summarize paper1.pdf paper2.pdf paper3.pdf

# 用 arXiv 網址
python paper_tool.py summarize https://arxiv.org/abs/2301.00001

# 開啟 Web UI
python paper_tool.py serve

# 搜尋知識庫
python paper_tool.py search "transformer"

# 切換語言
python paper_tool.py --lang zh-TW summarize paper.pdf
```

### Web UI

安裝 Gradio 後就能用瀏覽器介面：

```bash
pip install gradio
python paper_tool.py serve
```

打開 http://127.0.0.1:7860，上傳 PDF，選 AI 和語言，就能看結果。

## 所有指令

| 指令 | 功能 |
|------|------|
| `summarize` | AI 摘要（PDF、arXiv URL，支援批次） |
| `add` | 加入論文到知識庫 |
| `list` | 列出所有論文 |
| `search` | 搜尋知識庫 |
| `relate` | 分析兩篇論文的關聯 |
| `config` | 檢視/設定 API Key |
| `serve` | 開啟 Web UI |

## 免費版 vs Pro

| 功能 | 免費 | [Pro](https://judyailab.gumroad.com) |
|------|:----:|:----:|
| AI 引擎 | 4 家 | 4 家 |
| 每次論文數 | 批次 | 批次（最多 50） |
| 每篇字元數 | 25K | 50K |
| 分析框架 | 3 種 | 5 種 |
| 文獻綜述 | 基本 | 進階（自動分批） |
| Web UI | 有 | 有（全功能） |
| arXiv 支援 | 有 | 有 |
| 主題分群 | — | 有 |
| 跨論文辯論 | — | 有 |
| 引用關係圖 | — | 有 |
| 研究缺口偵測 | — | 有 |
| Notion 同步 | — | 有（4 個資料庫） |
| 知識庫 | 有 | — |
| 語言 | 3 種 | 3 種 |

## Windows 使用者

如果遇到 `UnicodeEncodeError: 'cp949'` 或 `'cp1252'` 等編碼錯誤，請在執行前設定環境變數：

```bash
set PYTHONUTF8=1
```

或在 PowerShell 中：

```powershell
$env:PYTHONUTF8 = "1"
python paper_tool.py summarize paper.pdf
```

## 測試

```bash
pip install pytest
pytest tests/ -v
```

## 授權

MIT — 見 [LICENSE](LICENSE)

---

Made with care by [Judy AI Lab](https://judyailab.com)

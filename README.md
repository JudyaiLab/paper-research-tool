# 📄 Paper Research Tool — AI 論文分析工具

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**🌐 Language: [English](README.en.md) | 繁體中文 | [한국어](README.ko.md)**

**免費開源 AI 學術論文分析工具** — 一行指令搞定文獻綜述、批判分析、研究缺口偵測、研究敘事脈絡。

專為碩博士生與研究人員設計。支援 Gemini（有免費額度）/ OpenAI / Claude / Ollama（本地免費）。

---

## 功能總覽

| 功能 | 說明 | 限制 |
|------|------|------|
| 📚 文獻綜述 | AI 自動生成主題式文獻綜述 | 一次最多 3 篇 |
| 🔍 批判分析 | 優缺點 / 方法論 / 辯論式分析 | 一次最多 3 篇 |
| 🔬 研究缺口 | 偵測研究空白、生成研究問題 | 一次最多 3 篇 |
| 🧭 研究敘事 | 串論文脈絡、生成論文第一章骨架 | 2-3 篇 |
| 📄 論文查詢 | AlphaXiv / arXiv 論文速查 | 無限制 |
| 🌐 多語言輸出 | 支援繁中 / English / 한국어 輸出 | 所有分析指令 |
| 📝 寫作模板 | 文獻綜述、PRISMA、議論文模板 | 無限制 |

> **想要更多？** [Pro 版](https://judyailab.com/products) 支援一次 50 篇、自動分群、Notion 整合、論文關聯圖。

---

## 快速開始

### 1. 安裝

```bash
git clone https://github.com/JudyAILab/paper-research-tool.git
cd paper-research-tool
pip install -r requirements.txt
```

### 2. 設定 AI（四選一）

**零成本入門 — Google Gemini（推薦新手）：**
```bash
# 去 https://aistudio.google.com/apikey 免費申請
export GEMINI_API_KEY=你的key
```

**品質最佳 — OpenAI / Claude：**
```bash
export OPENAI_API_KEY=sk-你的key
# 或
export ANTHROPIC_API_KEY=sk-ant-你的key
```

**完全離線 — Ollama 本地模型：**
```bash
# 安裝 Ollama（https://ollama.ai）
ollama pull llama3
ollama serve
```

**進階 — config.yaml：**
```bash
cp config.example.yaml config.yaml
# 編輯 config.yaml 填入 API Key
```

### 3. 使用

```bash
# 檢查 AI 設定是否正確
python paper_tool_pro.py config

# 查詢論文（不需要 AI）
python paper_tool_pro.py alphaxiv 2401.12345

# 文獻綜述（預設繁體中文輸出）
python paper_tool_pro.py synthesize --papers paper1.txt paper2.txt --topic "機器學習在教育上的應用"

# 文獻綜述（英文輸出）
python paper_tool_pro.py synthesize --papers paper1.txt paper2.txt --topic "ML in education" --lang en

# 批判性分析
python paper_tool_pro.py analyze --papers paper1.txt --framework strengths-weaknesses

# 研究缺口偵測
python paper_tool_pro.py gaps --papers paper1.txt paper2.txt --domain "自然語言處理"

# 研究敘事脈絡（有方向 → 完整分析 + 論文第一章骨架）
python paper_tool_pro.py narrative --papers paper1.txt paper2.txt --my-topic "transformer 在低資源語言的應用"

# 研究敘事脈絡（沒方向 → 3 個研究切入點建議）
python paper_tool_pro.py narrative --papers paper1.txt paper2.txt

# 查看寫作模板
python paper_tool_pro.py templates
```

> 📖 完整教學請看 [使用教學（繁中）](docs/GETTING_STARTED.md) | [English Guide](docs/GETTING_STARTED.en.md) | [한국어 가이드](docs/GETTING_STARTED.ko.md)

---

## 支援的 AI 模型

| 服務 | 模型 | 費用 | 優勢 |
|------|------|------|------|
| **Google Gemini** | gemini-2.0-flash | 有免費額度 | 零成本入門，速度最快 |
| **OpenAI** | gpt-4o-mini | ~$0.01/次 | 品質最穩定，學術寫作首選 |
| **Anthropic** | Claude Sonnet | ~$0.01/次 | 長文分析、深度推理最強 |
| **Ollama** | llama3 等 | 完全免費 | 離線可用，資料不上傳 |

---

## 檔案結構

```
paper-research-tool/
├── paper_tool_pro.py          # 主程式（CLI 入口）
├── core/
│   ├── config.py              # 設定管理
│   ├── ai_caller.py           # AI 統一呼叫介面
│   ├── ai_synthesizer.py      # 文獻綜述生成
│   ├── critical_analyzer.py   # 批判性分析
│   ├── gap_detector.py        # 研究缺口偵測
│   ├── narrative_builder.py   # 研究敘事脈絡分析
│   └── paper_reader.py        # 論文讀取共用模組
├── templates/
│   └── writing_templates.py   # 寫作模板
├── config.example.yaml        # 設定檔範本
├── requirements.txt
└── docs/                      # 教學文件（中/英/韓）
```

---

## 免費版 vs Pro 版

| | 免費版（本工具） | Pro 版 |
|---|---|---|
| 每次分析論文數 | 最多 3 篇 | 最多 50 篇 |
| AI 分析 | ✅ 基礎分析 | ✅ 進階深度分析 |
| 研究敘事脈絡 | ✅ | ✅ 含自動關聯圖 |
| 輸出格式 | Markdown 檔案 | Notion 自動整理 |
| 論文分群 | ❌ | ✅ 自動按主題分群 |
| 論文關聯圖 | ❌ | ✅ 視覺化關聯 |
| 新論文監控 | ❌ | ✅ 自動追蹤 |
| 價格 | 免費 | [查看定價](https://judyailab.com/products) |

---

## 使用場景

- **碩博士生** — 論文寫作的文獻綜述章節、研究脈絡建構
- **研究人員** — 快速掌握領域研究現況與缺口
- **教授** — 評估學生論文或審稿
- **學術寫作者** — 結構化寫作參考

---

## License

MIT — 免費使用、修改、分享。

*Made with ❤️ by [Judy AI Lab](https://judyailab.com)*

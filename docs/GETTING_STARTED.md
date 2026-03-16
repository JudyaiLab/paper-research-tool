# Paper Research Tool — 超白話使用教學

> 這份教學是寫給「從來沒用過命令列」的碩博士生。
> 每一步都有截圖等級的詳細說明，照著做就會動。

Other languages: [English](GETTING_STARTED.en.md) | [한국어](GETTING_STARTED.ko.md)

---

## Phase 0：安裝環境（第一次才需要）

### 0-1. 確認電腦有 Python

打開終端機（Terminal）：
- **Mac**：按 `Cmd + 空白鍵`，輸入 `Terminal`，按 Enter
- **Windows**：按 `Win + R`，輸入 `cmd`，按 Enter

輸入以下指令確認 Python 版本：

```bash
python3 --version
```

看到 `Python 3.10.x`（或更高）就 OK。

**沒有 Python？** → 去 https://www.python.org/downloads/ 下載安裝，安裝時記得勾選「Add Python to PATH」。

### 0-2. 下載工具

```bash
git clone https://github.com/JudyAILab/paper-research-tool.git
cd paper-research-tool
```

**沒有 git？** 直接去 GitHub 頁面點「Code → Download ZIP」，解壓縮後 `cd` 進去。

### 0-3. 安裝相依套件

```bash
pip install -r requirements.txt
```

這會安裝 `pyyaml`（讀取設定檔用的）。裝好後就不用再裝了。

---

## Phase 1：設定 AI 模型（重要！四選一）

工具本身不包含 AI，你需要告訴它「用哪個 AI」來分析論文。有四個選項：

### 各家 AI 比較表

先看一下差異，再決定用哪個：

| 模型 | 費用 | 品質 | 速度 | 適合誰 |
|------|------|------|------|--------|
| **Google Gemini** | 有免費額度（每分鐘 15 次） | ★★★★ | 最快 | 學生首選，零成本入門 |
| **OpenAI GPT-4o-mini** | 每次約 $0.01（需儲值） | ★★★★★ | 快 | 品質最穩定，學術寫作首選 |
| **Anthropic Claude** | 每次約 $0.01（需儲值） | ★★★★★ | 快 | 長文分析與細膩推理最強 |
| **Ollama 本地** | 完全免費 | ★★★ | 看電腦 | 資料不能上傳雲端的人 |

### 選項 A：Google Gemini（新手最推薦，有免費額度）

1. 去 https://aistudio.google.com/apikey 用 Google 帳號登入
2. 點「Create API Key」，複製那串 Key
3. 在終端機輸入：

```bash
export GEMINI_API_KEY=你複製的那串key
```

> 💡 **為什麼推薦？** Gemini 有每日免費額度（免費方案每分鐘 15 次請求），對學生來說基本上不用花錢。用 Google 帳號就能申請，30 秒搞定。
>
> 推薦模型：`gemini-2.0-flash`（預設，免費且快），想要更強可以在 config.yaml 改成 `gemini-2.5-pro`。

### 選項 B：OpenAI（品質最穩定）

1. 去 https://platform.openai.com/api-keys 註冊帳號
2. 點「Create new secret key」，複製那串 `sk-` 開頭的 Key
3. 在終端機輸入：

```bash
export OPENAI_API_KEY=sk-你複製的那串key
```

> 💡 需要先儲值（最低 $5 美金）。用 gpt-4o-mini 模型，分析一次大約 $0.01 美金（台幣不到 1 元），分析 500 次都花不到 $5。
>
> **優勢**：學術寫作品質最穩定，結構完整，引用格式準確。

### 選項 C：Anthropic Claude（深度分析最強）

1. 去 https://console.anthropic.com/ 註冊帳號
2. 建立 API Key
3. 在終端機輸入：

```bash
export ANTHROPIC_API_KEY=sk-ant-你的key
```

> 💡 需要先儲值。每次分析約 $0.01 美金。
>
> **優勢**：處理長篇論文最強，邏輯推理和批判性分析比其他模型更細膩。如果你的論文超長或需要深度比較，選 Claude。

### 選項 D：Ollama 本地模型（完全免費，不用網路）

1. 去 https://ollama.ai 下載 Ollama
2. 安裝完後，在終端機：

```bash
ollama pull llama3     # 下載模型（約 4GB，只需一次）
ollama serve           # 啟動模型
```

3. 完成！不需要任何 API Key，AI 直接在你電腦上跑。

> ⚠️ 需要至少 8GB RAM。分析品質比雲端 AI 稍差，但完全免費、不需網路、論文資料完全不離開你的電腦。
>
> **適合**：研究資料有保密需求、或是網路不穩定的環境。

### 確認 AI 設定成功

```bash
python3 paper_tool_pro.py config
```

會顯示類似：

```
⚙️  Paper Research Tool — AI Configuration

  Current AI model: OpenAI (gpt-4o-mini) — API Key: ✓
```

如果看到 `✗ Not configured` 或 `No AI model detected`，回去檢查上面的步驟。

---

## Phase 2：查論文（不需要 AI 就能用）

用 AlphaXiv 快速查論文摘要，不需要 AI Key：

```bash
python3 paper_tool_pro.py alphaxiv 2401.12345
```

把 `2401.12345` 換成你要查的 arXiv 論文 ID。

論文 ID 在哪？打開任何 arXiv 論文頁面，網址裡的數字就是：
`https://arxiv.org/abs/2401.12345` → ID 就是 `2401.12345`

想看更詳細的摘要：

```bash
python3 paper_tool_pro.py alphaxiv 2401.12345 --full
```

---

## Phase 3：準備你的論文檔案

把你要分析的論文存成 `.txt` 或 `.md` 文字檔。

**最簡單的做法：**
1. 打開論文 PDF
2. 全選（Ctrl+A）→ 複製（Ctrl+C）
3. 打開記事本 → 貼上（Ctrl+V）→ 存檔為 `paper1.txt`

**也可以直接用網址：**
```bash
# 直接用 arXiv 的網址也可以
python3 paper_tool_pro.py synthesize \
  --papers https://arxiv.org/abs/2401.12345 \
  --topic "你的研究主題"
```

> ⚠️ 免費版一次最多分析 **3 篇** 論文。超過 3 篇會自動只取前 3 篇。

---

## Phase 4：文獻綜述（最常用）

**情境：** 你的指導教授要你寫文獻綜述，你讀了 3 篇相關論文。

```bash
python3 paper_tool_pro.py synthesize \
  --papers paper1.txt paper2.txt paper3.txt \
  --topic "深度學習在醫學影像診斷的應用" \
  --output my_review.md
```

工具會：
1. ✓ 讀取你的 3 篇論文
2. 🤖 用 AI 分析並整合
3. ✅ 生成一份完整的文獻綜述，存到 `my_review.md`

打開 `my_review.md` 看結果。裡面會有：
- 引言（研究背景）
- 主題分析（按主題整理，不是逐篇摘要）
- 綜合比較（論文之間的共識與矛盾）
- 研究缺口
- 結論與未來方向

> 💡 這只是 **初稿**，你還是要自己修改和補充。AI 幫你建立骨架，血肉要自己填。

---

## Phase 5：批判性分析

**情境：** 教授要你「批判性地評價」這幾篇論文。

### 優缺點分析（預設）

```bash
python3 paper_tool_pro.py analyze \
  --papers paper1.txt paper2.txt \
  --output critique.md
```

### 方法論比較

```bash
python3 paper_tool_pro.py analyze \
  --papers paper1.txt paper2.txt \
  --framework methodology \
  --output methods.md
```

### 辯論式分析

```bash
python3 paper_tool_pro.py analyze \
  --papers paper1.txt paper2.txt \
  --framework comparative \
  --output debate.md
```

---

## Phase 6：研究缺口偵測

**情境：** 你正在找論文題目，想知道現有研究還缺什麼。

```bash
python3 paper_tool_pro.py gaps \
  --papers paper1.txt paper2.txt paper3.txt \
  --domain "自然語言處理" \
  --output gaps.md
```

結果會包含：
- 方法論缺口
- 理論缺口
- 情境缺口
- 優先序排名
- **具體可操作的研究問題建議** ← 對找論文題目超有用

---

## Phase 7：研究敘事脈絡（殺手功能）

**情境：** 你讀了幾篇論文，但不知道怎麼把它們「串成一條線」變成自己的研究脈絡。

### 已有研究方向 → 完整分析 + 第一章骨架

```bash
python3 paper_tool_pro.py narrative \
  --papers paper1.txt paper2.txt paper3.txt \
  --my-topic "transformer 在低資源語言的應用" \
  --output storyline.md
```

結果包含 4 個部分：
1. **研究演進時間軸** — A 發現了什麼 → B 延伸了什麼 → C 挑戰了什麼
2. **論文關係地圖** — 用符號 `→延伸` `⟳修正` `✗挑戰` `∥平行` `◎奠基` 標示關係
3. **向你的研究收斂** — 你站在哪裡、誰支撐你、缺口在哪、為什麼重要
4. **論文第一章骨架** — 可直接填肉的 Chapter 1 框架

### 還沒有方向 → 3 個研究切入點建議

```bash
python3 paper_tool_pro.py narrative \
  --papers paper1.txt paper2.txt paper3.txt \
  --output explore.md
```

不加 `--my-topic`，AI 會幫你分析脈絡後，建議 3 個可行的研究切入點（包含適合碩士還是博士）。

> 💡 這個功能至少需要 **2 篇** 論文。1 篇論文無法建立「演進」和「關係」。

---

## Phase 8：寫作模板

查看可用的寫作模板：

```bash
python3 paper_tool_pro.py templates
```

這些模板是學術寫作的標準結構，你可以參考來組織自己的論文：
- **literature_review** — 傳統文獻綜述
- **systematic** — PRISMA 系統性回顧
- **argumentative** — 議論文（正反合）

---

## 常見問題

### Q：分析一次要多久？

取決於論文長度和 AI 模型，通常 30 秒 ~ 2 分鐘。Ollama 本地模型會比雲端慢。

### Q：費用多少？

- **Gemini**：有免費額度，學生日常使用基本不花錢
- **Ollama**：完全免費（在你的電腦上跑）
- **OpenAI (gpt-4o-mini)**：每次分析約 $0.01 美金
- **Claude**：每次分析約 $0.01 美金

用 OpenAI 或 Claude 分析 100 次也不到 $1 美金。用 Gemini 免費額度更是零成本。

### Q：為什麼只能分析 3 篇？

這是免費版的限制。如果你需要一次分析更多論文（最多 50 篇）+ Notion 自動整理 + 論文關聯圖等進階功能，可以升級 [Pro 版](https://judyailab.com/products)。

### Q：輸出語言可以選嗎？

可以！所有分析指令都支援 `--lang` 參數：

```bash
# 繁體中文輸出（預設）
python3 paper_tool_pro.py synthesize --papers paper1.txt --topic "NLP" --lang zh-TW

# 英文輸出
python3 paper_tool_pro.py synthesize --papers paper1.txt --topic "NLP" --lang en

# 韓文輸出
python3 paper_tool_pro.py synthesize --papers paper1.txt --topic "NLP" --lang ko
```

支援的語言：`zh-TW`（繁體中文，預設）、`en`（English）、`ko`（한국어）。

### Q：可以分析中文論文嗎？

可以！AI 會自動偵測論文語言，中英文論文都支援。你也可以混合語言的論文一起分析。輸出語言由 `--lang` 決定，跟論文本身的語言無關。

### Q：AI 生成的內容可以直接交嗎？

**不行。** AI 生成的是初稿和框架，你必須：
1. 仔細檢查內容正確性
2. 補充自己的觀點和分析
3. 確認引用格式
4. 用自己的語言改寫

直接交 AI 生成的內容可能違反學術誠信規範。

### Q：支援 PDF 嗎？

目前支援 `.txt` 和 `.md` 文字檔，以及網址。如果你的論文是 PDF，請先複製文字內容到 txt 檔。

---

## 指令速查表

| 指令 | 功能 |
|------|------|
| `python3 paper_tool_pro.py config` | 檢查 AI 設定 |
| `python3 paper_tool_pro.py alphaxiv <ID>` | 查論文 |
| `python3 paper_tool_pro.py synthesize --papers ... --topic ...` | 文獻綜述 |
| `python3 paper_tool_pro.py analyze --papers ... --framework ...` | 批判分析 |
| `python3 paper_tool_pro.py gaps --papers ... --domain ...` | 研究缺口 |
| `python3 paper_tool_pro.py narrative --papers ... --my-topic ...` | 研究敘事脈絡 |
| `python3 paper_tool_pro.py templates` | 寫作模板 |

---

*Made with ❤️ by [Judy AI Lab](https://judyailab.com)*

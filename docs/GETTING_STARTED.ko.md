# Paper Research Tool — 완전 초보자를 위한 사용 가이드

> 이 가이드는 **명령줄(터미널)을 한 번도 써본 적 없는** 석·박사생을 위해 작성되었습니다.
> 한 단계씩 따라오시면 누구든 반드시 작동시킬 수 있습니다.

다른 언어 버전: [繁體中文](GETTING_STARTED.md) | [English](GETTING_STARTED.en.md)

---

## Phase 0：환경 설치（처음 한 번만）

### 0-1. Python이 설치되어 있는지 확인하기

터미널(Terminal)을 열어주세요：
- **Mac**：`Cmd + 스페이스바`를 누르고 `Terminal`을 입력한 후 Enter
- **Windows**：`Win + R`을 누르고 `cmd`를 입력한 후 Enter

아래 명령어를 입력하여 Python 버전을 확인합니다：

```bash
python3 --version
```

`Python 3.10.x`（또는 그 이상）가 표시되면 OK입니다.

**Python이 없다면？** → https://www.python.org/downloads/ 에서 다운로드하여 설치하세요. 설치 시 반드시 「Add Python to PATH」에 체크해야 합니다.

### 0-2. 도구 다운로드

```bash
git clone https://github.com/JudyAILab/paper-research-tool.git
cd paper-research-tool
```

**git이 없다면？** GitHub 페이지에서 「Code → Download ZIP」을 클릭하여 다운로드한 뒤, 압축을 풀고 `cd` 명령어로 해당 폴더로 이동하면 됩니다.

### 0-3. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

이 명령어는 `pyyaml`（설정 파일 읽기에 필요）을 설치합니다. 한 번만 설치하면 이후에는 다시 설치하지 않아도 됩니다.

---

## Phase 1：AI 모델 설정（중요！네 가지 옵션 중 하나 선택）

이 도구에는 AI가 내장되어 있지 않습니다. 논문을 분석할 때 「어떤 AI를 사용할지」를 먼저 설정해야 합니다. 아래 네 가지 옵션 중 하나를 선택하세요.

### AI 모델 비교표

선택하기 전에 각 모델의 차이점을 확인해보세요：

| 모델 | 비용 | 품질 | 속도 | 추천 대상 |
|------|------|------|------|-----------|
| **Google Gemini** | 무료 한도 있음（분당 15회） | ★★★★ | 가장 빠름 | 학생 첫 선택, 비용 없이 시작 |
| **OpenAI GPT-4o-mini** | 1회 약 $0.01（사전 충전 필요） | ★★★★★ | 빠름 | 품질이 가장 안정적, 학술 글쓰기에 최적 |
| **Anthropic Claude** | 1회 약 $0.01（사전 충전 필요） | ★★★★★ | 빠름 | 장문 분석과 심층 추론에 탁월 |
| **Ollama 로컬 모델** | 완전 무료 | ★★★ | PC 사양에 따라 다름 | 데이터를 외부로 올릴 수 없는 분 |

### 옵션 A：Google Gemini（초보자에게 가장 추천, 무료 한도 제공）

1. https://aistudio.google.com/apikey 에서 Google 계정으로 로그인
2. 「Create API Key」를 클릭하고, 생성된 키를 복사
3. 터미널에 아래와 같이 입력：

```bash
export GEMINI_API_KEY=복사한키를여기에붙여넣기
```

> 💡 **왜 추천하나요？** Gemini는 매일 무료 한도를 제공합니다（무료 플랜 기준 분당 15회 요청）. 학생이라면 사실상 비용이 들지 않습니다. Google 계정만 있으면 30초 만에 설정할 수 있습니다.
>
> 추천 모델：`gemini-2.0-flash`（기본값, 무료이면서 빠름）. 더 높은 품질을 원한다면 config.yaml에서 `gemini-2.5-pro`로 변경할 수 있습니다.

### 옵션 B：OpenAI（가장 안정적인 품질）

1. https://platform.openai.com/api-keys 에서 계정 등록
2. 「Create new secret key」를 클릭하고, `sk-`로 시작하는 키를 복사
3. 터미널에 아래와 같이 입력：

```bash
export OPENAI_API_KEY=sk-복사한키를여기에붙여넣기
```

> 💡 사전 충전이 필요합니다（최소 $5）. gpt-4o-mini 모델 기준으로 1회 분석에 약 $0.01로, $5만 충전해도 500회 이상 분석할 수 있습니다.
>
> **장점**：학술 글쓰기에서 품질이 가장 안정적이며, 구조가 완성도 높고 인용 형식도 정확합니다.

### 옵션 C：Anthropic Claude（심층 분석에 강점）

1. https://console.anthropic.com/ 에서 계정 등록
2. API 키 생성
3. 터미널에 아래와 같이 입력：

```bash
export ANTHROPIC_API_KEY=sk-ant-여기에키붙여넣기
```

> 💡 사전 충전이 필요합니다. 1회 분석에 약 $0.01입니다.
>
> **장점**：분량이 긴 논문 처리에 특히 강하며, 논리적 추론과 비판적 분석이 다른 모델보다 세밀합니다. 논문이 매우 길거나 깊이 있는 비교 분석이 필요하다면 Claude를 선택하세요.

### 옵션 D：Ollama 로컬 모델（완전 무료, 인터넷 불필요）

1. https://ollama.ai 에서 Ollama를 다운로드하여 설치
2. 설치 후 터미널에서：

```bash
ollama pull llama3     # 모델 다운로드（약 4GB, 최초 1회만）
ollama serve           # 모델 실행
```

3. 완료！API 키가 전혀 필요 없으며, AI가 내 컴퓨터에서 직접 실행됩니다.

> ⚠️ 최소 8GB RAM이 필요합니다. 클라우드 AI보다 분석 품질은 다소 낮지만, 완전 무료이고 인터넷이 필요 없으며 논문 데이터가 외부로 전혀 전송되지 않습니다.
>
> **적합한 경우**：연구 데이터의 기밀 유지가 필요하거나 인터넷 연결이 불안정한 환경.

### AI 설정이 잘 되었는지 확인하기

```bash
python3 paper_tool_pro.py config
```

아래와 비슷한 내용이 표시됩니다：

```
⚙️  Paper Research Tool — AI Configuration

  Current AI model: OpenAI (gpt-4o-mini) — API Key: ✓
```

`✗ Not configured` 또는 `No AI model detected`라고 나오면 위의 단계를 다시 확인해보세요.

---

## Phase 2：논문 검색（AI 없이도 사용 가능）

AlphaXiv를 통해 논문 초록을 빠르게 검색할 수 있습니다. AI 키 없이도 사용 가능합니다：

```bash
python3 paper_tool_pro.py alphaxiv 2401.12345
```

`2401.12345` 부분을 검색하고 싶은 arXiv 논문 ID로 교체하세요.

논문 ID는 어디서 찾나요？ 아무 arXiv 논문 페이지를 열면 주소창에서 확인할 수 있습니다：
`https://arxiv.org/abs/2401.12345` → ID는 `2401.12345`

더 자세한 초록을 보고 싶다면：

```bash
python3 paper_tool_pro.py alphaxiv 2401.12345 --full
```

---

## Phase 3：논문 파일 준비하기

분석할 논문을 `.txt` 또는 `.md` 텍스트 파일로 저장합니다.

**가장 간단한 방법：**
1. 논문 PDF를 열기
2. 전체 선택（Ctrl+A）→ 복사（Ctrl+C）
3. 메모장을 열고 → 붙여넣기（Ctrl+V）→ `paper1.txt`로 저장

**URL을 직접 사용하는 방법：**
```bash
# arXiv URL을 직접 사용할 수도 있습니다
python3 paper_tool_pro.py synthesize \
  --papers https://arxiv.org/abs/2401.12345 \
  --topic "연구 주제를 여기에 입력"
```

> ⚠️ 무료 버전에서는 한 번에 최대 **3편**의 논문만 분석할 수 있습니다. 3편을 초과하면 앞의 3편만 자동으로 분석됩니다.

---

## Phase 4：문헌 검토（가장 자주 사용되는 기능）

**상황：** 지도교수님께서 문헌 검토(literature review)를 작성하라고 하셨고, 관련 논문 3편을 읽었습니다.

```bash
python3 paper_tool_pro.py synthesize \
  --papers paper1.txt paper2.txt paper3.txt \
  --topic "교육에서의 머신러닝" \
  --output my_review.md
```

도구가 수행하는 작업：
1. ✓ 3편의 논문을 읽어들임
2. 🤖 AI로 분석하고 통합
3. ✅ 완성된 문헌 검토를 `my_review.md`로 저장

`my_review.md`를 열어 결과를 확인해보세요. 다음 내용이 포함됩니다：
- 서론（연구 배경）
- 주제별 분석（논문을 하나씩 요약하는 것이 아니라 주제 중심으로 정리）
- 종합 비교（논문 간의 공통점과 차이점）
- 연구 공백
- 결론 및 향후 방향

> 💡 이 결과는 어디까지나 **초고**입니다. 내용을 직접 검토하고 보완해야 합니다. AI가 뼈대를 만들어주면, 살을 붙이는 것은 여러분의 몫입니다.

---

## Phase 5：비판적 분석

**상황：** 교수님께서 논문들을 「비판적으로 평가」하라고 하셨습니다.

### 장단점 분석（기본값）

```bash
python3 paper_tool_pro.py analyze \
  --papers paper1.txt paper2.txt \
  --output critique.md
```

### 연구 방법론 비교

```bash
python3 paper_tool_pro.py analyze \
  --papers paper1.txt paper2.txt \
  --framework methodology \
  --output methods.md
```

### 논쟁적 비교 분석

```bash
python3 paper_tool_pro.py analyze \
  --papers paper1.txt paper2.txt \
  --framework comparative \
  --output debate.md
```

---

## Phase 6：연구 공백 탐지

**상황：** 논문 주제를 찾고 있으며, 기존 연구에서 아직 다루지 않은 부분이 무엇인지 알고 싶습니다.

```bash
python3 paper_tool_pro.py gaps \
  --papers paper1.txt paper2.txt paper3.txt \
  --domain "자연어 처리" \
  --output gaps.md
```

결과에는 다음 내용이 포함됩니다：
- 방법론적 공백
- 이론적 공백
- 맥락적 공백
- 우선순위 순위
- **구체적이고 실행 가능한 연구 문제 제안** ← 논문 주제 탐색에 매우 유용

---

## Phase 7：연구 서사 맥락화（핵심 기능）

**상황：** 논문들을 여러 편 읽었는데, 그것들을 「하나의 흐름」으로 엮어 자신의 연구 맥락으로 만드는 방법을 모르겠습니다.

### 연구 방향이 있는 경우 → 완전한 분석 + 1장 뼈대 생성

```bash
python3 paper_tool_pro.py narrative \
  --papers paper1.txt paper2.txt paper3.txt \
  --my-topic "저자원 언어에 대한 트랜스포머" \
  --output storyline.md
```

결과는 4가지 파트로 구성됩니다：
1. **연구 발전 타임라인** — A가 발견한 것 → B가 확장한 것 → C가 도전한 것
2. **논문 관계 지도** — `→확장` `⟳수정` `✗도전` `∥병렬` `◎기반` 기호로 관계 표시
3. **내 연구로의 수렴** — 나는 어디에 위치하는가, 어떤 논문이 나를 뒷받침하는가, 공백은 어디인가, 왜 중요한가
4. **논문 1장 뼈대** — 바로 내용을 채울 수 있는 Chapter 1 프레임워크

### 방향이 아직 없는 경우 → 3가지 연구 진입점 제안

```bash
python3 paper_tool_pro.py narrative \
  --papers paper1.txt paper2.txt paper3.txt \
  --output explore.md
```

`--my-topic`을 붙이지 않으면, AI가 맥락을 분석한 후 실현 가능한 연구 진입점 3가지를 제안합니다（석사 또는 박사 과정에 적합한지 여부도 포함）.

> 💡 이 기능은 최소 **2편** 이상의 논문이 필요합니다. 논문이 1편이면 「발전」과 「관계」를 파악할 수 없습니다.

---

## Phase 8：글쓰기 템플릿

사용 가능한 글쓰기 템플릿을 확인하려면：

```bash
python3 paper_tool_pro.py templates
```

이 템플릿들은 학술 글쓰기의 표준 구조입니다. 논문을 구성하는 참고 자료로 활용하세요：
- **literature_review** — 전통적인 문헌 검토
- **systematic** — PRISMA 체계적 문헌 고찰
- **argumentative** — 논증 구조（정·반·합）

---

## 자주 묻는 질문

### Q：한 번 분석하는 데 얼마나 걸리나요？

논문 길이와 AI 모델에 따라 다르지만, 보통 30초 ~ 2분입니다. Ollama 로컬 모델은 클라우드보다 속도가 느릴 수 있습니다.

### Q：비용은 얼마나 드나요？

- **Gemini**：무료 한도 제공, 학생 일상 사용이라면 사실상 비용 없음
- **Ollama**：완전 무료（내 컴퓨터에서 실행）
- **OpenAI (gpt-4o-mini)**：1회 분석에 약 $0.01
- **Claude**：1회 분석에 약 $0.01

OpenAI 또는 Claude로 100회 분석해도 $1도 들지 않습니다. Gemini 무료 한도를 활용하면 비용이 전혀 없습니다.

### Q：왜 3편만 분석할 수 있나요？

무료 버전의 제한입니다. 한 번에 더 많은 논문（최대 50편）을 분석하거나 Notion 자동 정리, 논문 관계도 등 고급 기능이 필요하다면 [Pro 버전](https://judyailab.com/products)으로 업그레이드할 수 있습니다.

### Q：한국어 논문도 분석할 수 있나요？

물론입니다！ AI가 언어를 자동으로 감지하므로 한국어, 영어 논문 모두 지원됩니다. 언어가 혼합된 논문들을 함께 분석하는 것도 가능합니다.

### Q：AI가 생성한 내용을 그대로 제출해도 되나요？

**안 됩니다.** AI가 생성한 것은 초고와 뼈대일 뿐입니다. 반드시 다음 과정을 거쳐야 합니다：
1. 내용의 정확성을 꼼꼼히 확인
2. 자신의 관점과 분석을 추가
3. 인용 형식 확인
4. 자신의 언어로 다시 작성

AI가 생성한 내용을 그대로 제출하는 것은 학문적 정직성 원칙에 위배될 수 있습니다.

### Q：PDF 파일을 지원하나요？

현재는 `.txt`와 `.md` 텍스트 파일 및 URL을 지원합니다. 논문이 PDF라면 먼저 텍스트 내용을 복사하여 txt 파일로 저장해주세요.

---

## 명령어 빠른 참조표

| 명령어 | 기능 |
|--------|------|
| `python3 paper_tool_pro.py config` | AI 설정 확인 |
| `python3 paper_tool_pro.py alphaxiv <ID>` | 논문 검색 |
| `python3 paper_tool_pro.py synthesize --papers ... --topic "교육에서의 머신러닝"` | 문헌 검토 |
| `python3 paper_tool_pro.py analyze --papers ... --framework ...` | 비판적 분석 |
| `python3 paper_tool_pro.py gaps --papers ... --domain "자연어 처리"` | 연구 공백 탐지 |
| `python3 paper_tool_pro.py narrative --papers ... --my-topic "저자원 언어에 대한 트랜스포머"` | 연구 서사 맥락화 |
| `python3 paper_tool_pro.py templates` | 글쓰기 템플릿 |

---

*Made with ❤️ by [Judy AI Lab](https://judyailab.com)*

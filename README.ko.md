# 📄 Paper Research Tool — AI 논문 분석 도구

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**🌐 Language: [English](README.en.md) | [繁體中文](README.md) | 한국어**

**무료 오픈소스 AI 학술 논문 분석 도구** — 명령어 한 줄로 문헌 리뷰, 비판적 분석, 연구 갭 탐지, 연구 서사 구축을 완료합니다.

석·박사생 및 연구자를 위해 설계. Gemini (무료 이용 가능) / OpenAI / Claude / Ollama (로컬, 무료) 지원.

---

## 기능 요약

| 기능 | 설명 | 제한 |
|------|------|------|
| 📚 문헌 리뷰 | AI 자동 주제별 문헌 리뷰 생성 | 최대 3편 |
| 🔍 비판적 분석 | 강점-약점 / 방법론 / 토론식 분석 | 최대 3편 |
| 🔬 연구 갭 | 연구 공백 탐지, 연구 질문 생성 | 최대 3편 |
| 🧭 연구 서사 | 논문 맥락 연결 + 논문 1장 골격 생성 | 2-3편 |
| 📄 논문 검색 | AlphaXiv / arXiv 빠른 검색 | 무제한 |
| 🌐 다국어 출력 | 繁中 / English / 한국어 출력 지원 | 모든 분석 명령 |
| 📝 작성 템플릿 | 문헌 리뷰, PRISMA, 논증문 템플릿 | 무제한 |

> **더 많은 기능이 필요하세요?** [Pro 버전](https://judyailab.com/products)은 최대 50편, 자동 클러스터링, Notion 연동, 인용 그래프를 지원합니다.

---

## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/JudyAILab/paper-research-tool.git
cd paper-research-tool
pip install -r requirements.txt
```

### 2. AI 설정 (택 1)

**무료 — Google Gemini (학생에게 추천):**
```bash
# https://aistudio.google.com/apikey 에서 무료 키 발급
export GEMINI_API_KEY=발급받은키
```

**최고 품질 — OpenAI / Claude:**
```bash
export OPENAI_API_KEY=sk-발급받은키
# 또는
export ANTHROPIC_API_KEY=sk-ant-발급받은키
```

**완전 오프라인 — Ollama 로컬 모델:**
```bash
# Ollama 설치 (https://ollama.ai)
ollama pull llama3
ollama serve
```

**고급 — config.yaml:**
```bash
cp config.example.yaml config.yaml
# config.yaml에 API Key 입력
```

### 3. 사용

```bash
# AI 설정 확인
python paper_tool_pro.py config

# 논문 검색 (AI 불필요)
python paper_tool_pro.py alphaxiv 2401.12345

# 문헌 리뷰 (기본: 繁體中文 출력)
python paper_tool_pro.py synthesize --papers paper1.txt paper2.txt --topic "교육에서의 머신러닝"

# 문헌 리뷰 (한국어 출력)
python paper_tool_pro.py synthesize --papers paper1.txt paper2.txt --topic "교육에서의 머신러닝" --lang ko

# 비판적 분석
python paper_tool_pro.py analyze --papers paper1.txt --framework strengths-weaknesses

# 연구 갭 탐지
python paper_tool_pro.py gaps --papers paper1.txt paper2.txt --domain "자연어 처리"

# 연구 서사 (주제 있음 → 전체 분석 + 논문 1장 골격)
python paper_tool_pro.py narrative --papers paper1.txt paper2.txt --my-topic "저자원 언어에 대한 트랜스포머"

# 연구 서사 (주제 없음 → 3개 연구 방향 제안)
python paper_tool_pro.py narrative --papers paper1.txt paper2.txt

# 작성 템플릿 보기
python paper_tool_pro.py templates
```

> 📖 상세 가이드: [한국어 가이드](docs/GETTING_STARTED.ko.md) | [English Guide](docs/GETTING_STARTED.en.md) | [繁中教學](docs/GETTING_STARTED.md)

---

## 지원 AI 모델

| 서비스 | 모델 | 비용 | 장점 |
|--------|------|------|------|
| **Google Gemini** | gemini-2.0-flash | 무료 이용 가능 | 비용 제로, 가장 빠름 |
| **OpenAI** | gpt-4o-mini | ~$0.01/회 | 학술 작성 품질 최고 |
| **Anthropic** | Claude Sonnet | ~$0.01/회 | 긴 논문 분석, 깊은 추론 |
| **Ollama** | llama3 등 | 완전 무료 | 오프라인 사용, 데이터 외부 전송 없음 |

---

## 무료 vs Pro 비교

| | 무료 (이 도구) | Pro |
|---|---|---|
| 분석 논문 수 | 최대 3편 | 최대 50편 |
| AI 분석 | ✅ 기본 분석 | ✅ 고급 심층 분석 |
| 연구 서사 | ✅ | ✅ 자동 인용 그래프 포함 |
| 출력 형식 | Markdown 파일 | Notion 자동 정리 |
| 논문 클러스터링 | ❌ | ✅ 주제별 자동 그룹화 |
| 인용 그래프 | ❌ | ✅ 시각화 연결 |
| 신규 논문 모니터링 | ❌ | ✅ 자동 추적 |
| 가격 | 무료 | [가격 보기](https://judyailab.com/products) |

---

## 사용 사례

- **석·박사생** — 문헌 리뷰 장 작성, 연구 맥락 구축
- **연구자** — 연구 현황과 갭 빠르게 파악
- **교수** — 학생 논문 평가 및 심사
- **학술 저술가** — 체계적 작성 참고

---

## License

MIT — 자유롭게 사용, 수정, 공유 가능.

*Made with ❤️ by [Judy AI Lab](https://judyailab.com)*

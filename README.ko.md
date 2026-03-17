# Paper Research Tool

[![CI](https://github.com/JudyaiLab/paper-research-tool/actions/workflows/ci.yml/badge.svg)](https://github.com/JudyaiLab/paper-research-tool/actions) [![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> AI 기반 학술 논문 연구 도우미

**언어: [English](README.md) | [繁體中文](README.zh-TW.md) | 한국어**

## 기능

PDF를 업로드하거나 arXiv URL을 붙여넣고, AI 제공자를 선택하면 구조화된 학술 요약을 받을 수 있습니다.

- **4개 AI 제공자** — OpenAI, Anthropic, Google Gemini, OpenRouter
- **3개 언어** — English, 繁體中文, 한국어
- **일괄 모드** — 여러 논문을 한 번에 요약
- **arXiv 지원** — arXiv URL로 직접 분석
- **Web UI** — 브라우저 기반 인터페이스
- **지식 기반** — 분석한 논문을 로컬에 저장하고 검색
- **CLI + Web** — 터미널 또는 브라우저에서 사용

## 빠른 시작

### 설치

```bash
pip install paper-research-tool
```

또는 소스에서:

```bash
git clone https://github.com/JudyaiLab/paper-research-tool.git
cd paper-research-tool
pip install -r requirements.txt
```

### API Key 설정

Google Gemini 추천 (무료 제공):

```bash
python paper_tool.py config --google-key YOUR_KEY
```

다른 제공자:

```bash
python paper_tool.py config --openai-key YOUR_KEY
python paper_tool.py config --anthropic-key YOUR_KEY
python paper_tool.py config --openrouter-key YOUR_KEY
```

### 사용법

```bash
# 논문 하나 요약
python paper_tool.py summarize paper.pdf

# 여러 논문 일괄 요약
python paper_tool.py summarize paper1.pdf paper2.pdf paper3.pdf

# arXiv URL로 요약
python paper_tool.py summarize https://arxiv.org/abs/2301.00001

# Web UI 실행
python paper_tool.py serve

# 지식 기반 검색
python paper_tool.py search "transformer"

# 언어 전환
python paper_tool.py --lang ko summarize paper.pdf
```

### Web UI

Gradio를 설치한 후 브라우저 인터페이스를 사용할 수 있습니다:

```bash
pip install gradio
python paper_tool.py serve
```

http://127.0.0.1:7860 을 열고 PDF를 업로드하고 AI와 언어를 선택하세요.

## 모든 명령어

| 명령어 | 기능 |
|--------|------|
| `summarize` | AI 요약 (PDF, arXiv URL, 일괄 지원) |
| `add` | 논문을 지식 기반에 추가 |
| `list` | 모든 논문 나열 |
| `search` | 지식 기반 검색 |
| `relate` | 두 논문의 관계 분석 |
| `config` | API Key 보기/설정 |
| `serve` | Web UI 실행 |

## 무료 vs Pro

| 기능 | 무료 | [Pro](https://judyailab.gumroad.com) |
|------|:----:|:----:|
| AI 제공자 | 4개 | 4개 |
| 논문 수/회 | 일괄 | 일괄 (최대 50) |
| 논문당 문자 | 25K | 50K |
| 분석 프레임워크 | 3개 | 5개 |
| 문헌 리뷰 | 기본 | 고급 (자동 배치) |
| Web UI | 있음 | 있음 (전체 기능) |
| arXiv 지원 | 있음 | 있음 |
| 주제 클러스터링 | — | 있음 |
| 논문 간 토론 | — | 있음 |
| 인용 그래프 | — | 있음 |
| 연구 갭 감지 | — | 있음 |
| Notion 동기화 | — | 있음 (4개 DB) |
| 지식 기반 | 있음 | — |
| 언어 | 3개 | 3개 |

## Windows 사용자

`UnicodeEncodeError: 'cp949'` 또는 `'cp1252'` 인코딩 오류가 발생하면 실행 전에 환경 변수를 설정하세요:

```bash
set PYTHONUTF8=1
```

또는 PowerShell에서:

```powershell
$env:PYTHONUTF8 = "1"
python paper_tool.py summarize paper.pdf
```

## 테스트

```bash
pip install pytest
pytest tests/ -v
```

## 라이선스

MIT — [LICENSE](LICENSE) 참조

---

Made with care by [Judy AI Lab](https://judyailab.com)

<p align="center">
  <img
    src="https://capsule-render.vercel.app/api?type=waving&height=300&color=0:0d9488,100:e9edf7&text=MulDetect&reversal=false&textBg=false&fontAlign=50&animation=fadeIn&fontColor=36454F&strokeWidth=0&descAlignY=55&descSize=20&desc=Multi-LLM을%20이용한%20강의%20영상%20자동%20검증%20오픈소스%20프레임워크&fontAlignY=40"
    width="100%"
  />
</p>

<p align="center"><i>멀티모달 기법과 사용자가 임의 구성한 Multi-LLM을 활용하여,<br/>강의 영상의 다양한 지식 오류를 탐지하고 상세한 피드백을 제공하는 오픈소스 프레임워크입니다.</i></p>

<p align="center">
  <img src="https://img.shields.io/badge/status-in%20development-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/python-3.12-3670A0?style=flat-square&logo=python&logoColor=ffdd54" />
  <img src="https://img.shields.io/badge/node-18%2B-339933?style=flat-square&logo=node.js&logoColor=white" />
  <img src="https://img.shields.io/badge/docker-required-2496ED?style=flat-square&logo=docker&logoColor=white" />
</p>

<br/>

## 📑 목차

- [1. 작품 개요](#-1-작품-개요)
- [2. 시스템 구조](#️-2-시스템-구조)
- [3. 기대 효과](#-3-기대-효과)
- [4. 활용 분야](#-4-활용-분야)
- [5. 프로젝트 결과](#-5-프로젝트-결과)
  - [페이지 캡쳐본](#페이지-캡쳐본)
  - [시연 영상](#시연-영상)
- [개발 가이드 (부록)](#-개발-가이드-부록)

<br/>
<br/>

## 📌 1. 작품 개요

### 1.1 문제 인식

&nbsp;&nbsp;오늘날 온라인 강의 영상은 지식을 전달하고 학습하는 주요 수단으로 자리 잡았다. 교육 분야에서도 교내 e-class와 같은 LMS는 물론 K-MOOC, KOCW와 같은 이러닝 플랫폼 등 수많은 강의 영상이 지속적으로 제공, 소비되고 있다. 하지만 강의 영상 내의 오류를 검증하지 않고 있으며, 이로 인해 학습자들은 잘못된 지식을 받아들이게 되고, 오류를 가진 지식이 전파되는 악순환 속에서 교육의 질이 하락하는 심각한 문제를 초래할 수 있다.

&nbsp;&nbsp;그러나 기존 영상 검증 시스템은 강의의 형식적 품질을 평가하거나, 가짜 뉴스·조작 영상의 진위 여부 판별에만 초점이 맞추어져 있어 강의 영상 오류 검증에는 활용할 수 없다. 한편으로 LLM에게 직접 질의하는 방식도 가능하지만, 영상을 직접 업로드하여 오류를 확인하기엔 용량의 한계가 있고, LLM이 아직 학습하지 않은 최신 정보는 반영하기 어려울 뿐만 아니라, 환각으로 인해 맞는 내용을 오류라고 잘못 판단하거나 같은 질문에 다른 답변을 내놓는 등 오류 탐지의 신뢰성을 단언하기 어렵다.

### 1.2 도전 목표

&nbsp;&nbsp;본 팀은 위 문제를 해결하기 위해 멀티모달 기법으로 영상을 심층 분석하고, 여러 LLM 모델을 교차 검증에 활용하여 강의에 내재된 5개 유형의 지식 오류를 탐지해 자세한 피드백을 제공하는 오픈소스 프레임워크, **VeriLec**을 개발하였다. 세부 목표는 다음과 같다.

1. **사용자의 자유로운 Multi-LLM 구성** — 목적이나 LLM의 발전을 반영하여, 사용자가 자유롭게 여러 LLM 모델들로 Multi-LLM을 구성할 수 있는 유연성 확보
2. **멀티모달 기법으로 영상 강의를 심층 분석해 문맥 반영** — 오류 탐지의 품질을 좌우하는 중요한 과정으로, 발화·텍스트·시각자료 등을 멀티모달로 분석
3. **Multi-LLM을 활용한 강의 내 지식 오류 탐지** — 여러 LLM의 교차 판단으로 단일 LLM이 놓치는 오류까지 폭넓게 탐지하여, 탐지율은 높이고 오탐률은 낮춤
4. **오류에 대한 판단 근거와 수정안을 포함한 자세한 피드백 제공** — 지식 오류를 5가지 유형으로 세분화하여 탐지하고, 판단 근거와 수정안 등을 함께 제공

<br/>
<br/>

## ⚙️ 2. 시스템 구조

&nbsp;&nbsp;VeriLec의 전체 시스템 구조는 아래 그림과 같다. 사용자가 LLM 조합을 설정하면, 강의 영상은 **멀티모달 강의 분석**과 **지식 오류 탐지**의 2단계로 처리된다.

<p align="center"><img src="./docs/img/system-architecture.png" alt="VeriLec 시스템 구조도" width="850"/></p>

1. **Multi-LLM 설정** — 사용자가 지식 오류 탐지에 사용할 LLM들을 등록하고 조합을 결정한다. 확정된 조합은 DB(PostgreSQL)에 저장되어, 이후 단계에서 참조되는 기준 설정으로 기능한다.
2. **멀티모달 강의 분석** — 강의 영상을 비디오와 오디오로 나누어 병렬 분석한 뒤, 강의 맥락을 연결한 하나의 멀티모달 통합 텍스트를 생성한다.
3. **지식 오류 탐지** — 사용자가 설정한 Multi-LLM을 활용해 통합 텍스트에서 강의 내 지식 오류를 탐지하고 상세 피드백을 제공한다.

### 2.1 VeriLec이 탐지하는 지식 오류 유형

| 오류 유형 | 설명 |
|---|---|
| 사실 오류 | 강의 내용이 객관적 사실과 다른 경우 |
| 오래된 내용 | 제작 이후 갱신되었거나 더 이상 유효하지 않은 정보 |
| 과도한 일반화 | 예외나 조건을 생략하거나 단정적으로 서술된 내용 |
| 혼동 가능 설명 | 문맥상 옳고 그름을 단정하기 어려우나 학습자에게 혼동을 줄 수 있는 모호한 설명 |
| 슬라이드 오류 | 오탈자, 수치·단위 오류, 코드 문법 오류, 시각적 결함 등 |

### 2.2 핵심 기능 상세

<details>
<summary><b>1) Multi-LLM 설정</b></summary>
<div markdown="1">
&nbsp;&nbsp;사용자는 지식 오류 탐지에 사용할 LLM들을 직접 등록하고 조합(셋)을 결정한다. 확정된 조합은 DB에 저장되며, 이후 전처리·탐지 전 단계에서 참조되는 기준 설정으로 기능한다. 특정 LLM에 종속되지 않는 어댑터 구조로, 목적이나 LLM의 발전에 맞춰 N개의 모델로 자유롭게 확장할 수 있다.
</div>
</details>

<details>
<summary><b>2) 멀티모달 비디오 분석</b></summary>
<div markdown="1">
&nbsp;&nbsp;인물을 마스킹 처리하고 영상 구간을 슬라이드/비디오/불명 구간으로 분류한 뒤, 프레임 변화량(MSE)·해시 유사도(pHash)·엣지 변화·히스토그램 비교를 함께 사용해 슬라이드가 바뀌는 시점을 탐지하고 대표 프레임(base)을 추출한다. 이후 동일 슬라이드 재등장 여부를 판단해 중복을 제거하고, base 프레임의 본문뿐 아니라 표·그림·도식·차트 등 시각 자료까지 텍스트로 변환한다.

<p align="center"><img src="./docs/img/slide-extraction-example.png" alt="슬라이드 텍스트 추출 예시" width="700"/></p>
</div>
</details>

<details>
<summary><b>3) 멀티모달 오디오 분석</b></summary>
<div markdown="1">
&nbsp;&nbsp;오디오를 분리해 길이·음량·침묵 구간·피치 등 음질을 분석한 뒤, Whisper로 음성을 전사한다. 이때 침묵 구간 기준으로 세그먼트를 나누어 전사함으로써 할루시네이션을 줄이고, 슬라이드 텍스트와 강의 맥락을 참고해 오인식·전문용어·띄어쓰기 오류를 교정한다. 이후 시간 간격·문장 부호·주제 연속성을 고려해 세그먼트를 의미 단위의 발화로 재구성하고, 슬라이드 정보를 기준으로 비디오·오디오 분석 결과를 하나의 통합 텍스트로 결합한다.
</div>
</details>

<details>
<summary><b>4) 지식 오류 탐지 (5단계)</b></summary>
<div markdown="1">
&nbsp;&nbsp;통합 텍스트를 입력으로, 발화 기반 탐지(5단계)와 슬라이드 기반 탐지가 병렬로 진행된다.

- **Step 1. Claim Extraction** — 사실 여부를 판단할 수 있는 문장(claim) 추출
- **Step 2. Issue Detection** — 각 LLM이 독립적으로 오류 가능성을 판단해 후보를 폭넓게 탐지
- **Step 3. Issue Classification** — 오류 후보를 5가지 유형으로 분류하고, 투표 알고리즘으로 최종 분류 결정
- **Step 4. Issue Filtering** — 웹서치를 통해 LLM이 학습하지 않은 최신 정보로 오탐 가능성이 있는 오류 분류
- **Step 5. Issue Judgement** — 갱신된 후보에 대해 모델별 세부 점수(`is_valid_issue`, `category_severity`, `context_resolution`)를 산출하고, 가중 평균으로 최종 점수를 계산해 지식 오류를 확정

&nbsp;&nbsp;슬라이드 기반 탐지는 이와 별개로, 슬라이드 텍스트만을 대상으로 모든 LLM이 오탈자·문장 간 논리적 모순 등 형식 오류를 병렬 탐지한다.
</div>
</details>

<details>
<summary><b>5) 상세 피드백 제공</b></summary>
<div markdown="1">
&nbsp;&nbsp;최종 확정된 이슈는 오류 유형별로 구분되어 제공된다. 오류가 발생한 원문 발화와 영상 구간 위치, Multi-LLM이 해당 오류를 판단한 근거, 오류의 심각도, 그리고 개선을 위한 수정 제안이 함께 제공된다. 플랫폼별·도메인별 오류 경향이나 분석 소요 시간 등을 파악할 수 있는 분석 통계 시각화도 함께 제공한다.
</div>
</details>

<br/>
<br/>

## 👀 3. 기대 효과

<details>
<summary><b>높은 정량적 성능 달성</b></summary>
<div markdown="1">
&nbsp;&nbsp;본 팀은 강의 영상 15개에 각각 10개의 오류를 의도적으로 주입한 데이터셋을 구성하고, GPT-5.4·Claude-Sonnet-5·Grok-4.5를 각각 단일로 사용한 경우와 세 모델을 함께 사용한 VeriLec의 성능을 비교 평가하였다.

<p align="center"><img src="./docs/img/performance-chart.png" alt="오류 탐지율 및 오탐률 비교" width="650"/></p>

- **오류 탐지율**: 주입 오류 중 실제 탐지된 오류의 비율 — VeriLec **94.0%**로 가장 높은 탐지 성능
- **오탐률**: 탐지된 오류 중 실제 오류가 아닌 것의 비율 — VeriLec **9.64%**
- **소요 시간**: 멀티모달 분석부터 오류 탐지·피드백 제공까지 — 강의 길이 1분당 **15.35초**

&nbsp;&nbsp;GPT-5.4는 탐지율은 높지만 오탐률도 함께 높았고, Claude-Sonnet-5는 오탐률이 가장 낮은 대신 탐지율이 낮았다. VeriLec은 두 지표를 함께 고려했을 때 가장 변동성이 적은, 균형 잡힌 성능을 보였다.
</div>
</details>

<details>
<summary><b>오픈소스로서의 우수성</b></summary>
<div markdown="1">
&nbsp;&nbsp;특정 LLM에 종속되지 않는 구조로, 기관이나 사용처의 환경과 목적에 맞는 LLM을 자유롭게 구성하여 활용할 수 있다. 사용자가 자체 환경에 맞게 기능을 추가하거나 오류 탐지 방식을 확장하는 것도 가능하다.
</div>
</details>

<details>
<summary><b>근거 기반의 자세한 피드백 제공</b></summary>
<div markdown="1">
&nbsp;&nbsp;단순히 오류 여부만 알려주는 것이 아니라, Multi-LLM 각각의 판단 근거와 문제의 심각도, 수정 제안까지 함께 제공하여 강의 제작자가 실제로 강의를 개선하는 데 바로 활용할 수 있다.
</div>
</details>

<details>
<summary><b>혁신성 및 차별점</b></summary>
<div markdown="1">

- 발화와 슬라이드를 함께 멀티모달로 분석해 강의 맥락을 반영
- 특정 LLM에 종속되지 않는 어댑터 구조로, 제약 없이 N개의 LLM 모델로 확장 가능
- 웹 검색을 통해 LLM이 학습하지 않은 최신 정보까지 반영해 오탐률 감소
- 지식 오류를 5개의 세부 유형으로 세분화하여 탐지
- Multi-LLM의 의견 불일치를 활용한 오탐 필터링 알고리즘 개발
</div>
</details>

<br/>
<br/>

## 👍 4. 활용 분야

<details>
<summary><b>이러닝 플랫폼 적용</b></summary>
<div markdown="1">
&nbsp;&nbsp;대학의 e-class나 K-MOOC, KOCW, Coursera 등 다양한 국내외 이러닝 플랫폼에 즉시 적용 가능하다.
</div>
</details>

<details>
<summary><b>기업 교육 영상 검증</b></summary>
<div markdown="1">
&nbsp;&nbsp;방대한 양의 직무 교육 영상을 자체 제작하는 대기업 및 기업 교육 전문 기업의 영상 검증에 도입할 수 있다.
</div>
</details>

<details>
<summary><b>개인 강의 제작자의 자가 점검 도구</b></summary>
<div markdown="1">
&nbsp;&nbsp;개인 강의 제작자가 YouTube, 인프런 등의 플랫폼에 영상을 업로드하기 전, 자가 점검 도구로 활용할 수 있다.
</div>
</details>

<br/>
<br/>

## 🖼 5. 프로젝트 결과

### 페이지 캡쳐본

- LLM 등록 및 Multi-LLM 셋 구성 화면
  <p align="center"><img src="./docs/img/capture-llm-setup.png" alt="LLM 등록 및 Multi-LLM 셋 구성 화면" width="1000"/></p>

- 지식 오류 탐지 결과 및 상세 피드백 화면
  <p align="center"><img src="./docs/img/capture-feedback.png" alt="오류 탐지 결과 및 피드백 화면" width="700"/></p>

<br/>

### 시연 영상

> 시연 영상은 준비 중입니다. 촬영 후 아래에 유튜브 링크와 썸네일이 추가될 예정입니다.
>
> ```
> [<img src="썸네일_이미지_URL" alt="VeriLec 시연"/>](유튜브_링크)
> ```

<br/>
<br/>

## 🔧 개발 가이드 (부록)

### 구성

```
backend/    FastAPI 백엔드 (업로드, 잡 큐, SSE 진행 스트림, 결과 API)
            └─ 워커는 별도 프로세스가 아니라 FastAPI lifespan 안에서 DB를 폴링하고,
               잡을 집으면 ProcessPoolExecutor 자식 프로세스로 파이프라인을 실행
pipeline/   분석 파이프라인
            ├─ preprocess/  슬라이드 추출, 텍스트화, 전사, 필기/오디오 분석
            ├─ verifier/    claim 추출 → 이슈 판단/분류 → 멀티 LLM 검증
            ├─ orchestration/  verify 워크플로우 조립 (진입점: pipeline/main.py)
            └─ cli/         python -m pipeline.cli.main (run-video / run-verify)
frontend/   테스트 콘솔 (Vite + React) — 업로드, 9단계 진행 표시, 결과 조회
scripts/    dev_backend.sh · dev_frontend.sh · run_video.sh · run_verify_only.sh
storage/    inputs/(업로드 영상) · results/(분석 산출물) — DB에는 이 디렉터리
            기준 상대경로로 저장되어 호스트/컨테이너 어디서든 해석된다
tests/      pytest 회귀 스위트 (DB/백엔드 없으면 통합 테스트 자동 skip)
```

### 빠른 시작 (로컬 개발)

요구사항: Python 3.12, Node 18+, Docker, ffmpeg

```bash
# 1. 환경변수 — .env.example을 복사해 API 키를 채운다
cp .env.example .env

# 2. 백엔드
./scripts/verilec_compose.sh up -d --build --force-recreate

# 3. 프론트 (5173, /api 프록시로 백엔드 연결)
VERILEC_API_TARGET=http://localhost:8003 ./scripts/dev_frontend.sh
```

브라우저에서 http://localhost:5173 → 영상 업로드 → 진행 표시 → 검증 결과.

#### 필수 환경변수

| 변수 | 용도 |
|---|---|
| `GOOGLE_API_KEY_1` (·`_2`) | Gemini — 슬라이드 텍스트화·필기 분석·텍스트 교정 |
| `GROQ_API_KEY` | Whisper 전체 전사 |
| `OPENAI_API_KEY` 등 | `ISSUE_JUDGE_MODELS`에 지정한 멀티 LLM 검증 모델의 키 |
| `DATABASE_URL` | 로컬 기본: `postgresql+asyncpg://user:pass@localhost:5433/verilec` |

키가 없으면 서버는 뜨지만, 해당 단계 실행 시점에 어떤 변수가 필요한지 명시된 에러가 난다.

#### Docker 실행

```bash
docker compose up -d          # db + backend (호스트 8020)
```

backend 이미지는 CPU 전용 torch를 사용한다 (~3.5GB). 프론트는 별도로
`VERILEC_API_TARGET=http://localhost:8020 ./scripts/dev_frontend.sh`.

#### CPU/GPU Docker 실행

macOS 또는 CUDA 없는 환경에서는 CPU 구성으로 실행합니다.

```bash
./scripts/verilec_compose.sh build
./scripts/verilec_compose.sh up -d
```

NVIDIA Docker 런타임을 사용할 수 있는 Linux에서는 같은 실행 스크립트가 GPU 구성을 자동 선택합니다. 필요하면 `VERILEC_MODE=cpu` 또는 `VERILEC_MODE=gpu`로 명시할 수 있습니다. CPU 구성은 OpenCV 디코딩, YOLO `.pt` 모델, RapidOCR를 사용합니다. GPU 구성은 CUDA PyTorch, FFmpeg CUDA 디코딩, TensorRT 사람 마스크 및 Nemotron OCR를 사용합니다.

### 개발 환경

| 구분 | 사용 기술/도구 |
|---|---|
| 인프라 | RTX 4090, FastAPI, PostgreSQL 16 |
| 영상 분석용 LLM | gemma4:12b, qwen3-vl:30b, qwen3.6:27b, qwen3.8:27b |
| Multi-LLM 테스트 | gpt-5.4, claude-sonnet-4.5, grok-4, qwen3.8:27b |
| 개발 스택 | Vite·React, Python·JavaScript·HTML5·CSS, OpenCV·YOLO·Librosa |

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/fastapi-005571?style=for-the-badge&logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/postgresql-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Ultralytics YOLO](https://img.shields.io/badge/YOLO-111F68?style=for-the-badge&logo=ultralytics&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

### 테스트

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

- 유닛: 스토리지 경로 해석, config lazy init, 내부 import 정적 검사, 워커-파이프라인 인자 계약, **검증 결과 API 스키마 계약**(`tests/test_verifier_contract.py`)
- 통합(자동 skip 가능): stage 업데이트 DB 회귀, 실행 중인 백엔드 API 스모크
- CI: GitHub Actions(`.github/workflows/ci.yml`)가 push/PR마다 Postgres 서비스와 함께 실행

### API 계약 메모

`GET /results/{id}/verifier` 응답의 키 구성은 프론트(`VerifierResults.jsx`)와의 계약으로,
`backend/app/services/lecture_service.py`의 `build_content_verification_response`가 생성하고
`tests/test_verifier_contract.py`가 고정한다. 키를 바꾸려면 매퍼·테스트·프론트를 함께 수정할 것.

핵심 키: `counts{final_confirmed, needs_review, rejected, slide_errors, …}`,
`final_confirmed_claims[]`, `needs_review_claims[]`, `verifier_rejected_claims[]`,
`slide_errors[]`, `models[]`, `verification_date`.

<br/>

<p align="center">
  <img
    src="https://capsule-render.vercel.app/api?type=waving&color=0:4F8CFF,100:7C4DFF&height=100&section=footer"
    width="100%"
  />
</p>

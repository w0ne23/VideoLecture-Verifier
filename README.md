# VeriLec

강의 영상의 내용 오류를 자동 검증하는 서비스. 영상을 업로드하면 슬라이드 추출·전사·필기 분석을 거쳐, 강의 발화에서 주장(claim)을 추출하고 멀티 LLM으로 사실성을 검증한 뒤 확정 이슈/검토 필요/슬라이드 오류를 리포트한다.

## 구성

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

## 빠른 시작 (로컬 개발)

요구사항: Python 3.12, Node 18+, Docker, ffmpeg

```bash
# 1. 환경변수 — .env.example을 복사해 API 키를 채운다
cp .env.example .env

# 2. Postgres (호스트 포트 5433로 노출)
docker compose up -d db

# 3. 파이썬 의존성
python3.12 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt -r pipeline/requirements.verify.txt

# 4. 백엔드 (기본 8000, 포트가 차 있으면 PORT=8010)
PORT=8010 ./scripts/dev_backend.sh

# 5. 프론트 (5173, /api 프록시로 백엔드 연결)
VERILEC_API_TARGET=http://localhost:8010 ./scripts/dev_frontend.sh
```

브라우저에서 http://localhost:5173 → 영상 업로드 → 진행 표시 → 검증 결과.

### 필수 환경변수

| 변수 | 용도 |
|---|---|
| `GOOGLE_API_KEY_1` (·`_2`) | Gemini — 슬라이드 텍스트화·필기 분석·텍스트 교정 |
| `GROQ_API_KEY` | Whisper 전체 전사 |
| `OPENAI_API_KEY` 등 | `ISSUE_JUDGE_MODELS`에 지정한 멀티 LLM 검증 모델의 키 |
| `DATABASE_URL` | 로컬 기본: `postgresql+asyncpg://user:pass@localhost:5433/verilec` |

키가 없으면 서버는 뜨지만, 해당 단계 실행 시점에 어떤 변수가 필요한지 명시된 에러가 난다.

## Docker 실행

```bash
docker compose up -d          # db + backend (호스트 8020)
```

backend 이미지는 CPU 전용 torch를 사용한다 (~3.5GB). 프론트는 별도로
`VERILEC_API_TARGET=http://localhost:8020 ./scripts/dev_frontend.sh`.

## 테스트

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

- 유닛: 스토리지 경로 해석, config lazy init, 내부 import 정적 검사, 워커-파이프라인 인자 계약, **검증 결과 API 스키마 계약**(`tests/test_verifier_contract.py`)
- 통합(자동 skip 가능): stage 업데이트 DB 회귀, 실행 중인 백엔드 API 스모크
- CI: GitHub Actions(`.github/workflows/ci.yml`)가 push/PR마다 Postgres 서비스와 함께 실행

## API 계약 메모

`GET /results/{id}/verifier` 응답의 키 구성은 프론트(`VerifierResults.jsx`)와의 계약으로,
`backend/app/services/lecture_service.py`의 `build_content_verification_response`가 생성하고
`tests/test_verifier_contract.py`가 고정한다. 키를 바꾸려면 매퍼·테스트·프론트를 함께 수정할 것.

핵심 키: `counts{final_confirmed, needs_review, rejected, slide_errors, …}`,
`final_confirmed_claims[]`, `needs_review_claims[]`, `verifier_rejected_claims[]`,
`slide_errors[]`, `models[]`, `verification_date`.

## CPU/GPU Docker 실행

macOS 또는 CUDA 없는 환경에서는 CPU 구성으로 실행합니다.

```bash
./scripts/verilec_compose.sh build
./scripts/verilec_compose.sh up -d
```

NVIDIA Docker 런타임을 사용할 수 있는 Linux에서는 같은 실행 스크립트가 GPU
구성을 자동 선택합니다. 필요하면 `VERILEC_MODE=cpu` 또는 `VERILEC_MODE=gpu`로
명시할 수 있습니다.

CPU 구성은 OpenCV 디코딩, YOLO `.pt` 모델, RapidOCR를 사용합니다. GPU 구성은
CUDA PyTorch, FFmpeg CUDA 디코딩, TensorRT 사람 마스크 및 Nemotron OCR를 사용합니다.

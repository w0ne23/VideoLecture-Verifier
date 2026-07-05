# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base-deps

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    gcc \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 파이프라인 deps
COPY app/backend/pipeline/requirements.txt ./requirements_pipeline.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements_pipeline.txt

# 백엔드 deps
COPY app/backend/requirements.txt ./requirements_backend.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements_backend.txt

# 코드 레이어
FROM base-deps AS final

COPY app/backend/pipeline/ ./pipeline/
COPY app/backend/app/ ./app/
COPY app/backend/recommender/ ./recommender/

ENV PYTHONPATH=/app
ENV PIPELINE_ROOT=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

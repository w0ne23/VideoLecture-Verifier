FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
COPY pipeline/requirements.verify.txt ./pipeline/requirements.verify.txt
# CPU 전용 torch를 먼저 설치 — 컨테이너에 GPU가 없으므로, 이후 ultralytics가
# PyPI 기본(CUDA 동반, ~7GB) torch를 끌고 오지 않게 한다.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    -r backend/requirements.txt \
    -r pipeline/requirements.verify.txt

COPY backend ./backend
COPY pipeline ./pipeline

ENV PYTHONPATH=/app:/app/backend
WORKDIR /app/backend

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

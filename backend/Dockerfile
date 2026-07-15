FROM python:3.12-slim

WORKDIR /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} \
    torch \
    torchvision \
    torchaudio
COPY pipeline/requirements.verify.txt ./pipeline/requirements.verify.txt
RUN pip install --no-cache-dir \
    -r backend/requirements.txt \
    -r pipeline/requirements.verify.txt

RUN mkdir -p /app/models && \
    python -c "from ultralytics import YOLO; import shutil; model=YOLO('yolo26n.pt'); source='yolo26n.pt'; shutil.move(source, '/app/models/yolo26n.pt') if __import__('pathlib').Path(source).exists() else None"

COPY backend ./backend
COPY pipeline ./pipeline

ENV PYTHONPATH=/app:/app/backend
WORKDIR /app/backend

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

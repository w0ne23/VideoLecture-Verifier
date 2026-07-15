ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.09-py3
FROM ${BASE_IMAGE}

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    build-essential \
    cmake \
    ninja-build \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY ocr_service/requirements.txt ./ocr_service/requirements.txt
RUN pip install --no-cache-dir -r ocr_service/requirements.txt \
    && pip install --no-cache-dir hatchling build wheel setuptools

RUN git lfs install && \
    git clone https://huggingface.co/nvidia/nemotron-ocr-v2 /tmp/nemotron-ocr-v2 && \
    pip install --no-cache-dir --no-build-isolation -v /tmp/nemotron-ocr-v2/nemotron-ocr && \
    rm -rf /tmp/nemotron-ocr-v2

COPY ocr_service ./ocr_service

ENV PYTHONPATH=/app
WORKDIR /app/ocr_service

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

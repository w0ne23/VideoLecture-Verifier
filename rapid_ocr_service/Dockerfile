FROM python:3.12-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    RAPIDOCR_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY rapid_ocr_service/requirements.txt ./rapid_ocr_service/requirements.txt
RUN pip install --no-cache-dir -r rapid_ocr_service/requirements.txt

COPY rapid_ocr_service ./rapid_ocr_service

ENV PYTHONPATH=/app
WORKDIR /app/rapid_ocr_service

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt

# Счёт мешков идёт на CPU и GPU не использует, поэтому torch ставится из
# CPU-индекса: обычное колесо тянет за собой CUDA, cuDNN и triton, а это
# примерно 4 ГБ в образе, которые здесь никогда не выполняются.
RUN pip install --no-cache-dir --timeout 300 --retries 10 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir --timeout 300 --retries 10 -r /tmp/requirements.txt

COPY backend/app /app/app
COPY frontend /app/frontend
COPY models /app/models

RUN mkdir -p /app/uploads

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

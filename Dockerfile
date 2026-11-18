# Sapphirus Assistant API (FastAPI + RAG pipeline + background ingestion worker)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface TRANSFORMERS_NO_ADVISORY_WARNINGS=1
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# CPU-only PyTorch keeps the image small (the embedding + reranker models run on CPU)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt requirements-optional.txt ./
RUN pip install -r requirements.txt pytesseract Pillow

COPY . .
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p data logs .cache/huggingface knowledge_bases \
    && chown -R appuser /app
USER appuser

# Download the embedding and reranker models at build time so the container starts offline and fast
ARG PRELOAD_MODELS=1
RUN if [ "$PRELOAD_MODELS" = "1" ]; then python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"; fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health/live || exit 1
# ONE worker process: it hosts the ingestion worker thread and is the single writer of the embedded vector store.
# Scale vertically (threads) or move to a client/server vector DB before running several replicas (see docs).
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--timeout-keep-alive", "30"]

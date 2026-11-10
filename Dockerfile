# CascadeIQ API image. Python 3.12 because the pinned numpy in requirements-api.txt needs >= 3.12.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY requirements-api.txt ./
RUN pip install -r requirements-api.txt

COPY api.py ./
COPY ingestion ./ingestion

# Download the embedding model at BUILD time into the image; the API then runs in
# local-cache-only mode (EMBEDDING_ALLOW_DOWNLOAD is never set at runtime).
RUN python -c "import sys; sys.path.insert(0, '.'); \
from ingestion.embedding_config import create_embedding_model; create_embedding_model(allow_download=True)" \
    && mkdir -p logs results \
    && chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"

# Secrets and settings come from the environment (see .env.example); nothing is baked into the image.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]

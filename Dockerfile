# =========================================================
# Stage 1: Build & Dependencies
# =========================================================
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Use a dedicated virtual environment in /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# =========================================================
# Stage 2: Final Production Runtime
# =========================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install minimal runtime libraries for PyMuPDF, onnxruntime, and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASHRANK_CACHE_DIR=/app/models/flashrank
ENV FASTEMBED_CACHE_PATH=/app/models/fastembed
ENV PORT=8000

# Copy application source code
COPY app/ ./app
COPY render.yaml .

# Pre-bake FlashRank cross-encoder and FastEmbed models into image
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-TinyBERT-L-2-v2', cache_dir='/app/models/flashrank')"
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/models/fastembed')"
RUN python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding(model_name='Qdrant/bm25', cache_dir='/app/models/fastembed')"

# Create non-root system user and assign permissions
RUN useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app /opt/venv
USER appuser

EXPOSE 8000

# Built-in container health check hitting the top-level /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# RAG Service Dockerfile
# Multi-stage build for RAG retrieval service

FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    sentence-transformers==2.2.2 \
    faiss-cpu==1.7.4 \
    pydantic==2.5.2 \
    numpy==1.26.2

FROM python:3.11-slim as production

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN groupadd -r rag && useradd -r -g rag rag

COPY services/rag/ /app/services/rag/
COPY data/docs/ /app/data/docs/ 2>/dev/null || true

ENV PYTHONPATH=/app
ENV PORT=8001

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

RUN chown -R rag:rag /app
USER rag

CMD ["uvicorn", "services.rag.retriever:app", "--host", "0.0.0.0", "--port", "8001"]

# Prediction Service Dockerfile
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Builder
# ============================================================================

FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    gunicorn==21.2.0 \
    pandas==2.1.3 \
    numpy==1.26.2 \
    pydantic==2.5.2 \
    python-multipart==0.0.6 \
    requests==2.31.0 \
    joblib==1.3.2 \
    mlflow==2.9.2


# ============================================================================
# Stage 2: Production
# ============================================================================

FROM python:3.11-slim as production

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r predict && useradd -r -g predict predict

# Copy application code
COPY services/predict/ /app/services/predict/
COPY models/ /app/models/ 2>/dev/null || true
COPY conf/ /app/conf/ 2>/dev/null || true

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV WORKERS=4

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Change ownership and switch to non-root user
RUN chown -R predict:predict /app
USER predict

# Start command using gunicorn with uvicorn workers
CMD ["sh", "-c", "gunicorn services.predict.app:app \
    --bind 0.0.0.0:${PORT} \
    --workers ${WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -"]


# ============================================================================
# Stage 3: Development
# ============================================================================

FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    httpx==0.25.2 \
    ipython==8.18.1

USER predict

# Override command for development
CMD ["uvicorn", "services.predict.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

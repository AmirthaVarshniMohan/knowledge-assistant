# ==========================================
# Stage 1: Build & Dependencies
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==========================================
# Stage 2: Production Lightweight Runtime
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Create unprivileged application user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source code and configs
COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/ ./data/

# Create data directories with appropriate permissions
RUN mkdir -p /app/data/vector_store /app/logs && \
    chown -R appuser:appuser /app/data /app/logs && \
    pip install --no-cache-dir -e .

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "knowledge_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Multi-stage build for Python 3.8.8 - Fixed version
FROM python:3.8.8-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install pip and packages in separate steps to avoid issues
RUN pip install --no-cache-dir --upgrade pip==21.3.1
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.8.8-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r apiuser && useradd -r -g apiuser apiuser

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY requirements.txt .
COPY app/ ./app/
COPY model/ ./model/

# Set ownership
RUN chown -R apiuser:apiuser /app && \
    chmod -R 755 /app

USER apiuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log"]
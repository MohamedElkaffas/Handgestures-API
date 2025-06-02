FROM python:3.8.8-slim as builder

# Install build dependencies 
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Upgrade pip for Python 3.8 compatibility and install dependencies
RUN pip install --no-cache-dir --upgrade pip==21.3.1 && \
    pip install --no-cache-dir -r requirements.txt

# Production stage - minimal image
FROM python:3.8.8-slim as production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r apiuser && useradd -r -g apiuser apiuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.8/site-packages /usr/local/lib/python3.8/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code (in correct order for Docker layer caching)
COPY requirements.txt .
COPY app/ ./app/
COPY model/ ./model/

# Set proper ownership and permissions
RUN chown -R apiuser:apiuser /app && \
    chmod -R 755 /app

# Switch to non-root user
USER apiuser

# Comprehensive health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Set environment variables for production
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Run application with production settings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log"]
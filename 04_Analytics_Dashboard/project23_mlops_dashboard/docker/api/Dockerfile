FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi==0.109.0 \
    uvicorn[standard]==0.25.0 \
    tritonclient[all]==2.40.0 \
    mlflow==2.9.2 \
    pydantic==2.5.3 \
    prometheus-client==0.19.0

# Copy application code
COPY src /app/src
COPY config /app/config

# Expose API port
EXPOSE 8080

# Expose metrics port
EXPOSE 9091

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run FastAPI with uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]

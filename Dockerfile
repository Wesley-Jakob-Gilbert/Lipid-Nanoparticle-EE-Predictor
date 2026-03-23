FROM python:3.11-slim

LABEL maintainer="lnp-ee-predictor"
LABEL description="LNP Encapsulation Efficiency Prediction API"

# System dependencies for RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and artifacts
COPY src/ ./src/
COPY api/ ./api/
COPY artifacts/ ./artifacts/

# Make src importable from api/
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

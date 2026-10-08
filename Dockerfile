# Dockerfile for codelist-rag API service
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration and source code
COPY pyproject.toml /app/
COPY src/ /app/src/
COPY tests/fixtures/ /app/tests/fixtures/

# Install codelist-rag package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Expose port for FastAPI application
EXPOSE 8000

# Declare environment variables for index and fixture paths
ENV CHROMA_PATH=/app/data/chroma_db
ENV BM25_PATH=/app/data/bm25_index.pkl
ENV TERMINOLOGY_RELEASE="not recorded"

# Run FastAPI service with Uvicorn
CMD ["uvicorn", "codelist_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

# syntax: docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY agent/     agent/
COPY indexer/   indexer/
COPY main.py    .

# Pre-built catalog data (committed to repo after scraping)
COPY data/      data/

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

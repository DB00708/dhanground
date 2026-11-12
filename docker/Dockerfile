# CPU-only integration image: requirements.txt + CuratorKIT core. No torch, no GPU, no API keys.
# Build context is the repo root (see docker-compose.yml). Data is mounted read-only at /data.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app DHANGROUND_DATA=/data E2E_WORK=/work
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt curatorkit==0.1.24

COPY . .

# Unit suite, then the e2e driver on a small slice.
CMD ["sh", "-c", "python -m pytest -q -p no:cacheprovider && python tests/e2e/run_e2e.py"]

# Crate — self-contained app image. Serves the FastAPI search app on :8000.
FROM python:3.11-slim

# ffmpeg: decode mic (webm/opus) + arbitrary dropped/uploaded audio
# git: pip may resolve VCS deps
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY crate ./crate
RUN pip install --no-cache-dir ".[stems]"

# data/ and models/ are mounted at runtime (see docker-compose.yaml);
# HF_HOME caches the fine-tuned adapter downloaded from the Hub.
ENV CRATE_DATA=/app/data \
    CRATE_MODELS=/app/models \
    CRATE_SESSIONS=/app/sessions \
    HF_HOME=/app/.hfcache \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/latency')" || exit 1
CMD ["uvicorn", "crate.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

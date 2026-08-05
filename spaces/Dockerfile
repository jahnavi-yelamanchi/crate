# Crate demo — Hugging Face Docker Space. Listens on 7860.
FROM python:3.11-slim

# ffmpeg: decode mic (webm/opus) + arbitrary dropped audio
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run as uid 1000; keep caches writable
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    CRATE_MODELS=/home/user/app/models \
    CRATE_DATA=/home/user/app/data
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Install the package (app + stem-separation extras) from GitHub.
RUN pip install --no-cache-dir \
    "crate[stems] @ git+https://github.com/jahnavi-yelamanchi/crate.git"

COPY --chown=user boot.py .

EXPOSE 7860
CMD ["python", "boot.py"]

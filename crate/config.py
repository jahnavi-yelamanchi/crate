"""Central config: paths, model ids, audio params. Everything else imports from here."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv optional at runtime
    pass

# --- Paths (repo-relative; data/models are gitignored) ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("CRATE_DATA", ROOT / "data"))
AUDIO_DIR = DATA_DIR / "audio"          # preprocessed 48k mono clips
RAW_DIR = DATA_DIR / "raw"              # downloads before preprocessing
META_PATH = DATA_DIR / "metadata.jsonl"  # one clip per line: {id, path, text, license, source}
PAIRS_DIR = DATA_DIR / "pairs"          # train/val/held-out (audio, text) splits
MODELS_DIR = Path(os.getenv("CRATE_MODELS", ROOT / "models"))
INDEX_PATH = MODELS_DIR / "index.faiss"
IDS_PATH = MODELS_DIR / "ids.json"
LORA_DIR = MODELS_DIR / "clap-lora"     # local adapter dir
ONNX_PATH = MODELS_DIR / "audio_encoder.int8.onnx"
SESSIONS_DIR = Path(os.getenv("CRATE_SESSIONS", ROOT / "sessions"))
EVAL_DIR = ROOT / "eval"
VOCAB_PATH = ROOT / "crate" / "vocab" / "producer_terms.txt"

# --- Models ---
CLAP_BASE = "laion/clap-htsat-unfused"
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "")   # your-username/crate-clap-lora

# --- Audio ---
SAMPLE_RATE = 48_000     # CLAP-HTSAT expects 48kHz
CLIP_SECONDS = 10.0      # fixed length; trim/pad to this
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
EMBED_DIM = 512          # CLAP projection dim

# --- Retrieval / ranking ---
TOP_K = 20
DEDUP_COSINE = 0.98      # near-duplicate flood guard: drop results this similar to a kept one
TASTE_ALPHA = 0.7        # weight on similarity
TASTE_BETA = 0.3         # weight on personal taste score

# --- Secrets (from .env) ---
FREESOUND_KEY = os.getenv("FREESOUND_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")


def ensure_dirs() -> None:
    """Make the runtime dirs. Safe to call repeatedly."""
    for d in (DATA_DIR, AUDIO_DIR, RAW_DIR, PAIRS_DIR, MODELS_DIR, SESSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)

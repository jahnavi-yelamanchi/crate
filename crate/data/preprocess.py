"""Preprocess raw audio → 48k mono, fixed length, cached as .npy.

CLAP-HTSAT wants 48kHz mono. We trim/pad to CLIP_SECONDS so batching is trivial
and the encoder sees a consistent window.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from crate import config


def load_audio(path: str | Path) -> np.ndarray:
    """Load any format → float32 mono at SAMPLE_RATE. librosa handles resampling."""
    import librosa

    wav, _ = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True)
    return wav.astype(np.float32)


def fix_length(wav: np.ndarray, n: int = config.CLIP_SAMPLES) -> np.ndarray:
    """Trim or zero-pad to exactly n samples."""
    if len(wav) >= n:
        return wav[:n]
    return np.pad(wav, (0, n - len(wav)))


def preprocess_one(src: str | Path) -> np.ndarray:
    return fix_length(load_audio(src))


def run() -> int:
    """Preprocess every metadata clip whose cache is missing. Returns count done."""
    config.ensure_dirs()
    if not config.META_PATH.exists():
        raise FileNotFoundError("no metadata.jsonl — run ingest first")
    done = 0
    for line in config.META_PATH.read_text().splitlines():
        rec = json.loads(line)
        out = config.AUDIO_DIR / f"{rec['id']}.npy"
        if out.exists():
            continue
        try:
            wav = preprocess_one(rec["path"])
        except Exception as e:  # skip corrupt/missing files, keep going
            print(f"skip {rec['id']}: {e}")
            continue
        np.save(out, wav)
        done += 1
    return done


if __name__ == "__main__":
    print(f"preprocessed {run()} clips → {config.AUDIO_DIR}")

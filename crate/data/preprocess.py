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


def _process(rec: dict) -> bool:
    """Decode+resample one clip to cache. Returns True if written."""
    out = config.AUDIO_DIR / f"{rec['id']}.npy"
    if out.exists():
        return False
    try:
        np.save(out, preprocess_one(rec["path"]))
        return True
    except Exception as e:  # skip corrupt/missing files, keep going
        print(f"skip {rec['id']}: {e}")
        return False


def run(workers: int = 8) -> int:
    """Preprocess every uncached clip in parallel. Returns count done.

    # ponytail: threads — librosa/soundfile release the GIL during decode+resample,
    # so this overlaps well; switch to ProcessPoolExecutor if pure-CPU resampling
    # dominates on a many-core box.
    """
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    config.ensure_dirs()
    if not config.META_PATH.exists():
        raise FileNotFoundError("no metadata.jsonl — run ingest first")
    recs = [json.loads(line) for line in config.META_PATH.read_text().splitlines()]
    with ThreadPoolExecutor(workers) as ex:
        return sum(tqdm(ex.map(_process, recs), total=len(recs), desc="preprocess"))


if __name__ == "__main__":
    print(f"preprocessed {run()} clips → {config.AUDIO_DIR}")

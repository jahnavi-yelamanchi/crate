"""Build (audio, text) pairs from metadata + producer vocab, split train/val/held-out.

Output: pairs/{train,val,heldout}.jsonl — each line {id, audio_npy, text}. The
held-out split is what recall@k eval scores against, so it never touches training.
"""

from __future__ import annotations

import json
import random

from crate import config


def load_vocab(path=config.VOCAB_PATH) -> list[str]:
    """Producer terms, comments/blanks stripped."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def _records() -> list[dict]:
    if not config.META_PATH.exists():
        raise FileNotFoundError("no metadata.jsonl — run ingest first")
    recs = []
    for line in config.META_PATH.read_text().splitlines():
        rec = json.loads(line)
        npy = config.AUDIO_DIR / f"{rec['id']}.npy"
        if npy.exists():  # only pair clips that survived preprocessing
            recs.append({"id": rec["id"], "audio_npy": str(npy), "text": rec["text"]})
    return recs


def build(seed: int = 13, val_frac: float = 0.1, heldout_frac: float = 0.1) -> dict[str, int]:
    """Shuffle, split, write the three jsonl files. Returns per-split counts."""
    config.ensure_dirs()
    recs = _records()
    if not recs:
        raise RuntimeError("no preprocessed clips — run preprocess before pairs")
    random.Random(seed).shuffle(recs)
    n = len(recs)
    n_val = int(n * val_frac)
    n_held = int(n * heldout_frac)
    splits = {
        "heldout": recs[:n_held],
        "val": recs[n_held:n_held + n_val],
        "train": recs[n_held + n_val:],
    }
    counts = {}
    for name, rows in splits.items():
        path = config.PAIRS_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        counts[name] = len(rows)
    return counts


def load_split(name: str) -> list[dict]:
    path = config.PAIRS_DIR / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


if __name__ == "__main__":
    print(build())

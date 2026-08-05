"""Text-vs-audio query parity — do both modalities land in the same place?

For each held-out clip we run its text caption and its audio through the retriever
and measure Jaccard overlap of the top-k id sets. High overlap = a text query and
an audio query for the same thing return the same crate. That's the whole promise:
one embedding space, three ways in.
"""

from __future__ import annotations

import json

import numpy as np

from crate import config
from crate.data.pairs import load_split


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def run(k: int = 10, limit: int = 200) -> dict:
    from crate.index.search import get_retriever

    rows = load_split("heldout")[:limit]
    if not rows:
        raise RuntimeError("empty heldout split — build data + index first")
    r = get_retriever()

    overlaps = []
    for row in rows:
        wav = np.load(row["audio_npy"])
        text_ids = {x["id"] for x in r.search(row["text"], k=k, dedup=False)}
        audio_ids = {x["id"] for x in r.search(wav, k=k, dedup=False)}
        overlaps.append(jaccard(text_ids, audio_ids))

    result = {"k": k, "n": len(rows), "mean_jaccard": float(np.mean(overlaps))}
    config.EVAL_DIR.mkdir(exist_ok=True)
    (config.EVAL_DIR / "parity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()

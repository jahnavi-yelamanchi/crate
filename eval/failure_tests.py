"""Failure testing — the ways sound search breaks in the wild.

1. Bad hums     : people who can't hum. Heavily degrade a query, check retrieval
                  degrades gracefully instead of returning garbage or crashing.
2. Genre drift  : slang the model never trained on ("phonk cowbell" vs "cowbell,
                  aggressive, memphis"). Measure top-k overlap with the canonical term.
3. Dup flooding : a pack with the same loop 12 times. Confirm the dedup guard keeps
                  near-identical results out of the top-k.

Runs against a built index; skips with a note if artifacts are missing so it never
hard-fails CI.
"""

from __future__ import annotations

import json

import numpy as np

from crate import config
from crate.data.pairs import load_split

# canonical producer term → slangy paraphrase a user might actually type
DRIFT_PAIRS = [
    ("boom-bap", "dusty 90s hip hop drums"),
    ("reese bass", "growly detuned dnb bass"),
    ("phonk cowbell", "memphis aggressive cowbell"),
    ("tape-saturated", "warm crunchy analog"),
]


def _retriever():
    try:
        from crate.index.search import get_retriever

        return get_retriever()
    except (FileNotFoundError, ImportError):
        return None


def bad_hums(r, k: int = 10) -> dict:
    """Median rank of the true clip: clean audio vs heavily-noised audio."""
    from crate.data.augment import add_noise

    rng = np.random.default_rng(0)
    rows = load_split("heldout")[:100]
    clean_ranks, bad_ranks = [], []
    for row in rows:
        wav = np.load(row["audio_npy"])
        bad = add_noise(wav, snr_db=-5, rng=rng)  # louder noise than signal
        clean_ranks.append(_rank_of(r, wav, row["id"], k))
        bad_ranks.append(_rank_of(r, bad, row["id"], k))
    return {"clean_hit_rate": _hit_rate(clean_ranks, k),
            "degraded_hit_rate": _hit_rate(bad_ranks, k)}


def genre_drift(r, k: int = 10) -> dict:
    """Top-k overlap between the canonical term and its slang paraphrase."""
    from eval.parity import jaccard

    overlaps = []
    for canon, slang in DRIFT_PAIRS:
        a = {x["id"] for x in r.search(canon, k=k, dedup=False)}
        b = {x["id"] for x in r.search(slang, k=k, dedup=False)}
        overlaps.append(jaccard(a, b))
    return {"mean_overlap": float(np.mean(overlaps)), "pairs": len(DRIFT_PAIRS)}


def dup_flooding(r, k: int = 10) -> dict:
    """Max count of mutually near-identical results with dedup on. Should be low."""
    rows = load_split("heldout")[:20]
    worst = 0
    for row in rows:
        res = r.search(np.load(row["audio_npy"]), k=k, dedup=True)
        # re-embed to check pairwise similarity among returned results
        embs = [x.get("emb") for x in res if x.get("emb") is not None]
        for i in range(len(embs)):
            dupes = sum(float(embs[i] @ embs[j]) >= config.DEDUP_COSINE
                        for j in range(len(embs)) if i != j)
            worst = max(worst, dupes)
    return {"max_near_dupes_in_topk": int(worst)}


def _rank_of(r, wav, true_id, k) -> int:
    ids = [x["id"] for x in r.search(wav, k=k, dedup=False)]
    return ids.index(true_id) if true_id in ids else k


def _hit_rate(ranks, k) -> float:
    return float(np.mean([rk < k for rk in ranks]))


def run() -> dict:
    r = _retriever()
    if r is None:
        note = {"note": "index not built — run the data pipeline + build_index first"}
        print(json.dumps(note))
        return note
    out = {"bad_hums": bad_hums(r), "genre_drift": genre_drift(r),
           "dup_flooding": dup_flooding(r)}
    config.EVAL_DIR.mkdir(exist_ok=True)
    (config.EVAL_DIR / "failure.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()

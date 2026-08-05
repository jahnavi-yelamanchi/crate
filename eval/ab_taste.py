"""A/B: does the taste head rank your saves above your skips better than similarity?

Splits logged events, trains taste on the train half, and compares AUC — the
probability a random saved clip outranks a random skipped one — for taste vs
raw similarity on the held-out half. Taste > similarity == the reco head earns
its place.
"""

from __future__ import annotations

import json

import numpy as np

from crate import config
from crate.rank.taste import _EVENTS, TasteModel, _features


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC. Ties count as half."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def run(test_frac: float = 0.3, seed: int = 0) -> dict:
    if not _EVENTS.exists():
        raise RuntimeError("no events logged — save/skip some results first")
    rows = [json.loads(x) for x in _EVENTS.read_text().splitlines()]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    n_test = max(1, int(len(rows) * test_frac))
    test, train = rows[:n_test], rows[n_test:]

    Xtr = np.array([_features(np.array(e["q"]), np.array(e["r"])) for e in train], np.float32)
    ytr = np.array([e["y"] for e in train], np.float32)
    model = TasteModel().fit(Xtr, ytr)

    y = np.array([e["y"] for e in test], np.float32)
    sim = np.array([float(np.array(e["q"]) @ np.array(e["r"])) for e in test], np.float32)
    taste = np.array([model.proba(np.array(e["q"]), np.array(e["r"])) for e in test], np.float32)

    result = {"n_test": len(test), "auc_similarity": auc(sim, y), "auc_taste": auc(taste, y)}
    config.EVAL_DIR.mkdir(exist_ok=True)
    (config.EVAL_DIR / "ab_taste.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()

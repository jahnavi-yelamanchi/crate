"""Personal taste ranker — your saves/skips are the training signal.

Digging history *is* collaborative signal. Every save/skip logs a
(query_emb, result_emb, label) event; a tiny logistic head learns which clips
you keep, and search results get re-ranked by α·similarity + β·taste. Pure numpy
so there's no sklearn/torch dependency on the serving path.
"""

from __future__ import annotations

import json

import numpy as np

from crate import config

_EVENTS = config.SESSIONS_DIR / "events.jsonl"
_WEIGHTS = config.MODELS_DIR / "taste.npz"


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _features(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Concat query and result embeddings → the head sees both sides of the match."""
    return np.concatenate([np.asarray(q, np.float32).ravel(),
                           np.asarray(r, np.float32).ravel()])


class TasteModel:
    def __init__(self, dim: int = 2 * config.EMBED_DIM):
        self.w = np.zeros(dim, np.float32)
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, lr: float = 0.1, l2: float = 1e-3):
        """Batch gradient descent logistic regression with L2. Small data, so this is plenty."""
        n = len(y)
        for _ in range(epochs):
            p = _sigmoid(X @ self.w + self.b)
            grad = X.T @ (p - y) / n + l2 * self.w
            self.w -= lr * grad
            self.b -= lr * float(np.mean(p - y))
        return self

    def proba(self, q: np.ndarray, r: np.ndarray) -> float:
        return float(_sigmoid(_features(q, r) @ self.w + self.b))

    def save(self):
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(_WEIGHTS, w=self.w, b=self.b)

    @classmethod
    def load(cls) -> "TasteModel":
        m = cls()
        if _WEIGHTS.exists():
            d = np.load(_WEIGHTS)
            m.w, m.b = d["w"], float(d["b"])
        return m


def log_event(query_emb, result_emb, saved: bool):
    """Append one save(True)/skip(False) event."""
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_EVENTS, "a") as f:
        f.write(json.dumps({
            "q": np.asarray(query_emb, np.float32).ravel().tolist(),
            "r": np.asarray(result_emb, np.float32).ravel().tolist(),
            "y": int(saved),
        }) + "\n")


def _load_events() -> tuple[np.ndarray, np.ndarray] | None:
    if not _EVENTS.exists():
        return None
    X, y = [], []
    for line in _EVENTS.read_text().splitlines():
        e = json.loads(line)
        X.append(_features(np.array(e["q"]), np.array(e["r"])))
        y.append(e["y"])
    if len({*y}) < 2:  # need both a save and a skip to learn anything
        return None
    return np.array(X, np.float32), np.array(y, np.float32)


def retrain() -> TasteModel | None:
    """Refit from all logged events and persist. Returns None until data is separable-ish."""
    data = _load_events()
    if data is None:
        return None
    model = TasteModel().fit(*data)
    model.save()
    return model


def rerank(query_emb, results: list[dict], model: TasteModel | None = None) -> list[dict]:
    """Blend similarity with taste. No model yet → similarity order unchanged."""
    model = model or TasteModel.load()
    if not np.any(model.w):  # untrained: pure similarity
        return sorted(results, key=lambda r: -r["score"])
    for r in results:
        taste = model.proba(query_emb, r["emb"])
        r["final"] = config.TASTE_ALPHA * r["score"] + config.TASTE_BETA * taste
    return sorted(results, key=lambda r: -r.get("final", r["score"]))

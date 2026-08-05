"""Distill the two-stage ranker (similarity + logistic taste) into one linear head.

Teacher at serve time does: cosine sim, then a logistic taste score, then blend.
The student fits a single linear map over [query, result] features to the teacher's
final score, so ranking becomes one dot product. Small quality trade for a simpler,
faster serving path.

# ponytail: least-squares student over logged events; re-distill whenever the taste
# model is retrained. Upgrade to a small MLP student only if a linear fit visibly
# reorders results vs the teacher.
"""

from __future__ import annotations

import json

import numpy as np

from crate import config
from crate.rank.taste import _EVENTS, TasteModel, _features

_STUDENT = config.MODELS_DIR / "ranker_student.npz"


def _teacher_scores(X: np.ndarray, sims: np.ndarray, teacher: TasteModel) -> np.ndarray:
    taste = 1.0 / (1.0 + np.exp(-np.clip(X @ teacher.w + teacher.b, -30, 30)))
    return config.TASTE_ALPHA * sims + config.TASTE_BETA * taste


def distill() -> np.ndarray | None:
    """Fit a linear student to teacher final scores. Returns student weights or None."""
    if not _EVENTS.exists():
        return None
    X, sims = [], []
    for line in _EVENTS.read_text().splitlines():
        e = json.loads(line)
        q, r = np.array(e["q"], np.float32), np.array(e["r"], np.float32)
        X.append(_features(q, r))
        sims.append(float(q @ r))  # cosine, both normalized
    X = np.array(X, np.float32)
    sims = np.array(sims, np.float32)

    teacher = TasteModel.load()
    if not np.any(teacher.w):
        return None
    target = _teacher_scores(X, sims, teacher)

    # least squares with bias column
    A = np.hstack([X, np.ones((len(X), 1), np.float32)])
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_STUDENT, coef=coef.astype(np.float32))
    return coef


def student_score(query_emb, result_emb) -> float | None:
    if not _STUDENT.exists():
        return None
    coef = np.load(_STUDENT)["coef"]
    feat = np.concatenate([_features(query_emb, result_emb), [1.0]]).astype(np.float32)
    return float(feat @ coef)


if __name__ == "__main__":
    w = distill()
    print("distilled student" if w is not None else "not enough data to distill")

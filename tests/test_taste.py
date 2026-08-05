"""Taste-ranker unit tests — logistic head learns, rerank reorders. Pure numpy."""

import numpy as np

from crate import config
from crate.rank.taste import TasteModel, rerank
from eval.ab_taste import auc


def test_logistic_learns_separable_signal():
    rng = np.random.default_rng(0)
    dim = 2 * config.EMBED_DIM
    # saves cluster in +w direction, skips in -w
    w_true = rng.standard_normal(dim).astype(np.float32)
    X = rng.standard_normal((200, dim)).astype(np.float32)
    y = (X @ w_true > 0).astype(np.float32)
    m = TasteModel(dim).fit(X, y, epochs=300, lr=0.5)
    preds = (X @ m.w + m.b > 0).astype(np.float32)
    assert (preds == y).mean() > 0.9


def test_auc_perfect_and_chance():
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert abs(auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1, 0, 1, 0])) - 0.5) < 1e-9


def test_rerank_untrained_is_similarity_order():
    results = [
        {"id": "a", "score": 0.3, "emb": np.ones(config.EMBED_DIM, np.float32)},
        {"id": "b", "score": 0.9, "emb": np.ones(config.EMBED_DIM, np.float32)},
    ]
    out = rerank(np.ones(config.EMBED_DIM, np.float32), results, model=TasteModel())
    assert [r["id"] for r in out] == ["b", "a"]  # higher similarity first

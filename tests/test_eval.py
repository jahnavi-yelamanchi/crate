"""Eval-metric unit tests — pure numpy, no model weights."""

import numpy as np

from eval.recall_at_k import recall_at_k


def test_recall_perfect_when_text_equals_audio():
    emb = np.eye(6, dtype=np.float32)  # each query's true clip is uniquely closest
    r = recall_at_k(emb, emb, ks=(1, 5))
    assert r["recall@1"] == 1.0
    assert r["recall@5"] == 1.0


def test_recall_at_1_worst_case():
    # text i most similar to a non-matching clip → recall@1 low, recall@n = 1.
    n = 8
    text = np.eye(n, dtype=np.float32)
    audio = np.roll(np.eye(n, dtype=np.float32), 1, axis=0)  # shift matches away
    r = recall_at_k(text, audio, ks=(1, n))
    assert r["recall@1"] == 0.0
    assert r[f"recall@{n}"] == 1.0

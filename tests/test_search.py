"""Retrieval-logic unit tests — the dedup guard, no index/weights needed."""

import numpy as np

from crate import config
from crate.index.search import _is_near_dup


def test_near_dup_flags_identical_vector():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert _is_near_dup(v, [v]) is True


def test_near_dup_passes_distinct_vector():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert _is_near_dup(v, [other]) is False


def test_near_dup_threshold_boundary():
    # cosine just under DEDUP_COSINE must pass; just over must flag.
    base = np.array([1.0, 0.0], dtype=np.float32)
    ang_lo = np.arccos(config.DEDUP_COSINE) * 1.1  # slightly less similar
    ang_hi = np.arccos(config.DEDUP_COSINE) * 0.5  # more similar
    lo = np.array([np.cos(ang_lo), np.sin(ang_lo)], dtype=np.float32)
    hi = np.array([np.cos(ang_hi), np.sin(ang_hi)], dtype=np.float32)
    assert _is_near_dup(lo, [base]) is False
    assert _is_near_dup(hi, [base]) is True

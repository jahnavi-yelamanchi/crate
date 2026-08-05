"""Data-pipeline unit tests — no model weights, safe in CI.

Covers the two things that silently corrupt training: overlapping splits and
label-breaking augmentation.
"""

import numpy as np

from crate import config
from crate.data import pairs
from crate.data.augment import augment
from crate.data.packs import _caption_from_path
from crate.data.preprocess import fix_length


def test_vocab_loads_and_strips_comments():
    terms = pairs.load_vocab()
    assert terms, "producer vocab should be non-empty"
    assert all(not t.startswith("#") for t in terms)
    assert "boom-bap" in terms


def test_fix_length_trims_and_pads():
    n = config.CLIP_SAMPLES
    assert len(fix_length(np.zeros(n + 5000, np.float32))) == n
    assert len(fix_length(np.zeros(1000, np.float32))) == n


def test_augment_preserves_shape_and_is_finite():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)
    for _ in range(10):
        y = augment(x, rng)
        assert y.shape == x.shape and y.dtype == np.float32
        assert np.isfinite(y).all()


def test_caption_from_path_dedups_and_lowercases():
    from pathlib import Path

    root = Path("/packs")
    cap = _caption_from_path(root / "Boom_Bap/drums/dusty_kick_01.wav", root)
    assert "boom" in cap and "kick" in cap
    assert cap == cap.lower()
    assert "01" not in cap.split()  # pure digits dropped


def test_splits_disjoint(tmp_path, monkeypatch):
    # Fake a metadata + preprocessed set, then assert build() splits don't overlap.
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(config, "META_PATH", tmp_path / "metadata.jsonl")
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path / "pairs")
    config.ensure_dirs()

    import json

    with open(config.META_PATH, "w") as f:
        for i in range(50):
            np.save(config.AUDIO_DIR / f"c{i}.npy", np.zeros(10, np.float32))
            f.write(json.dumps({"id": f"c{i}", "path": "x", "text": f"t{i}"}) + "\n")

    counts = pairs.build(seed=1)
    assert sum(counts.values()) == 50
    ids = {name: {r["id"] for r in pairs.load_split(name)} for name in counts}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["heldout"])
    assert ids["val"].isdisjoint(ids["heldout"])

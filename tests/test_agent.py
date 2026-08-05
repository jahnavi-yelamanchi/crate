"""Agent routing tests — pure numpy, no Demucs/index/weights."""

import numpy as np

from crate import config
from crate.agent.router import route
from crate.agent.stems import STEM_MAP


def test_route_text():
    assert route("dusty boom-bap break")["modality"] == "text"


def test_route_dense_long_audio_is_decomposable_track():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(config.SAMPLE_RATE * 5).astype(np.float32)  # 5s, dense
    d = route(noise)
    assert d["modality"] == "track" and d["decomposable"] is True


def test_route_short_tone_is_hum_not_decomposable():
    t = np.linspace(0, 1, config.SAMPLE_RATE, dtype=np.float32)
    tone = np.sin(2 * np.pi * 220 * t)  # 1s pure tone → simple + short
    d = route(tone)
    assert d["modality"] == "hum" and d["decomposable"] is False


def test_stem_map_folds_vocals_into_texture():
    assert STEM_MAP["other"] == "texture"
    assert STEM_MAP["vocals"] == "texture"
    assert set(STEM_MAP.values()) == {"drums", "bass", "texture"}

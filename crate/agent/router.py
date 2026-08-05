"""Query router — first node in the agent graph.

Decides how a query enters the one embedding space:
  - text        → describe it
  - hum         → short/simple audio, treat as a single-sound query
  - track       → richer audio worth decomposing into stems (crate builder)

Hum-vs-track is a heuristic (duration + spectral flatness): a hum is short and
tonally simple, a track is longer and denser. Wrong guess only changes whether we
offer stem decomposition — retrieval works either way.
"""

from __future__ import annotations

import numpy as np

from crate import config

TRACK_MIN_SECONDS = 4.0        # shorter than this → treat as a hum, not a track
DENSITY_THRESHOLD = 0.35       # spectral flatness above this → dense/polyphonic


def route(query) -> dict:
    """Return {'modality': 'text'|'hum'|'track', 'decomposable': bool}."""
    if isinstance(query, str):
        return {"modality": "text", "decomposable": False}

    wav = np.asarray(query, np.float32).ravel()
    seconds = len(wav) / config.SAMPLE_RATE
    dense = _spectral_flatness(wav) > DENSITY_THRESHOLD
    if seconds >= TRACK_MIN_SECONDS and dense:
        return {"modality": "track", "decomposable": True}
    return {"modality": "hum", "decomposable": False}


def _spectral_flatness(wav: np.ndarray) -> float:
    """Geometric/arithmetic mean of the power spectrum — high for noisy/dense audio."""
    mag = np.abs(np.fft.rfft(wav)) ** 2 + 1e-10
    gmean = np.exp(np.mean(np.log(mag)))
    amean = np.mean(mag)
    return float(gmean / amean)


if __name__ == "__main__":
    # self-check: text routes to text; silence routes to hum (not decomposable).
    assert route("dusty boom-bap")["modality"] == "text"
    assert route(np.zeros(config.SAMPLE_RATE))["decomposable"] is False
    print("router self-check ok")

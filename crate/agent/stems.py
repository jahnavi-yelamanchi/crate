"""Stem decomposition via Demucs (htdemucs, pretrained — we do NOT train it).

Splits a reference track into drums / bass / other(texture). The kit builder then
searches per stem — the taste move nobody makes: searching at stem granularity.
Returns 48k mono stems, fixed-length, ready for embed_audio.
"""

from __future__ import annotations

import numpy as np

from crate import config

# Demucs stem names → the buckets we search on. "vocals" folds into texture.
STEM_MAP = {"drums": "drums", "bass": "bass", "other": "texture", "vocals": "texture"}

_SEPARATOR = None


def _separator():
    global _SEPARATOR
    if _SEPARATOR is None:
        from demucs.api import Separator

        _SEPARATOR = Separator(model="htdemucs")
    return _SEPARATOR


def decompose(wav: np.ndarray) -> dict[str, np.ndarray]:
    """48k mono float32 in → {'drums','bass','texture'} 48k mono, fixed-length."""
    import torch

    from crate.data.preprocess import fix_length

    sep = _separator()
    model_sr = sep.samplerate
    # demucs wants (channels, samples) at its own sample rate
    src = _resample(wav, config.SAMPLE_RATE, model_sr)
    tensor = torch.from_numpy(src).float().unsqueeze(0).repeat(2, 1)  # stereo in
    _, stems = sep.separate_tensor(tensor, sr=model_sr)

    out: dict[str, np.ndarray] = {}
    for name, tens in stems.items():
        bucket = STEM_MAP.get(name)
        if bucket is None:
            continue
        mono = tens.mean(0).cpu().numpy()
        clip = fix_length(_resample(mono, model_sr, config.SAMPLE_RATE))
        out[bucket] = out.get(bucket, np.zeros_like(clip)) + clip  # sum folded buckets
    return out


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav.astype(np.float32)
    import librosa

    return librosa.resample(wav.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr)

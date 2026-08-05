"""Audio augmentation → positive contrastive pairs.

An augmented clip should still match the same text. We use light, label-preserving
transforms: pitch shift, time stretch, add noise, EQ tilt. All operate on a
48k mono float32 array and return the same.
"""

from __future__ import annotations

import numpy as np

from crate import config


def pitch_shift(wav: np.ndarray, semitones: float) -> np.ndarray:
    import librosa

    return librosa.effects.pitch_shift(wav, sr=config.SAMPLE_RATE, n_steps=semitones)


def time_stretch(wav: np.ndarray, rate: float) -> np.ndarray:
    import librosa

    return librosa.effects.time_stretch(wav, rate=rate)


def add_noise(wav: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white noise at a target signal-to-noise ratio."""
    sig_power = np.mean(wav**2) + 1e-9
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape).astype(np.float32)
    return wav + noise


def eq_tilt(wav: np.ndarray, tilt: float) -> np.ndarray:
    """Cheap spectral tilt: brighten (tilt>0) or darken (tilt<0) via a 1st-order shelf.

    Applied in the frequency domain — scale magnitudes by a linear ramp.
    """
    spec = np.fft.rfft(wav)
    ramp = np.linspace(1.0 - tilt, 1.0 + tilt, num=spec.shape[0]).clip(0.1, None)
    return np.fft.irfft(spec * ramp, n=len(wav)).astype(np.float32)


def augment(wav: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply a random label-preserving transform. Result re-fixed to clip length."""
    from crate.data.preprocess import fix_length

    rng = rng or np.random.default_rng()
    choice = rng.integers(0, 4)
    if choice == 0:
        out = pitch_shift(wav, rng.uniform(-2, 2))
    elif choice == 1:
        out = time_stretch(wav, rng.uniform(0.9, 1.1))
    elif choice == 2:
        out = add_noise(wav, rng.uniform(15, 30), rng)
    else:
        out = eq_tilt(wav, rng.uniform(-0.4, 0.4))
    return fix_length(out.astype(np.float32))


if __name__ == "__main__":
    # ponytail self-check: every augment returns the right shape/dtype, stays finite.
    rng = np.random.default_rng(0)
    x = rng.standard_normal(config.CLIP_SAMPLES).astype(np.float32)
    for _ in range(20):
        y = augment(x, rng)
        assert y.shape == x.shape, y.shape
        assert y.dtype == np.float32
        assert np.isfinite(y).all()
    print("augment self-check ok")

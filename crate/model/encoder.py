"""CLAP encoder wrapper — one embedding space for audio and text.

`embed_audio(wav)` and `embed_text(str)` both return L2-normalized float32 vectors
of EMBED_DIM, so cosine == dot product and everything downstream (FAISS, taste,
stems) speaks the same geometry.

Loads base CLAP; if a LoRA adapter dir/repo is given, applies it. torch and
transformers are imported lazily so this module is cheap to import in CI.
"""

from __future__ import annotations

import numpy as np

from crate import config


class ClapEncoder:
    def __init__(self, adapter: str | None = None, device: str | None = None):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ClapProcessor.from_pretrained(config.CLAP_BASE)
        model = ClapModel.from_pretrained(config.CLAP_BASE)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()  # fold LoRA in for fast inference
        self.model = model.to(self.device).eval()

    @staticmethod
    def _norm(x):
        import torch

        return torch.nn.functional.normalize(x, dim=-1)

    def embed_audio(self, wav: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """One clip or a batch → (n, EMBED_DIM) L2-normalized."""
        import torch

        batch = [wav] if isinstance(wav, np.ndarray) else wav
        inputs = self.processor(audios=batch, sampling_rate=config.SAMPLE_RATE,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            feats = self.model.get_audio_features(**inputs)
        return self._norm(feats).cpu().numpy().astype(np.float32)

    def embed_text(self, text: str | list[str]) -> np.ndarray:
        """One string or a batch → (n, EMBED_DIM) L2-normalized."""
        import torch

        batch = [text] if isinstance(text, str) else text
        inputs = self.processor(text=batch, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return self._norm(feats).cpu().numpy().astype(np.float32)


_DEFAULT: ClapEncoder | None = None


def get_encoder(adapter: str | None = None) -> ClapEncoder:
    """Process-wide singleton so the app doesn't reload weights per request.

    Defaults to HF_MODEL_REPO (the fine-tuned adapter) if set, else base CLAP.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ClapEncoder(adapter or (config.HF_MODEL_REPO or None))
    return _DEFAULT

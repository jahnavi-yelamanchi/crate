"""Recall@k on held-out (audio, text) pairs — fine-tuned vs base CLAP.

For each text query, rank all held-out clips by similarity; a hit means the
clip that actually goes with that text lands in the top k. This is the number
that proves the LoRA earned its place on producer vocabulary.

Writes eval/recall.json → consumed by scripts/publish_hf.py for the model card.
"""

from __future__ import annotations

import json

import numpy as np

from crate import config
from crate.data.pairs import load_split


def recall_at_k(text_emb: np.ndarray, audio_emb: np.ndarray, ks=(1, 5, 10)) -> dict:
    """Row i of text matches row i of audio. Returns {'recall@k': value}."""
    sims = text_emb @ audio_emb.T                       # (n_text, n_audio)
    ranks = (-sims).argsort(axis=1)                     # best first
    gold = np.arange(len(text_emb))[:, None]
    hit_rank = (ranks == gold).argmax(axis=1)           # position of the true clip
    return {f"recall@{k}": float(np.mean(hit_rank < k)) for k in ks}


def _embed_split(encoder, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    wavs = [np.load(r["audio_npy"]) for r in rows]
    texts = [r["text"] for r in rows]
    # batch to keep memory sane on 100K
    audio = np.vstack([encoder.embed_audio(wavs[i:i + 32]) for i in range(0, len(wavs), 32)])
    text = np.vstack([encoder.embed_text(texts[i:i + 64]) for i in range(0, len(texts), 64)])
    return text, audio


def run() -> dict:
    from crate.model.encoder import ClapEncoder

    rows = load_split("heldout")
    if not rows:
        raise RuntimeError("empty heldout split — run the data pipeline first")

    out = {}
    base = ClapEncoder(adapter=None)
    tb, ab = _embed_split(base, rows)
    out["base"] = recall_at_k(tb, ab)

    if config.LORA_DIR.exists() or config.HF_MODEL_REPO:
        ft = ClapEncoder(adapter=str(config.LORA_DIR) if config.LORA_DIR.exists()
                         else config.HF_MODEL_REPO)
        tf, af = _embed_split(ft, rows)
        out["finetuned"] = recall_at_k(tf, af)

    config.EVAL_DIR.mkdir(exist_ok=True)
    (config.EVAL_DIR / "recall.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()

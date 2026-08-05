"""LoRA fine-tune of CLAP on producer vocab via symmetric InfoNCE.

We adapt the audio/text projection heads (cheap, publishable as a small adapter)
so "boom-bap" / "tape-saturated" / "rimshot" pull the right clips — words base
CLAP barely knows. Runs on Colab GPU (see notebooks/01_train_lora.ipynb) but the
loop is plain torch and works on M3 MPS with a small subset too.
"""

from __future__ import annotations

import numpy as np

from crate import config
from crate.data.pairs import load_split


def _find_projection_linears(model) -> list[str]:
    """Names of Linear modules inside the audio/text projection heads.

    Discovered at runtime so we don't hard-code names that drift across
    transformers versions.
    """
    import torch.nn as nn

    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and "projection" in name:
            names.append(name)
    return names


def _batches(rows: list[dict], bs: int, rng: np.random.Generator):
    idx = rng.permutation(len(rows))
    for i in range(0, len(rows), bs):
        chunk = [rows[j] for j in idx[i:i + bs]]
        wavs = [np.load(r["audio_npy"]) for r in chunk]
        texts = [r["text"] for r in chunk]
        yield wavs, texts


def train(
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    augment_prob: float = 0.5,
    push_to_hub: bool = False,
    seed: int = 0,
):
    """Fine-tune, save adapter to LORA_DIR, optionally push to HF_MODEL_REPO."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import ClapModel, ClapProcessor

    from crate.data.augment import augment as aug

    rng = np.random.default_rng(seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    processor = ClapProcessor.from_pretrained(config.CLAP_BASE)
    model = ClapModel.from_pretrained(config.CLAP_BASE)

    targets = _find_projection_linears(model)
    if not targets:
        raise RuntimeError("no projection Linear modules found — check transformers version")
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=targets, bias="none",
    ))
    model.print_trainable_parameters()
    model = model.to(device).train()

    train_rows = load_split("train")
    if not train_rows:
        raise RuntimeError("empty train split — run the data pipeline first")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), device=device))

    for epoch in range(epochs):
        total, steps = 0.0, 0
        for wavs, texts in _batches(train_rows, batch_size, rng):
            if augment_prob:  # augmented audio must still match its text
                wavs = [aug(w, rng) if rng.random() < augment_prob else w for w in wavs]
            audio_in = processor(audios=wavs, sampling_rate=config.SAMPLE_RATE,
                                 return_tensors="pt").to(device)
            text_in = processor(text=texts, return_tensors="pt", padding=True).to(device)

            a = torch.nn.functional.normalize(model.get_audio_features(**audio_in), dim=-1)
            t = torch.nn.functional.normalize(model.get_text_features(**text_in), dim=-1)
            logits = scale.exp() * a @ t.t()
            labels = torch.arange(len(wavs), device=device)
            loss = 0.5 * (torch.nn.functional.cross_entropy(logits, labels)
                          + torch.nn.functional.cross_entropy(logits.t(), labels))

            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        print(f"epoch {epoch+1}/{epochs}  loss={total/max(steps,1):.4f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.LORA_DIR)
    print(f"saved adapter → {config.LORA_DIR}")

    if push_to_hub:
        if not config.HF_MODEL_REPO:
            raise RuntimeError("set HF_MODEL_REPO in .env to push")
        model.push_to_hub(config.HF_MODEL_REPO, token=config.HF_TOKEN or True)
        print(f"pushed → https://huggingface.co/{config.HF_MODEL_REPO}")
    return config.LORA_DIR


if __name__ == "__main__":
    train()

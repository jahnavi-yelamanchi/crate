"""Push the fine-tuned LoRA adapter + a model card to the Hugging Face Hub.

Usage:
    python scripts/publish_hf.py                 # uses HF_MODEL_REPO from .env
    python scripts/publish_hf.py user/crate-clap-lora

Pulls recall@k numbers from eval/recall.json if it exists so the card shows
fine-tuned vs base CLAP on producer vocab.
"""

from __future__ import annotations

import json
import sys

from crate import config

CARD = """---
license: mit
base_model: {base}
library_name: peft
tags:
  - clap
  - audio-retrieval
  - music-information-retrieval
  - lora
  - contrastive
pipeline_tag: feature-extraction
---

# Crate — CLAP LoRA for producer vocabulary

LoRA adapter over [`{base}`]({base_url}) fine-tuned so producer terms base CLAP
barely knows — *boom-bap, tape-saturated, rimshot, reese bass* — pull the right
audio. Part of [Crate](https://github.com/jahnavi-yelamanchi/crate), sound-native
search for music producers (hum / drop a track / describe it → one embedding space).

## Results — producer-vocab retrieval (held-out)

{results}

## Usage

```python
from transformers import ClapModel, ClapProcessor
from peft import PeftModel

base = "{base}"
model = PeftModel.from_pretrained(ClapModel.from_pretrained(base), "{repo}").merge_and_unload()
proc = ClapProcessor.from_pretrained(base)
# proc(text=[...]) / proc(audios=[...], sampling_rate=48000) → get_text/audio_features
```

## Training

- Symmetric InfoNCE on (audio, text) pairs from Freesound (CC0/CC-BY) + free packs.
- LoRA on the audio/text projection heads (r=16, alpha=32).
- Label-preserving augmentation (pitch/stretch/noise/EQ) for positive pairs.

See the repo for the full pipeline and eval.
"""


def _results_table() -> str:
    path = config.EVAL_DIR / "recall.json"
    if not path.exists():
        return "_Run `python eval/recall_at_k.py` and re-publish to fill this in._"
    d = json.loads(path.read_text())
    lines = ["| metric | base CLAP | fine-tuned |", "|---|---|---|"]
    for k in sorted(d.get("base", {})):
        lines.append(f"| {k} | {d['base'][k]:.3f} | {d['finetuned'][k]:.3f} |")
    return "\n".join(lines)


def main(repo: str | None = None):
    from huggingface_hub import HfApi, upload_folder

    repo = repo or config.HF_MODEL_REPO
    if not repo:
        sys.exit("set HF_MODEL_REPO in .env or pass a repo id")
    if not config.LORA_DIR.exists():
        sys.exit(f"no adapter at {config.LORA_DIR} — train first")

    token = config.HF_TOKEN or True
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True, token=token)

    card = CARD.format(
        base=config.CLAP_BASE,
        base_url=f"https://huggingface.co/{config.CLAP_BASE}",
        repo=repo,
        results=_results_table(),
    )
    (config.LORA_DIR / "README.md").write_text(card)
    upload_folder(folder_path=str(config.LORA_DIR), repo_id=repo, token=token)
    print(f"published → https://huggingface.co/{repo}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

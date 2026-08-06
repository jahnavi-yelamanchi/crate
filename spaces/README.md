---
title: Crate — Sound-Native Search
emoji: 🎚️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Crate (Hugging Face Space)

Live demo of [Crate](https://github.com/jahnavi-yelamanchi/crate) — hum, drop a
track, or describe a sound, and search a sample library in one embedding space.

## How this Space is wired

- **Docker** Space; runs the FastAPI app (`crate.app.main`) on port 7860.
- On boot, `boot.py` pulls prebuilt artifacts from a Hugging Face **dataset** repo:
  `index.faiss`, `ids.json`, `metadata.jsonl` (built by `notebooks/02_build_index.ipynb`).
- The fine-tuned encoder comes from the **model** repo via `HF_MODEL_REPO`.

## Space secrets / variables (Settings → Variables and secrets)

| name | example | what |
|---|---|---|
| `HF_MODEL_REPO` | `jahnavi-yelamanchi/crate-clap-lora` | fine-tuned LoRA adapter |
| `CRATE_ARTIFACTS_REPO` | `jahnavi-yelamanchi/crate-index` | dataset repo with the 3 artifact files |
| `HF_TOKEN` | *(secret)* | only if the artifact/model repos are private |

## Deploy — one command

```bash
export HF_TOKEN=...   # a write token: https://huggingface.co/settings/tokens
python scripts/deploy_space.py [space-id] [model-repo] [index-repo]
# defaults: jahnaviym/crate  jahnaviym/crate-clap-lora  jahnaviym/crate-index
```

Creates the Space, uploads these three files, and sets the two variables above —
no clicking. Re-run any time to update. The Dockerfile installs the `crate` package
straight from GitHub, so the Space needs nothing else.

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

## Deploy

```bash
# from repo root, with the HF CLI logged in
hf repo create crate --repo-type space --space_sdk docker
git clone https://huggingface.co/spaces/<you>/crate hf-space && cd hf-space
cp -r ../spaces/* .            # this folder + the crate/ package + pyproject
git add . && git commit -m "deploy crate" && git push
```

The Dockerfile installs the package straight from GitHub, so the Space only needs
these files plus the two repo variables above.

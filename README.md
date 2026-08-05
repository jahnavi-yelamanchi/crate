# 🎚️ Crate

**Sound-native search & discovery for music producers.** The thing you want is a *sound*, not a word. Crate lets you **hum it, drop a reference track, or describe it** — all three land in one embedding space, retrieve from a sample library in under a second, and get re-ranked by your own digging taste.

> "dusty boom-bap break but darker" — finally searchable.

---

## Why

Producers dig through 50K-sample libraries and Splice with **text** search. But audio text-metadata is garbage and everyone in the community knows it. Crate makes the query modality **native to audio**:

- **Query-by-example, three ways** — mic hum · dropped track · text — into one contrastive space.
- **Sub-second search** over 100K+ clips via ANN.
- **Self-hosted, quantized audio encoder** — latency benchmarked.
- **Taste ranker** — your saves/skips train a personal reco head; your digging history *is* the signal.
- **Agentic crate builder** — drop a reference track, it splits into drums/bass/texture stems and searches *per stem* to assemble a matching kit. Nobody searches at stem granularity. That's the move.

## System

```
query (hum / track / text)
        │
   query router ──► CLAP encoder (LoRA fine-tuned on producer vocab) ──► one embedding space
        │                                                                      │
        │                                                                 FAISS ANN
        │                                                                      │
   stem decomposer (Demucs) ──► per-stem retrieval ──► kit assembler     taste re-ranker
                                                                    (saves/skips, live session memory)
```

## Artifacts

| What | Where |
|---|---|
| Fine-tuned model (LoRA over CLAP) | Hugging Face — _link after Phase 2_ |
| Live demo (mic + drop + text) | Hugging Face Space — _link after Phase 8_ |
| Code | this repo |

## Stack

CLAP (`laion/clap-htsat-unfused`) + PEFT LoRA · Demucs stems · FAISS · ONNX int8 encoder · FastAPI + vanilla JS · Freesound data. Training on Colab, inference on-device.

## Quickstart

```bash
pip install -e ".[dev]"
cp .env.example .env          # add FREESOUND_KEY + HF_TOKEN
bash scripts/ingest.sh        # pull licensing-clean audio, build pairs
# train on Colab: notebooks/01_train_lora.ipynb  → pushes adapter to HF
python crate/index/build_index.py
uvicorn crate.app.main:app --reload   # open http://localhost:8000
```

## Eval

```bash
python eval/recall_at_k.py     # fine-tuned vs base CLAP, held-out producer vocab
python eval/parity.py          # text-query vs audio-query top-k overlap
python eval/ab_taste.py        # taste-ranked vs similarity-only (AUC)
python eval/failure_tests.py   # bad hums, genre drift, dup flooding
```

Each writes a JSON under `eval/`; `scripts/publish_hf.py` folds `recall.json` into
the model card.

## Project layout

| dir | what |
|---|---|
| `crate/data/` | Freesound + pack ingest, preprocess, augment, pairs |
| `crate/model/` | CLAP encoder wrapper, LoRA fine-tune, int8 quantize |
| `crate/index/` | FAISS build + query router / ANN / dedup |
| `crate/rank/` | taste reco head + distilled re-ranker |
| `crate/agent/` | router → stems → per-stem retrieval → kit assembler |
| `crate/app/` | FastAPI + single-page mic/drop/text UI |
| `notebooks/` | Colab: train LoRA, build index |
| `spaces/` | HF Docker Space deploy |
| `eval/` | recall, parity, taste A/B, failure tests |

## Status

Built phase by phase — see commit history. Landing page: `docs/` (GitHub Pages).
Results table (fine-tuned vs base CLAP on producer vocab) lands after training.

## License

MIT. Audio data is licensing-clean (CC0 / CC-BY from Freesound + free packs); attributions tracked in `data/`.

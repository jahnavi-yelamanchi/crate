# Changelog

## 0.1.0 — initial build

Full pipeline, end to end:

- **Data** — Freesound (license-filtered) + local pack ingest, 48k mono preprocess,
  label-preserving augmentation, disjoint train/val/held-out pairs.
- **Model** — CLAP encoder wrapper (audio+text → one space), LoRA fine-tune via
  symmetric InfoNCE on producer vocab, int8 ONNX export + latency benchmark.
- **Retrieval** — FAISS (flat → IVF), query router (hum/track/text), near-duplicate
  flood guard.
- **Taste** — save/skip logging, logistic reco head, similarity+taste rerank,
  distilled single-head serving path.
- **Agent** — Demucs stem decomposition, per-stem retrieval, kit assembler, live
  session memory feeding taste.
- **App** — FastAPI + single-page mic/drop-zone/text UI, latency dashboard.
- **Eval** — recall@k, text/audio parity, taste A/B (AUC), failure tests.
- **Ship** — Colab training + index notebooks, HF model publish script, HF Docker
  Space, GitHub Pages landing.

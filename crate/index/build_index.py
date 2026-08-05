"""Embed the whole corpus with the fine-tuned encoder → FAISS index.

Inner-product index over L2-normalized vectors = cosine search. Flat is exact and
plenty fast to ~100K; we switch to IVF above that (still sub-second, tiny recall
hit). Saves index.faiss + ids.json (row → clip id) for search.py.
"""

from __future__ import annotations

import json

import numpy as np

from crate import config

IVF_THRESHOLD = 100_000  # flat below this, IVF above


def _corpus() -> list[dict]:
    """Every preprocessed clip with its metadata."""
    if not config.META_PATH.exists():
        raise FileNotFoundError("no metadata.jsonl — run the data pipeline first")
    rows = []
    for line in config.META_PATH.read_text().splitlines():
        rec = json.loads(line)
        npy = config.AUDIO_DIR / f"{rec['id']}.npy"
        if npy.exists():
            rec["audio_npy"] = str(npy)
            rows.append(rec)
    return rows


def build(adapter: str | None = None, batch: int = 32) -> str:
    import faiss

    from crate.model.encoder import ClapEncoder

    rows = _corpus()
    if not rows:
        raise RuntimeError("no preprocessed clips to index")
    enc = ClapEncoder(adapter=adapter or (config.HF_MODEL_REPO or None))

    embs = []
    for i in range(0, len(rows), batch):
        wavs = [np.load(r["audio_npy"]) for r in rows[i:i + batch]]
        embs.append(enc.embed_audio(wavs))
    mat = np.vstack(embs).astype(np.float32)  # already L2-normalized

    dim = mat.shape[1]
    if len(rows) > IVF_THRESHOLD:
        nlist = int(4 * np.sqrt(len(rows)))
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(mat)
        index.nprobe = 16
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(mat)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_PATH))
    config.IDS_PATH.write_text(json.dumps([r["id"] for r in rows]))
    print(f"indexed {len(rows)} clips → {config.INDEX_PATH}")
    return str(config.INDEX_PATH)


if __name__ == "__main__":
    build()

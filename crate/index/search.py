"""Query router + ANN retrieval. Hum, dropped track, or text — one path in.

All three modalities embed into the same CLAP space, so routing is just picking
the right encoder call, then FAISS top-k, then a near-duplicate guard so a pack
full of the-same-loop-12-times doesn't flood the results.
"""

from __future__ import annotations

import json

import numpy as np

from crate import config


class Retriever:
    def __init__(self, adapter: str | None = None):
        import faiss

        from crate.model.encoder import get_encoder

        if not config.INDEX_PATH.exists():
            raise FileNotFoundError("no index — run crate.index.build_index first")
        self.index = faiss.read_index(str(config.INDEX_PATH))
        self.ids = json.loads(config.IDS_PATH.read_text())
        self.meta = self._load_meta()
        self.enc = get_encoder(adapter)

    @staticmethod
    def _load_meta() -> dict[str, dict]:
        out = {}
        for line in config.META_PATH.read_text().splitlines():
            rec = json.loads(line)
            out[rec["id"]] = rec
        return out

    # --- query router: dispatch on input type ---
    def search(self, query, k: int = config.TOP_K, dedup: bool = True) -> list[dict]:
        """str → text query; np.ndarray/list → audio query (hum or track)."""
        if isinstance(query, str):
            q = self.enc.embed_text(query)
        else:
            q = self.enc.embed_audio(query)
        return self._ann(q, k, dedup)

    def _ann(self, q: np.ndarray, k: int, dedup: bool) -> list[dict]:
        # over-fetch when deduping so we still return k after dropping near-dupes
        fetch = k * 4 if dedup else k
        scores, idx = self.index.search(q.reshape(1, -1).astype(np.float32), fetch)
        vecs = self.index.reconstruct_batch(idx[0]) if dedup else None

        results, kept_vecs = [], []
        for rank, (i, s) in enumerate(zip(idx[0], scores[0])):
            if i < 0:
                continue
            if dedup and _is_near_dup(vecs[rank], kept_vecs):
                continue
            rec = self.meta.get(self.ids[i], {})
            results.append({
                "id": self.ids[i],
                "score": float(s),
                "text": rec.get("text", ""),
                "path": rec.get("path", ""),
                "attribution": rec.get("attribution", ""),
                "license": rec.get("license", ""),
            })
            if dedup:
                kept_vecs.append(vecs[rank])
            if len(results) >= k:
                break
        return results


def _is_near_dup(vec: np.ndarray, kept: list[np.ndarray]) -> bool:
    """Cosine (== dot, vectors are normalized) above threshold to any kept result."""
    return any(float(vec @ kv) >= config.DEDUP_COSINE for kv in kept)


_RETRIEVER: Retriever | None = None


def get_retriever() -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever()
    return _RETRIEVER

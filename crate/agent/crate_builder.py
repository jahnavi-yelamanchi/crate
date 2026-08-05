"""Per-stem retrieval → assembled kit.

Decompose a reference into drums/bass/texture, embed each stem, and run a
separate search per stem so the returned kit is coherent across roles — a
matching drum break, a bass that sits under it, a texture that glues it.
"""

from __future__ import annotations

import numpy as np


def build_kit(reference: np.ndarray, k_per_stem: int = 8, taste=True) -> dict[str, list[dict]]:
    """Reference track → {'drums':[...], 'bass':[...], 'texture':[...]}.

    Each list is dedup'd search results, optionally taste-reranked.
    """
    from crate.agent.stems import decompose
    from crate.index.search import get_retriever
    from crate.rank.taste import TasteModel, rerank

    retriever = get_retriever()
    model = TasteModel.load() if taste else None
    stems = decompose(reference)

    kit: dict[str, list[dict]] = {}
    for role, stem_wav in stems.items():
        q_emb = retriever.enc.embed_audio(stem_wav)[0]
        results = retriever.search(stem_wav, k=k_per_stem)
        if taste:
            results = rerank(q_emb, results, model=model)
        kit[role] = [_strip(r) for r in results]
    return kit


def _strip(r: dict) -> dict:
    """Drop the internal embedding before the kit leaves the process."""
    return {k: v for k, v in r.items() if k != "emb" and not isinstance(v, np.ndarray)}

"""Agent graph: router → (stem decomposer → per-stem retriever → kit assembler)
                        ↳ or plain retriever, with live session memory feeding taste.

One entry point (`CrateAgent`) wires the nodes and remembers, per session, the
query embedding and each result's embedding so a save/skip can be turned into a
taste-training event immediately — the kept-vs-skipped signal updates the reco
head live, not in a nightly batch.
"""

from __future__ import annotations

import numpy as np

from crate import config
from crate.agent.router import route
from crate.rank.taste import TasteModel, log_event, rerank, retrain


class CrateAgent:
    def __init__(self):
        from crate.index.search import get_retriever

        self.retriever = get_retriever()
        self.taste = TasteModel.load()
        # per-session: query embedding + {result_id: embedding} for feedback lookup
        self._sessions: dict[str, dict] = {}

    def query(self, session_id: str, q, k: int = config.TOP_K) -> dict:
        decision = route(q)
        if decision["decomposable"]:
            from crate.agent.crate_builder import build_kit

            kit = build_kit(np.asarray(q, np.float32), k_per_stem=max(4, k // 2))
            # kit items ship without embeddings, so taste feedback re-queries a stem
            # rather than acting on the kit directly. Plain search is the live-taste path.
            return {**decision, "kit": kit}

        q_emb = (self.retriever.enc.embed_text(q) if isinstance(q, str)
                 else self.retriever.enc.embed_audio(q))[0]
        results = self.retriever.search(q, k=k)
        results = rerank(q_emb, results, model=self.taste)
        self._remember(session_id, q_emb, {r["id"]: r["emb"] for r in results})
        return {**decision, "results": [_strip(r) for r in results]}

    def feedback(self, session_id: str, result_id: str, saved: bool) -> None:
        """Log a save/skip and retrain taste live off the running session memory."""
        sess = self._sessions.get(session_id)
        if not sess:
            return
        q_emb = sess.get("q_emb")
        r_emb = sess["embs"].get(result_id)
        if q_emb is None or r_emb is None:
            return
        log_event(q_emb, r_emb, saved)
        updated = retrain()          # cheap logistic refit over all events
        if updated is not None:
            self.taste = updated

    def _remember(self, session_id: str, q_emb, embs: dict) -> None:
        s = self._sessions.setdefault(session_id, {"q_emb": None, "embs": {}})
        if q_emb is not None:
            s["q_emb"] = q_emb
        s["embs"].update(embs)


def _strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "emb" and not isinstance(v, np.ndarray)}

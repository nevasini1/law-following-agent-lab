"""Retrieve governing provisions for a case file: BM25, dense (bge-small), and reciprocal-rank fusion."""
from __future__ import annotations

import functools
import re

import numpy as np
from rank_bm25 import BM25Okapi

from lawful.law.provisions import P
from lawful.llm.registry import MODEL_DIR

# Interpretive and precedent entries are retrieved alongside the rule they gloss.
DOC_IDS = [pid for pid, p in P.items() if p.authority != "precedent" and p.source != "sherman_1"]
_tok = functools.partial(re.findall, r"[a-z0-9]+")


def _doc(pid: str) -> str:
    p = P[pid]
    return f"{p.cite}. {pid.replace('_', ' ')}. {p.quote}"


@functools.lru_cache(maxsize=1)
def _bm25():
    return BM25Okapi([_tok(_doc(pid).lower()) for pid in DOC_IDS])


@functools.lru_cache(maxsize=1)
def _encoder():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(MODEL_DIR / "hf" / "bge-small-en-v1.5"), device="cpu")
    return model, model.encode([_doc(pid) for pid in DOC_IDS], normalize_embeddings=True)


def rank_bm25(query: str) -> list[str]:
    scores = _bm25().get_scores(_tok(query.lower()))
    return [DOC_IDS[i] for i in np.argsort(-scores)]


def rank_dense(query: str) -> list[str]:
    model, emb = _encoder()
    q = model.encode(["Represent this sentence for searching relevant passages: " + query], normalize_embeddings=True)
    return [DOC_IDS[i] for i in np.argsort(-(emb @ q[0]))]


def rank_hybrid(query: str, k: int = 60) -> list[str]:
    rrf = {}
    for ranking in (rank_bm25(query), rank_dense(query)):
        for r, pid in enumerate(ranking):
            rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (k + r + 1)
    return sorted(rrf, key=rrf.get, reverse=True)


def render(pids: list[str]) -> str:
    return "\n".join(f"- {P[p].cite}: “{P[p].quote}”" for p in pids)

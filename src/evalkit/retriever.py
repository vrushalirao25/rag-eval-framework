"""A dependency-free BM25 retriever.

This is a deliberate stand-in for a managed vector store (Bedrock Knowledge
Bases, OpenSearch, pgvector). The evaluation layer only depends on the
`search()` contract — ranked chunks with stable IDs — so the retriever can be
swapped without touching any metric.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .corpus import Chunk

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be",
    "by", "with", "it", "its", "this", "that", "do", "does", "we", "our", "us", "what",
    "which", "who", "how", "can", "must", "any", "all", "from", "at", "as", "after",
    "before", "into", "their", "they", "there", "has", "have", "need", "now", "just",
    "much", "many", "long", "often", "allowed", "if", "not", "no", "only", "every",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("Cannot build a retriever over an empty corpus")
        self.chunks = chunks
        self.k1, self.b = k1, b
        docs = [tokenize(f"{c.doc_title} {c.section} {c.text}") for c in chunks]
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = sum(self.doc_lens) / len(docs)
        self.tfs = [Counter(d) for d in docs]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, query_terms: list[str], i: int) -> float:
        tf, dl = self.tfs[i], self.doc_lens[i]
        score = 0.0
        for term in query_terms:
            f = tf.get(term)
            if f:
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf[term] * f * (self.k1 + 1) / denom
        return score

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        terms = tokenize(query)
        scored = sorted(
            ((self._score(terms, i), i) for i in range(len(self.chunks))),
            key=lambda x: -x[0],
        )
        return [
            RetrievedChunk(self.chunks[i], round(s, 4), rank)
            for rank, (s, i) in enumerate(scored[:top_k], start=1)
            if s > 0
        ]

"""The system under test: a policy & compliance Q&A assistant.

Output contract (validated by the deterministic schema check):
    {"answer": str, "citations": [chunk_id, ...], "refused": bool}
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from .llm import LLMClient, extract_json
from .retriever import BM25Retriever, RetrievedChunk, tokenize

REFUSAL_TEXT = "The policy documents do not contain enough information to answer this question."

SYSTEM_PROMPT = """You are a policy and compliance assistant for a financial services company.
Answer ONLY from the policy excerpts provided. Do not use outside knowledge, do not speculate,
and do not give legal advice.

If the excerpts do not contain the information needed, refuse.

Respond with a single JSON object and nothing else:
{"answer": "<concise answer, or a short refusal>",
 "citations": ["<chunk_id of every excerpt that supports the answer>"],
 "refused": <true if the excerpts are insufficient, otherwise false>}"""

USER_TEMPLATE = """Policy excerpts:
{context}

Question: {question}"""


@dataclass
class AssistantResult:
    question: str
    retrieved: list[RetrievedChunk]
    raw_output: str
    parsed: dict | None
    latency_ms: float


def format_context(retrieved: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{r.chunk.chunk_id}] ({r.chunk.doc_title} – {r.chunk.section})\n{r.chunk.text}" for r in retrieved
    )


class PolicyAssistant:
    def __init__(
        self,
        retriever: BM25Retriever,
        llm: LLMClient | None = None,
        top_k: int = 3,
        min_retrieval_score: float = 3.0,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.min_retrieval_score = min_retrieval_score

    def answer(self, question: str) -> AssistantResult:
        start = time.perf_counter()
        retrieved = self.retriever.search(question, top_k=self.top_k)
        if self.llm is None:
            raw = self._extractive_answer(question, retrieved)
        else:
            prompt = USER_TEMPLATE.format(context=format_context(retrieved) or "(no excerpts)", question=question)
            raw = self.llm.complete_with_retry(SYSTEM_PROMPT, prompt)
        latency = (time.perf_counter() - start) * 1000
        return AssistantResult(question, retrieved, raw, extract_json(raw), round(latency, 1))

    def _extractive_answer(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        """Mock generator: pick the best-matching sentence from the top chunk.

        Refuses when retrieval confidence is below a threshold – the simplest
        form of an evidence-sufficiency check.
        """
        if not retrieved or retrieved[0].score < self.min_retrieval_score:
            return json.dumps({"answer": REFUSAL_TEXT, "citations": [], "refused": True})
        top = retrieved[0].chunk
        text = " ".join(line.lstrip("- ").strip() for line in top.text.splitlines() if line.strip())
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
        q_terms = set(tokenize(question))
        best = max(sentences, key=lambda s: len(q_terms & set(tokenize(s))))
        return json.dumps({"answer": best, "citations": [top.chunk_id], "refused": False})

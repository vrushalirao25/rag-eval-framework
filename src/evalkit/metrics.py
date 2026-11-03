"""Deterministic, code-based metrics.

Used wherever the check is objective: these are fast, free, reproducible and
never need calibration. Retrieval and citation metrics rely on stable chunk IDs
labelled in the golden set.
"""
from __future__ import annotations


def schema_valid(parsed: dict | None) -> bool:
    """Output contract: {"answer": str, "citations": list[str], "refused": bool}."""
    return (
        isinstance(parsed, dict)
        and isinstance(parsed.get("answer"), str)
        and bool(parsed["answer"].strip())
        and isinstance(parsed.get("citations"), list)
        and all(isinstance(c, str) for c in parsed["citations"])
        and isinstance(parsed.get("refused"), bool)
    )


def context_precision(retrieved_ids: list[str], expected: list[str]) -> float:
    """Rank-aware precision (average precision over relevant positions).

    1.0 when every relevant chunk is ranked above every irrelevant one;
    penalises relevant chunks that are buried under noise.
    """
    expected_set = set(expected)
    hits, precision_sum = 0, 0.0
    for k, cid in enumerate(retrieved_ids, start=1):
        if cid in expected_set:
            hits += 1
            precision_sum += hits / k
    return round(precision_sum / hits, 3) if hits else 0.0


def context_recall(retrieved_ids: list[str], expected: list[str]) -> float:
    """Share of the required evidence that made it into the context window."""
    if not expected:
        return 1.0
    return round(len(set(expected) & set(retrieved_ids)) / len(set(expected)), 3)


def citation_accuracy(citations: list[str], retrieved_ids: list[str], expected: list[str]) -> float:
    """Share of citations that point to a chunk that was actually retrieved
    AND is labelled as supporting evidence. An answer with no citations scores 0."""
    if not citations:
        return 0.0
    valid = [c for c in citations if c in set(retrieved_ids) and c in set(expected)]
    return round(len(valid) / len(citations), 3)


def refusal_outcome(refused: bool, answerable: bool) -> str:
    if answerable and refused:
        return "false_refusal"
    if not answerable and not refused:
        return "missed_refusal"
    return "correct"

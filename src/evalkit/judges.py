"""LLM-as-a-judge evaluators for dimensions that need interpretation.

Design choices:
- Rubric prompts ask for reasoning *before* the verdict.
- Verdicts are categorical; numeric scores are computed in code, not by the
  judge (e.g. faithfulness = supported claims / total claims).
- Parse failures are recorded as errors, never silently scored.
- `HeuristicJudge` is a deterministic offline stand-in used in mock mode / CI
  to exercise the pipeline. Its scores are NOT a substitute for a real judge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMClient, extract_json
from .retriever import tokenize

PROMPT_DIR = Path(__file__).parent / "prompts"
JUDGE_SYSTEM = "You are a strict, impartial evaluator. Follow the rubric exactly and output valid JSON only."

RELEVANCE_SCORES = {"relevant": 1.0, "partially_relevant": 0.5, "irrelevant": 0.0}
SCOPE_SCORES = {"in_scope": 1.0, "out_of_scope": 0.0}


@dataclass
class JudgeResult:
    score: float | None
    verdict: str | None = None
    reasoning: str = ""
    details: dict = field(default_factory=dict)
    error: str | None = None


def _load(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


class LLMJudge:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.name = llm.name
        self._prompts = {n: _load(n) for n in ("faithfulness", "relevance", "scope")}

    def _ask(self, prompt_name: str, **fields) -> tuple[dict | None, str]:
        prompt = self._prompts[prompt_name].format(**fields)
        raw = self.llm.complete_with_retry(JUDGE_SYSTEM, prompt, max_tokens=1200)
        return extract_json(raw), raw

    def faithfulness(self, question: str, answer: str, context: str) -> JudgeResult:
        data, raw = self._ask("faithfulness", question=question, answer=answer, context=context)
        claims = (data or {}).get("claims")
        if not isinstance(claims, list) or not claims:
            return JudgeResult(None, error=f"unparseable judge output: {raw[:200]}")
        supported = sum(1 for c in claims if isinstance(c, dict) and c.get("supported") is True)
        score = supported / len(claims)
        verdict = "grounded" if score == 1.0 else "partially_grounded" if score > 0 else "ungrounded"
        return JudgeResult(round(score, 3), verdict, data.get("reasoning", ""), {"claims": claims})

    def _categorical(self, prompt_name: str, mapping: dict, **fields) -> JudgeResult:
        data, raw = self._ask(prompt_name, **fields)
        verdict = str((data or {}).get("verdict", "")).strip().lower()
        if verdict not in mapping:
            return JudgeResult(None, error=f"unexpected verdict in judge output: {raw[:200]}")
        return JudgeResult(mapping[verdict], verdict, data.get("reasoning", ""))

    def relevance(self, question: str, answer: str, context: str) -> JudgeResult:
        return self._categorical("relevance", RELEVANCE_SCORES, question=question, answer=answer)

    def scope(self, question: str, answer: str, context: str) -> JudgeResult:
        return self._categorical("scope", SCOPE_SCORES, question=question, answer=answer, context=context)


class HeuristicJudge:
    """Offline stand-in for CI and local smoke tests (lexical overlap based)."""

    name = "heuristic-mock"
    HEDGES = ("typically", "generally", "in most", "i think", "i believe", "you should consult", "probably")

    def faithfulness(self, question: str, answer: str, context: str) -> JudgeResult:
        ans = set(tokenize(answer))
        if not ans:
            return JudgeResult(None, error="empty answer")
        score = len(ans & set(tokenize(context))) / len(ans)
        verdict = "grounded" if score >= 0.9 else "partially_grounded" if score > 0.5 else "ungrounded"
        return JudgeResult(round(score, 3), verdict, "lexical overlap with retrieved context")

    def relevance(self, question: str, answer: str, context: str) -> JudgeResult:
        q = set(tokenize(question))
        overlap = len(q & set(tokenize(answer))) / max(len(q), 1)
        verdict = "relevant" if overlap >= 0.5 else "partially_relevant" if overlap >= 0.2 else "irrelevant"
        return JudgeResult(RELEVANCE_SCORES[verdict], verdict, f"question-term overlap {overlap:.2f}")

    def scope(self, question: str, answer: str, context: str) -> JudgeResult:
        hedged = any(h in answer.lower() for h in self.HEDGES)
        verdict = "out_of_scope" if hedged else "in_scope"
        return JudgeResult(SCOPE_SCORES[verdict], verdict, "hedging-phrase check")

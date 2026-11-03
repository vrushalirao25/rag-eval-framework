"""Tests for the 'fail loudly' guarantees: nothing that went wrong may be reported as a pass."""
import pytest

from evalkit.llm import LLMClient, is_transient
from evalkit.runner import aggregate


def _row(item_id="x", answerable=True, metrics=None, judges=None, error=None):
    return {"id": item_id, "question": "q", "answerable": answerable, "expected_sources": [],
            "retrieved": [], "output": None, "latency_ms": 1.0, "metrics": metrics or {},
            "judges": judges or {}, "refusal_outcome": "error" if error else "correct",
            **({"error": error} if error else {})}


def test_gated_dimension_with_no_measurements_fails():
    summary = aggregate([_row(metrics={"schema_validity": 1.0})], {"faithfulness": 0.9})
    assert summary["dimensions"]["faithfulness"]["status"] == "fail"
    assert not summary["passed"]


def test_ungated_dimension_with_no_measurements_is_na():
    summary = aggregate([_row(metrics={"schema_validity": 1.0})], {})
    assert summary["dimensions"]["faithfulness"]["status"] == "n/a"


def test_judge_error_rate_is_gated():
    judges = {"faithfulness": {"error": "unparseable"}, "answer_relevance": {"error": None}}
    summary = aggregate([_row(judges=judges)], {"max_judge_error_rate": 0.05})
    assert summary["reliability"]["judge_error_rate"]["rate"] == 0.5
    assert not summary["passed"]


def test_item_errors_are_gated_and_excluded_from_latency():
    rows = [_row("ok"), {**_row("bad", error="AccessDeniedException: no model access"), "latency_ms": None}]
    summary = aggregate(rows, {"max_item_error_rate": 0.0})
    assert summary["reliability"]["item_error_rate"]["errors"] == 1
    assert summary["latency_ms"]["mean"] == 1.0
    assert not summary["passed"]


class _AwsError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def test_only_transient_errors_are_retried():
    assert is_transient(_AwsError("ThrottlingException"))
    assert not is_transient(_AwsError("AccessDeniedException"))
    assert not is_transient(ValueError("bad request"))


class _FlakyLLM(LLMClient):
    name = "flaky"

    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def complete(self, system, prompt, max_tokens=800):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "OK"


def test_permanent_error_fails_immediately(monkeypatch):
    monkeypatch.setattr("evalkit.llm.time.sleep", lambda s: None)
    llm = _FlakyLLM([_AwsError("AccessDeniedException")])
    with pytest.raises(_AwsError):
        llm.complete_with_retry("s", "p")
    assert llm.calls == 1


def test_transient_error_is_retried(monkeypatch):
    monkeypatch.setattr("evalkit.llm.time.sleep", lambda s: None)
    llm = _FlakyLLM([_AwsError("ThrottlingException")])
    assert llm.complete_with_retry("s", "p") == "OK"
    assert llm.calls == 2


def test_preflight_ping_raises_on_failure():
    with pytest.raises(_AwsError):
        _FlakyLLM([_AwsError("AccessDeniedException")]).ping()

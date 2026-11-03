import json
from pathlib import Path

from evalkit.assistant import PolicyAssistant
from evalkit.corpus import load_corpus
from evalkit.judges import LLMJudge
from evalkit.llm import LLMClient, extract_json
from evalkit.retriever import BM25Retriever
from evalkit.runner import aggregate, evaluate_item

ROOT = Path(__file__).resolve().parents[1]
RETRIEVER = BM25Retriever(load_corpus(ROOT / "data/corpus"))


class FakeLLM(LLMClient):
    """Returns canned responses so the LLM code paths are tested without network calls."""

    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, system, prompt, max_tokens=800):
        return self.responses.pop(0)


def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}
    assert extract_json("no json here") is None


def test_retriever_ranks_expected_chunk_first():
    hits = RETRIEVER.search("How quickly must a suspected incident be reported to the Security Operations Centre?")
    assert hits[0].chunk.chunk_id == "incident-reporting-policy#reporting-timelines"


def test_faithfulness_score_is_computed_in_code_not_by_judge():
    judge = LLMJudge(FakeLLM([json.dumps({
        "reasoning": "one claim unsupported",
        "score": 0.99,  # judge-reported score must be ignored
        "claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}],
    })]))
    result = judge.faithfulness("q", "a. b.", "ctx")
    assert result.score == 0.5 and result.verdict == "partially_grounded"


def test_unparseable_judge_output_is_an_error_not_a_score():
    judge = LLMJudge(FakeLLM(["I think it's fine"]))
    result = judge.relevance("q", "a", "ctx")
    assert result.score is None and result.error


def test_end_to_end_with_llm_generator_and_judge():
    item = {"id": "t1", "question": "Who must approve a cross-border transfer of customer data?", "answerable": True,
            "expected_sources": ["third-party-risk-policy#data-location"], "reference_answer": "..."}
    generator = FakeLLM([json.dumps({"answer": "The Compliance team and the Data Protection Officer.",
                                     "citations": ["third-party-risk-policy#data-location"], "refused": False})])
    judge = LLMJudge(FakeLLM([
        json.dumps({"reasoning": "ok", "claims": [{"claim": "Compliance and DPO approve", "supported": True}]}),
        json.dumps({"reasoning": "ok", "verdict": "relevant"}),
        json.dumps({"reasoning": "ok", "verdict": "in_scope"}),
    ]))
    row = evaluate_item(item, PolicyAssistant(RETRIEVER, generator).answer(item["question"]), judge)
    assert row["metrics"]["citation_accuracy"] == 1.0
    assert row["metrics"]["faithfulness"] == 1.0
    assert row["refusal_outcome"] == "correct"
    assert aggregate([row], {"faithfulness": 0.9})["passed"]


def test_malformed_generator_output_fails_schema_and_gate():
    item = {"id": "t2", "question": "How often are standard vendors reassessed?", "answerable": True,
            "expected_sources": ["third-party-risk-policy#ongoing-monitoring"], "reference_answer": "..."}
    row = evaluate_item(item, PolicyAssistant(RETRIEVER, FakeLLM(["Every 3 years."])).answer(item["question"]), None)
    assert row["metrics"]["schema_validity"] == 0.0
    assert not aggregate([row], {"schema_validity": 1.0})["passed"]

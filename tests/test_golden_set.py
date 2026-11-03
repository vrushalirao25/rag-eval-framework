"""Guards the golden set itself: a broken label silently corrupts every metric."""
from pathlib import Path

from evalkit.corpus import load_corpus
from evalkit.runner import load_golden

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = load_golden(ROOT / "data/golden/golden_set.jsonl")
CHUNK_IDS = {c.chunk_id for c in load_corpus(ROOT / "data/corpus")}


def test_ids_are_unique():
    ids = [g["id"] for g in GOLDEN]
    assert len(ids) == len(set(ids))


def test_expected_sources_exist_in_corpus():
    missing = {s for g in GOLDEN for s in g["expected_sources"] if s not in CHUNK_IDS}
    assert not missing, f"Golden set references unknown chunks: {missing}"


def test_labels_are_consistent():
    for g in GOLDEN:
        assert {"id", "question", "answerable", "expected_sources", "reference_answer"} <= g.keys()
        if g["answerable"]:
            assert g["expected_sources"], f"{g['id']} is answerable but has no expected sources"
        else:
            assert not g["expected_sources"], f"{g['id']} is unanswerable but lists sources"


ALLOWED_CATEGORIES = {"factual", "paraphrase", "multi_hop", "false_premise",
                      "unanswerable_near_miss", "unanswerable_out_of_domain"}


def test_categories_are_known_and_match_answerability():
    for g in GOLDEN:
        assert g["category"] in ALLOWED_CATEGORIES, g["id"]
        assert g["category"].startswith("unanswerable") == (not g["answerable"]), g["id"]


def test_set_is_large_enough_to_gate_refusal_rates():
    # A refusal-rate gate on a handful of items is statistically meaningless.
    unanswerable = [g for g in GOLDEN if not g["answerable"]]
    assert len(unanswerable) >= 25
    assert sum(g["category"] == "unanswerable_near_miss" for g in unanswerable) >= 10

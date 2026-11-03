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


def test_set_covers_unanswerable_cases():
    assert sum(not g["answerable"] for g in GOLDEN) >= 5

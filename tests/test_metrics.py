from evalkit.metrics import citation_accuracy, context_precision, context_recall, refusal_outcome, schema_valid


def test_schema_valid_accepts_contract():
    assert schema_valid({"answer": "8 years.", "citations": ["a#b"], "refused": False})


def test_schema_valid_rejects_broken_outputs():
    assert not schema_valid(None)
    assert not schema_valid({"answer": "", "citations": [], "refused": False})
    assert not schema_valid({"answer": "x", "citations": "a#b", "refused": False})
    assert not schema_valid({"answer": "x", "citations": [], "refused": "no"})


def test_context_precision_is_rank_aware():
    assert context_precision(["rel", "x", "y"], ["rel"]) == 1.0
    assert context_precision(["x", "rel", "y"], ["rel"]) == 0.5
    assert context_precision(["x", "y"], ["rel"]) == 0.0


def test_context_recall():
    assert context_recall(["a", "b"], ["a", "c"]) == 0.5
    assert context_recall(["a", "c"], ["a", "c"]) == 1.0


def test_citation_accuracy_requires_retrieved_and_expected():
    # cites a chunk that was never retrieved -> hallucinated citation
    assert citation_accuracy(["ghost"], ["a"], ["ghost"]) == 0.0
    assert citation_accuracy(["a", "b"], ["a", "b"], ["a"]) == 0.5
    assert citation_accuracy([], ["a"], ["a"]) == 0.0


def test_refusal_outcomes():
    assert refusal_outcome(refused=True, answerable=True) == "false_refusal"
    assert refusal_outcome(refused=False, answerable=False) == "missed_refusal"
    assert refusal_outcome(refused=True, answerable=False) == "correct"

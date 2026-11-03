"""Traceability: every run must be attributable to an exact dataset, corpus, prompt set and config."""
import json
from pathlib import Path

import yaml

from evalkit.manifest import estimate_cost, prompt_versions, sha256_dir
from evalkit.runner import run

ROOT = Path(__file__).resolve().parents[1]


def test_cost_is_none_without_pricing_never_zero():
    usage = {"calls": 3, "input_tokens": 2000, "output_tokens": 500}
    assert estimate_cost(usage, None) is None
    assert estimate_cost(usage, {"input_per_1k": 0.001, "output_per_1k": None}) is None
    assert estimate_cost(usage, {"input_per_1k": 0.001, "output_per_1k": 0.004}) == 0.004


def test_prompt_versions_cover_assistant_and_all_judges():
    versions = prompt_versions()
    assert {"assistant", "judge_faithfulness", "judge_relevance", "judge_scope"} <= versions.keys()


def test_corpus_hash_changes_when_content_changes(tmp_path):
    (tmp_path / "a.md").write_text("# A\n## S\nx")
    before = sha256_dir(tmp_path)
    (tmp_path / "a.md").write_text("# A\n## S\ny")
    assert sha256_dir(tmp_path) != before


def test_run_writes_manifest_with_hashes_and_category_breakdown(tmp_path):
    cfg = yaml.safe_load((ROOT / "config/eval_config.yaml").read_text())
    cfg["paths"] = {"corpus": str(ROOT / "data/corpus"), "golden_set": str(ROOT / "data/golden/golden_set.jsonl"),
                    "reports": str(tmp_path / "reports")}
    (tmp_path / "config").mkdir()
    config_path = tmp_path / "config" / "eval_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    run(str(config_path), provider="mock", limit=10, fail=False)

    run_dir = next((tmp_path / "reports").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert len(manifest["dataset"]["sha256"]) == 64 and manifest["dataset"]["n_items"] == 10
    assert manifest["corpus"]["n_chunks"] > 0 and manifest["prompts"]["assistant"]
    assert manifest["models"]["generator"]["name"] == "mock"
    assert manifest["cost_usd"]["total"] is None
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "factual" in summary["by_category"]
    assert "Results by Question Category" in (run_dir / "report.md").read_text()

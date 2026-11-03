"""Evaluation runner: golden set → assistant → metrics/judges → report → gate."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import yaml

from .assistant import AssistantResult, PolicyAssistant, format_context
from .corpus import load_corpus
from .diagnostics import DIMENSIONS
from .judges import HeuristicJudge, JudgeResult, LLMJudge
from .llm import get_client
from .manifest import build_manifest
from .metrics import citation_accuracy, context_precision, context_recall, refusal_outcome, schema_valid
from .retriever import BM25Retriever

log = logging.getLogger("evalkit")

# ---------------------------------------------------------------- config / data
def _expand(obj):
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return os.path.expandvars(obj) if isinstance(obj, str) else obj


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing env vars take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | Path) -> dict:
    load_dotenv(Path(path).resolve().parent.parent / ".env")
    with open(path, encoding="utf-8") as f:
        return _expand(yaml.safe_load(f))


def load_golden(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- per-item eval
def _judge_entry(result: JudgeResult) -> dict:
    return {"score": result.score, "verdict": result.verdict, "reasoning": result.reasoning,
            "error": result.error, **({"claims": result.details["claims"]} if "claims" in result.details else {})}


def evaluate_item(item: dict, res: AssistantResult, judge) -> dict:
    retrieved_ids = [r.chunk.chunk_id for r in res.retrieved]
    expected = item.get("expected_sources", [])
    answerable = item["answerable"]
    valid = schema_valid(res.parsed)

    row: dict = {
        "id": item["id"],
        "category": item.get("category"),
        "question": item["question"],
        "answerable": answerable,
        "expected_sources": expected,
        "retrieved": [{"chunk_id": r.chunk.chunk_id, "score": r.score} for r in res.retrieved],
        "output": res.parsed if valid else None,
        "raw_output": None if valid else res.raw_output,
        "latency_ms": res.latency_ms,
        "metrics": {"schema_validity": 1.0 if valid else 0.0},
        "judges": {},
    }
    if answerable:
        row["metrics"]["context_precision"] = context_precision(retrieved_ids, expected)
        row["metrics"]["context_recall"] = context_recall(retrieved_ids, expected)

    if not valid:
        row["refusal_outcome"] = "invalid_output"
        row["metrics"]["refusal_accuracy"] = 0.0
        return row

    refused = res.parsed["refused"]
    outcome = refusal_outcome(refused, answerable)
    row["refusal_outcome"] = outcome
    row["metrics"]["refusal_accuracy"] = 1.0 if outcome == "correct" else 0.0

    # Answer-quality metrics only apply when the assistant actually answered an answerable question.
    if answerable and not refused:
        answer = res.parsed["answer"]
        context = format_context(res.retrieved)
        row["metrics"]["citation_accuracy"] = citation_accuracy(res.parsed["citations"], retrieved_ids, expected)
        for dim, fn in (("faithfulness", judge.faithfulness), ("answer_relevance", judge.relevance),
                        ("scope_adherence", judge.scope)):
            jr = fn(item["question"], answer, context)
            row["judges"][dim] = _judge_entry(jr)
            if jr.score is not None:
                row["metrics"][dim] = jr.score
    return row


# ---------------------------------------------------------------- aggregation
def aggregate(rows: list[dict], thresholds: dict) -> dict:
    dims = {}
    for dim, meta in DIMENSIONS.items():
        values = [r["metrics"][dim] for r in rows if dim in r["metrics"]]
        score = round(mean(values), 3) if values else None
        threshold = thresholds.get(dim)
        if score is None:
            # A gated dimension with nothing measured is not a pass.
            status = "fail" if threshold is not None else "n/a"
        else:
            status = "pass" if threshold is None or score >= threshold else "fail"
        dims[dim] = {**meta, "score": score, "threshold": threshold, "n": len(values), "status": status}

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    # Aggregate refusal accuracy hides the minority class (unanswerable items),
    # so each error type gets its own ceiling.
    refusal = {}
    for key, subset, outcome in (("false_refusal_rate", answerable, "false_refusal"),
                                 ("missed_refusal_rate", unanswerable, "missed_refusal")):
        rate = round(sum(r["refusal_outcome"] == outcome for r in subset) / max(len(subset), 1), 3)
        ceiling = thresholds.get(f"max_{key}")
        refusal[key] = {"rate": rate, "max": ceiling, "n": len(subset),
                        "status": "pass" if ceiling is None or rate <= ceiling else "fail"}
    # Pipeline reliability: averages computed over a handful of surviving items are not trustworthy,
    # so judge failures and crashed items are gated explicitly instead of silently dropped.
    judge_calls = sum(len(r["judges"]) for r in rows)
    judge_errors = sum(1 for r in rows for j in r["judges"].values() if j["error"])
    item_errors = sum(1 for r in rows if r.get("error"))
    reliability = {}
    for key, errors, total in (("judge_error_rate", judge_errors, judge_calls),
                               ("item_error_rate", item_errors, len(rows))):
        rate = round(errors / max(total, 1), 3)
        ceiling = thresholds.get(f"max_{key}")
        reliability[key] = {"rate": rate, "errors": errors, "max": ceiling, "n": total,
                            "status": "pass" if ceiling is None or rate <= ceiling else "fail"}

    # Per-category breakdown: an aggregate can pass while one question type fails badly.
    by_category = {}
    for cat in sorted({r.get("category") or "uncategorised" for r in rows}):
        subset = [r for r in rows if (r.get("category") or "uncategorised") == cat]
        entry = {"n": len(subset)}
        for metric in ("refusal_accuracy", "context_recall", "faithfulness", "citation_accuracy"):
            values = [r["metrics"][metric] for r in subset if metric in r["metrics"]]
            entry[metric] = round(mean(values), 3) if values else None
        by_category[cat] = entry

    latencies = sorted(r["latency_ms"] for r in rows if r.get("latency_ms") is not None)
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))] if latencies else None
    return {
        "dimensions": dims,
        "refusal_breakdown": refusal,
        "reliability": reliability,
        "by_category": by_category,
        "latency_ms": {"mean": round(mean(latencies), 1) if latencies else None, "p95": p95},
        "n_items": len(rows),
        "passed": all(d["status"] != "fail" for d in dims.values())
        and all(v["status"] != "fail" for v in refusal.values())
        and all(v["status"] != "fail" for v in reliability.values()),
    }


# ---------------------------------------------------------------- reporting
def _fmt(v) -> str:
    return "–" if v is None else f"{v:.2f}"


def _failing_examples(rows: list[dict], dim: str, limit: int = 3) -> list[dict]:
    bad = [r for r in rows if dim in r["metrics"] and r["metrics"][dim] < 1.0]
    return sorted(bad, key=lambda r: r["metrics"][dim])[:limit]


def _usage_line(manifest: dict) -> str:
    parts = []
    for role in ("generator", "judge"):
        u = manifest["usage"].get(role)
        parts.append(f"{role}: " + (f"{u['calls']} calls, {u['input_tokens']:,} in / {u['output_tokens']:,} out tokens"
                                    if u else "mock (no tokens)"))
    cost = manifest["cost_usd"]
    if cost["total"] is not None:
        per_item = cost.get("per_answered_item")
        parts.append(f"cost: ${cost['total']:.4f}" + (f" (${per_item:.5f} per answered item)" if per_item else ""))
    else:
        parts.append("cost: not reported (set `pricing` in config)")
    return " · ".join(parts)


def render_markdown(summary: dict, rows: list[dict], run_info: dict, manifest: dict | None = None) -> str:
    icon = {"pass": "✅", "fail": "❌", "n/a": "➖"}
    lines = [
        "# Evaluation Report",
        "",
        f"- **Run:** `{run_info['run_id']}`",
        f"- **Generator:** `{run_info['generator']}` · **Judge:** `{run_info['judge']}`",
        f"- **Golden set:** {summary['n_items']} items · **Gate:** {'PASSED ✅' if summary['passed'] else 'FAILED ❌'}",
    ]
    if manifest:
        git = manifest["git"]
        commit = f"`{git['commit'][:8]}`" + (" (uncommitted changes)" if git["dirty"] else "") if git["commit"] else "not a git checkout"
        lines += [
            f"- **Dataset:** `{manifest['dataset']['sha256'][:12]}` · **Corpus:** `{manifest['corpus']['sha256'][:12]}` "
            f"({manifest['corpus']['n_documents']} docs, {manifest['corpus']['n_chunks']} chunks) · "
            f"**Assistant prompt:** `{manifest['prompts']['assistant'][:12]}` · **Code:** {commit}",
            f"- **Usage:** {_usage_line(manifest)}",
            "- Full configuration, prompt versions and model versions: `manifest.json`",
        ]
    if run_info["mock"]:
        lines += ["", "> ⚠️ Mock mode: extractive generator + heuristic judge. Use to validate the pipeline, "
                  "not to report model quality."]
    lines += ["", "## Scorecard", "",
              "| Dimension | Method | Score | Threshold | n | Status |",
              "|---|---|---|---|---|---|"]
    for d in summary["dimensions"].values():
        lines.append(f"| {d['label']} | {d['method']} | {_fmt(d['score'])} | {_fmt(d['threshold'])} | {d['n']} | {icon[d['status']]} |")
    for key, v in summary["refusal_breakdown"].items():
        label = "Scope – false refusals (answerable questions refused)" if key.startswith("false") \
            else "Scope – missed refusals (unanswerable questions answered)"
        lines.append(f"| {label} | code | {_fmt(v['rate'])} | ≤ {_fmt(v['max'])} | {v['n']} | {icon[v['status']]} |")
    rel_labels = {"judge_error_rate": "Reliability – judge errors (unparseable verdicts)",
                  "item_error_rate": "Reliability – item errors (API failures)"}
    for key, v in summary["reliability"].items():
        lines.append(f"| {rel_labels[key]} | pipeline | {_fmt(v['rate'])} | ≤ {_fmt(v['max'])} | {v['n']} | {icon[v['status']]} |")
    lines += ["", f"Latency mean/p95: **{summary['latency_ms']['mean']} / {summary['latency_ms']['p95']} ms**"]

    lines += ["", "## Results by Question Category", "",
              "| Category | n | Refusal accuracy | Context recall | Faithfulness | Citation accuracy |",
              "|---|---|---|---|---|---|"]
    for cat, c in summary["by_category"].items():
        lines.append(f"| {cat} | {c['n']} | {_fmt(c['refusal_accuracy'])} | {_fmt(c['context_recall'])} | "
                     f"{_fmt(c['faithfulness'])} | {_fmt(c['citation_accuracy'])} |")

    failing = [(k, d) for k, d in summary["dimensions"].items() if d["status"] == "fail"]
    lines += ["", "## Diagnosis", ""]
    for key, v in summary["reliability"].items():
        if v["status"] == "fail":
            lines += [f"### {rel_labels[key]} – {_fmt(v['rate'])} (max {_fmt(v['max'])})",
                      "**Where to look:** scores below are computed on fewer items than expected and are not "
                      "trustworthy. Check `results.jsonl` for the error messages (credentials, throttling, "
                      "judge output format).", ""]
            for r in [r for r in rows if r.get("error")][:3]:
                lines.append(f"- `{r['id']}` {r['error']}")
            lines.append("")
    for key, v in summary["refusal_breakdown"].items():
        if v["status"] == "fail":
            outcome = key.replace("_rate", "")
            lines += [f"### {key.replace('_', ' ').capitalize()} – {_fmt(v['rate'])} (max {_fmt(v['max'])})",
                      f"**Where to look:** {DIMENSIONS['refusal_accuracy']['likely_causes']}", ""]
            for r in [r for r in rows if r.get("refusal_outcome") == outcome][:3]:
                top = r["retrieved"][0] if r["retrieved"] else {"chunk_id": None, "score": 0}
                answer = (r["output"] or {}).get("answer")
                lines.append(f"- `{r['id']}` {r['question']}  \n"
                             f"  top retrieval: `{top['chunk_id']}` (score {top['score']}) · answer: {answer}")
            lines.append("")
    if not failing and summary["passed"]:
        lines.append("All dimensions meet their thresholds.")
    for key, d in failing:
        if d["score"] is None:
            lines += [f"### {d['label']} – not measured (threshold {_fmt(d['threshold'])})",
                      "**Where to look:** no item produced a score for this gated dimension, so it cannot pass. "
                      "Check item and judge errors above.", ""]
            continue
        lines += [f"### {d['label']} – {_fmt(d['score'])} (threshold {_fmt(d['threshold'])})",
                  f"**Where to look:** {d['likely_causes']}", ""]
        for r in _failing_examples(rows, key):
            got = [x["chunk_id"] for x in r["retrieved"]]
            answer = (r["output"] or {}).get("answer", r.get("raw_output"))
            lines.append(f"- `{r['id']}` ({_fmt(r['metrics'][key])}) {r['question']}  \n"
                         f"  expected: `{r['expected_sources']}` · retrieved: `{got}`  \n"
                         f"  answer: {answer}")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- entry point
def run(config_path: str, provider: str | None = None, limit: int | None = None, fail: bool = True) -> int:
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cfg = load_config(config_path)
    root = Path(config_path).resolve().parent.parent
    paths = {k: (root / v) for k, v in cfg["paths"].items()}

    gen_provider = provider or cfg["assistant"]["provider"]
    judge_provider = provider or cfg["judge"]["provider"]
    try:
        gen_llm = get_client(gen_provider, cfg["assistant"].get("model"))
        judge_llm = get_client(judge_provider, cfg["judge"].get("model"))
    except ImportError as exc:
        log.error("preflight failed: provider SDK not installed (%s). Run: pip install -e \".[%s]\"",
                  exc.name, gen_provider if gen_provider != "mock" else judge_provider)
        return 2
    except ValueError as exc:
        log.error("preflight failed: %s", exc)
        return 2

    # Preflight health check: fail fast on credentials/region/model access, before spending on a full run.
    for role, client in (("generator", gen_llm), ("judge", judge_llm)):
        if client is None:
            log.warning("%s: mock mode (no LLM) – results validate the pipeline, not model quality", role)
            continue
        try:
            client.ping()
            log.info("preflight ok: %s=%s", role, client.name)
        except Exception as exc:
            log.error("preflight failed: %s=%s: %s: %s", role, client.name, type(exc).__name__, exc)
            return 2

    chunks = load_corpus(paths["corpus"])
    retriever = BM25Retriever(chunks)
    assistant = PolicyAssistant(retriever, gen_llm, cfg["retrieval"]["top_k"], cfg["assistant"].get("min_retrieval_score", 3.0))
    judge = LLMJudge(judge_llm) if judge_llm else HeuristicJudge()

    golden = load_golden(paths["golden_set"])[: limit or None]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log.info("run %s: %d chunks indexed, %d golden items", run_id, len(chunks), len(golden))

    rows = []
    for i, item in enumerate(golden, start=1):
        try:
            row = evaluate_item(item, assistant.answer(item["question"]), judge)
            log.info("item %s (%d/%d) refusal=%s", item["id"], i, len(golden), row["refusal_outcome"])
        except Exception as exc:
            # Isolate failures: one bad API call must not lose the rest of the run, and must be visible.
            log.error("item %s (%d/%d) failed: %s: %s", item["id"], i, len(golden), type(exc).__name__, exc)
            row = {"id": item["id"], "category": item.get("category"), "question": item["question"],
                   "answerable": item["answerable"], "expected_sources": item.get("expected_sources", []),
                   "retrieved": [], "output": None, "raw_output": None, "latency_ms": None,
                   "metrics": {}, "judges": {}, "refusal_outcome": "error",
                   "error": f"{type(exc).__name__}: {exc}"}
        rows.append(row)

    summary = aggregate(rows, cfg.get("thresholds", {}))
    run_info = {"run_id": run_id, "generator": gen_llm.name if gen_llm else "mock-extractive",
                "judge": judge.name, "mock": gen_llm is None or judge_llm is None}

    manifest = build_manifest(
        run_id=run_id, started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        root=root, paths=paths, cfg=cfg, golden=golden, n_chunks=len(chunks),
        clients={"generator": gen_llm, "judge": judge_llm},
    )
    answered = sum(1 for r in rows if (r.get("output") or {}).get("refused") is False)
    if manifest["cost_usd"]["total"] is not None and answered:
        manifest["cost_usd"]["per_answered_item"] = round(manifest["cost_usd"]["total"] / answered, 6)
    summary["manifest"] = {"dataset_sha256": manifest["dataset"]["sha256"], "corpus_sha256": manifest["corpus"]["sha256"],
                           "prompts": manifest["prompts"], "git_commit": manifest["git"]["commit"]}

    out_dir = paths["reports"] / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"run": run_info, **summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render_markdown(summary, rows, run_info, manifest)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    print("\n" + report)  # the report is program output, not a log line
    log.info("gate=%s artifacts=%s", "PASSED" if summary["passed"] else "FAILED", out_dir)
    return 0 if (summary["passed"] or not fail) else 1

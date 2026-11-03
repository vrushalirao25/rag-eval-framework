"""Run manifest: everything needed to reproduce, audit and compare an evaluation run.

Two runs are only comparable when their dataset, corpus, prompts and config match.
Content hashes make that check mechanical instead of a matter of memory.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

from . import __version__
from .assistant import SYSTEM_PROMPT, USER_TEMPLATE
from .judges import JUDGE_SYSTEM, PROMPT_DIR


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_dir(path: Path, pattern: str = "*.md") -> str:
    digest = hashlib.sha256()
    for file in sorted(Path(path).glob(pattern)):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def prompt_versions() -> dict:
    versions = {"assistant": sha256_text(SYSTEM_PROMPT + USER_TEMPLATE)}
    for prompt in sorted(PROMPT_DIR.glob("*.md")):
        versions[f"judge_{prompt.stem}"] = sha256_text(JUDGE_SYSTEM + prompt.read_text(encoding="utf-8"))
    return versions


def git_state(root: Path) -> dict:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=5)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def estimate_cost(usage: dict, pricing: dict | None) -> float | None:
    """USD cost from token counts. Returns None unless both prices are configured,
    so a missing price is never reported as zero cost."""
    if not pricing:
        return None
    inp, out = pricing.get("input_per_1k"), pricing.get("output_per_1k")
    if inp is None or out is None:
        return None
    return round(usage["input_tokens"] / 1000 * float(inp) + usage["output_tokens"] / 1000 * float(out), 6)


def build_manifest(*, run_id: str, started_at: str, finished_at: str, root: Path, paths: dict, cfg: dict,
                   golden: list[dict], n_chunks: int, clients: dict) -> dict:
    usage, cost = {}, {}
    for role, client in clients.items():
        usage[role] = dict(client.usage) if client else None
        cost[role] = estimate_cost(client.usage, (cfg.get("pricing") or {}).get(role)) if client else None
    known = [c for c in cost.values() if c is not None]

    def rel(p) -> str:
        try:
            return str(Path(p).resolve().relative_to(root.resolve()))
        except ValueError:
            return str(p)

    return {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "evalkit_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_state(root),
        "dataset": {
            "path": rel(paths["golden_set"]),
            "sha256": sha256_file(paths["golden_set"]),
            "n_items": len(golden),
            "categories": dict(Counter(g.get("category") for g in golden)),
        },
        "corpus": {
            "path": rel(paths["corpus"]),
            "sha256": sha256_dir(paths["corpus"]),
            "n_documents": len(list(Path(paths["corpus"]).glob("*.md"))),
            "n_chunks": n_chunks,
        },
        "prompts": prompt_versions(),
        "config": {
            "retrieval": cfg.get("retrieval"),
            "assistant": {k: v for k, v in cfg.get("assistant", {}).items()},
            "judge": {k: v for k, v in cfg.get("judge", {}).items()},
            "thresholds": cfg.get("thresholds"),
        },
        "models": {
            role: ({"name": c.name, "resolved_model": c.resolved_model} if c else {"name": "mock"})
            for role, c in clients.items()
        },
        "usage": usage,
        "cost_usd": {**cost, "total": round(sum(known), 6) if known else None, "per_answered_item": None}  # set by the runner,
    }

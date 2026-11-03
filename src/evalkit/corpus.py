"""Load markdown policy documents and split them into section-level chunks.

Each `## Heading` becomes one chunk with a stable ID of the form
`<doc-file-stem>#<heading-slug>`. Stable chunk IDs are what make
deterministic retrieval and citation metrics possible: the golden set
references the same IDs the retriever returns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _chunk_document(path: Path) -> list[Chunk]:
    doc_id = path.stem
    doc_title = doc_id
    section: str | None = None
    lines: list[str] = []
    chunks: list[Chunk] = []

    def emit() -> None:
        body = "\n".join(lines).strip()
        if section and body:
            chunks.append(Chunk(f"{doc_id}#{slugify(section)}", doc_id, doc_title, section, body))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            doc_title = line[2:].strip()
        elif line.startswith("## "):
            emit()
            section, lines = line[3:].strip(), []
        else:
            lines.append(line)
    emit()
    return chunks


def load_corpus(corpus_dir: str | Path) -> list[Chunk]:
    paths = sorted(Path(corpus_dir).glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No .md documents found in {corpus_dir}")
    return [chunk for p in paths for chunk in _chunk_document(p)]

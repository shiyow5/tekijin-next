"""Compute and store dense embeddings for the corpus (index-time side of C3).

Reads rows that carry an ``embedding`` column — employee profiles, questions,
answers, documents — encodes their text with an :class:`Embedder`, and writes
the vectors back into the pgvector columns. This is the counterpart to
:mod:`tekijin.retrieval.dense`, which reads those vectors at query time.

The function is embedder-agnostic (dependency-injected), so tests exercise it
with a deterministic fake, while ``scripts/embed_fixtures.py`` wires in the real
:class:`SentenceTransformerEmbedder`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.ingest.chunking import passage_text
from tekijin.models.tables import (
    Answer,
    Chunk,
    DailyReport,
    Document,
    EmployeeProfile,
    Question,
)
from tekijin.retrieval.embedding import PASSAGE, Embedder


def _document_text(row: Document) -> str:
    return f"{row.title or ''}\n{row.body or ''}".strip()


def _daily_text(row: DailyReport) -> str:
    # #433: index the report's PROBLEM (issue) and ACTIVITY (content) — that is
    # where a daily report's tacit knowledge lives, so a question can retrieve it.
    return f"{row.issue or ''}\n{row.content or ''}".strip()


# (logical name, model, text extractor). Order is irrelevant; each is independent.
# ``Any`` for the model so mypy allows the pgvector ``.embedding`` column access.
_SPECS: tuple[tuple[str, Any, Callable[[Any], str | None]], ...] = (
    ("employee_profiles", EmployeeProfile, lambda r: r.description),
    ("questions", Question, lambda r: r.body),
    ("answers", Answer, lambda r: r.body),
    ("documents", Document, _document_text),
    # #433: daily reports as a searchable knowledge source for System 1.
    ("daily_reports", DailyReport, _daily_text),
    # #70: ingested chunks, so an embedding-model change can re-embed them.
    ("chunks", Chunk, lambda r: passage_text(r.source_item.title, r.text)),
)


def embed_corpus(
    session: Session,
    embedder: Embedder,
    *,
    only_missing: bool = True,
    batch_size: int = 64,
) -> dict[str, int]:
    """Encode and persist embeddings for every embeddable row.

    Args:
        session: Active session; rows are mutated and flushed (the caller
            commits).
        embedder: Passage embedder.
        only_missing: When true (default), skip rows that already have a vector,
            so re-runs only fill gaps. When false, re-embed everything.
        batch_size: Rows per ``encode`` call.

    Returns:
        Number of rows embedded, per logical group.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    counts: dict[str, int] = {}
    for name, model, extract in _SPECS:
        stmt = select(model)
        if only_missing:
            stmt = stmt.where(model.embedding.is_(None))
        rows = list(session.scalars(stmt))

        pending = [(row, text) for row in rows if (text := extract(row)) and text.strip()]
        embedded = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            vectors = embedder.encode([text for _, text in batch], kind=PASSAGE)
            for (row, _text), vector in zip(batch, vectors, strict=True):
                row.embedding = vector
                embedded += 1
        session.flush()
        counts[name] = embedded
    return counts

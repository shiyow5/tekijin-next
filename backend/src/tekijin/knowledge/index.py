"""Embed knowledge articles into their pgvector column (#112).

A draft is stored without a vector; this encodes each article's KCS text (issue,
environment, resolution, cause) and writes it back, the counterpart of
:func:`tekijin.data.knowledge.search_verified`. Embedder-agnostic, so tests use a
deterministic fake and the app and ``scripts/embed_knowledge.py`` the real encoder.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.models.tables import KnowledgeArticle
from tekijin.retrieval.embedding import PASSAGE, Embedder


def article_text(row: KnowledgeArticle) -> str:
    """The text an article is found by: issue, environment, resolution, cause."""

    parts = [row.issue, row.environment, row.resolution, row.cause]
    return "\n".join(p.strip() for p in parts if p and p.strip())


def embed_articles(
    session: Session,
    embedder: Embedder,
    *,
    article_ids: Sequence[int] | None = None,
    only_missing: bool = True,
    batch_size: int = 64,
) -> int:
    """Encode and store vectors; returns how many were written.

    ``article_ids`` limits the run to those articles. ``only_missing`` (default)
    skips articles that already have a vector. Retired articles are never embedded:
    nothing searches them. Drafts are, so verifying one needs no second pass and the
    review queue can count the questions a draft would have answered.
    """

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    stmt = select(KnowledgeArticle).where(KnowledgeArticle.state != "retired")
    if article_ids is not None:
        if not article_ids:
            return 0
        stmt = stmt.where(KnowledgeArticle.id.in_(list(article_ids)))
    if only_missing:
        stmt = stmt.where(KnowledgeArticle.embedding.is_(None))
    pending = [(row, t) for row in session.scalars(stmt) if (t := article_text(row))]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = embedder.encode([t for _, t in batch], kind=PASSAGE)
        for (row, _text), vector in zip(batch, vectors, strict=True):
            row.embedding = vector
    session.flush()
    return len(pending)

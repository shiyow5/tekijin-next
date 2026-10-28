"""Store, list, delete and search ingested source items (#70, R2).

Every ingested thing is a ``source_items`` row whose body is split into embedded
``chunks`` (``ingest/chunking.py``). Wiring chunk search into the question graph is
R3; until then :func:`search_chunks` is the entry point for it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tekijin.ingest.chunking import split_text
from tekijin.models.tables import Chunk, SourceItem
from tekijin.retrieval.embedding import PASSAGE, Embedder

UpsertStatus = Literal["added", "updated", "unchanged"]


@dataclass(frozen=True, slots=True)
class UpsertResult:
    status: UpsertStatus
    item_id: int
    chunks: int


@dataclass(frozen=True, slots=True)
class SourceItemSummary:
    id: int
    provider: str
    title: str | None
    url: str | None
    author_id: int | None
    updated_at: dt.datetime
    chunks: int


@dataclass(frozen=True, slots=True)
class ChunkHit:
    chunk_id: int
    source_item_id: int
    title: str | None
    url: str | None
    text: str
    similarity: float


def content_hash(title: str | None, body: str) -> str:
    return hashlib.sha256(f"{title or ''}\x00{body}".encode()).hexdigest()


def upsert_source_item(
    session: Session,
    embedder: Embedder,
    *,
    org_id: int,
    provider: str,
    external_id: str,
    title: str | None,
    body: str,
    url: str | None,
    author_id: int | None,
    now: dt.datetime,
) -> UpsertResult:
    """Add the item, or update it when ``(org, provider, external_id)`` exists.

    An unchanged title and body is a no-op. Otherwise the chunks are rebuilt and
    re-embedded. Embedding runs before any write, so a failing embedder leaves the
    stored item as it was. The caller commits.
    """

    pieces = split_text(body)
    if not pieces:
        raise ValueError("body must not be blank")
    digest = content_hash(title, body)
    item = _find(session, org_id, provider, external_id)
    if item is not None and item.content_hash == digest:
        return UpsertResult("unchanged", item.id, _count_chunks(session, item.id))

    heading = f"{title}\n" if title else ""
    vectors = embedder.encode([heading + p for p in pieces], kind=PASSAGE)
    status: UpsertStatus = "updated"
    if item is None:
        item, status = _insert(session, org_id, provider, external_id), "added"
    item.title, item.body, item.url = title, body, url
    item.author_id, item.content_hash, item.updated_at = author_id, digest, now
    session.execute(delete(Chunk).where(Chunk.source_item_id == item.id))
    session.add_all(
        Chunk(org_id=org_id, source_item_id=item.id, seq=seq, text=piece, embedding=vector)
        for seq, (piece, vector) in enumerate(zip(pieces, vectors, strict=True))
    )
    session.flush()
    return UpsertResult(status, item.id, len(pieces))


def _find(session: Session, org_id: int, provider: str, external_id: str) -> SourceItem | None:
    return session.scalars(
        select(SourceItem).where(
            SourceItem.org_id == org_id,
            SourceItem.provider == provider,
            SourceItem.external_id == external_id,
        )
    ).first()


def _insert(session: Session, org_id: int, provider: str, external_id: str) -> SourceItem:
    """A placeholder row the caller fills in; a concurrent insert of the same item
    is absorbed at a savepoint and the winner's row is returned instead."""

    try:
        with session.begin_nested():
            created = SourceItem(
                org_id=org_id,
                provider=provider,
                external_id=external_id,
                body="",
                content_hash="",
                updated_at=dt.datetime.min,
            )
            session.add(created)
        return created
    except IntegrityError:
        winner = _find(session, org_id, provider, external_id)
        assert winner is not None  # inserted by the concurrent winner
        return winner


def _count_chunks(session: Session, item_id: int) -> int:
    count = session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.source_item_id == item_id)
    )
    return int(count or 0)


def list_source_items(
    session: Session, *, org_id: int, limit: int = 100
) -> list[SourceItemSummary]:
    """The organization's items, newest first."""

    chunk_count = (
        select(func.count())
        .select_from(Chunk)
        .where(Chunk.source_item_id == SourceItem.id)
        .scalar_subquery()
    )
    rows = session.execute(
        select(SourceItem, chunk_count)
        .where(SourceItem.org_id == org_id)
        .order_by(SourceItem.id.desc())
        .limit(limit)
    ).all()
    return [
        SourceItemSummary(
            id=item.id,
            provider=item.provider,
            title=item.title,
            url=item.url,
            author_id=item.author_id,
            updated_at=item.updated_at,
            chunks=int(count),
        )
        for item, count in rows
    ]


def delete_source_item(session: Session, *, item_id: int, org_id: int) -> bool:
    """Delete one item and its chunks. False when it is not in this organization."""

    result = session.execute(
        delete(SourceItem).where(SourceItem.id == item_id, SourceItem.org_id == org_id)
    )
    return bool(result.rowcount)  # type: ignore[attr-defined]


def search_chunks(
    session: Session, *, org_id: int, query_vec: Sequence[float], top_k: int = 10
) -> list[ChunkHit]:
    """The organization's chunks nearest to ``query_vec``, most similar first."""

    distance = Chunk.embedding.cosine_distance(query_vec).label("distance")
    rows = session.execute(
        select(Chunk, SourceItem.title, SourceItem.url, distance)
        .join(SourceItem, SourceItem.id == Chunk.source_item_id)
        .where(Chunk.org_id == org_id, Chunk.embedding.isnot(None))
        # Ties broken on the id so equal vectors come back in a stable order.
        .order_by(distance, Chunk.id)
        .limit(top_k)
    ).all()
    return [
        ChunkHit(
            chunk_id=chunk.id,
            source_item_id=chunk.source_item_id,
            title=title,
            url=url,
            text=chunk.text,
            similarity=1.0 - float(dist),
        )
        for chunk, title, url, dist in rows
    ]

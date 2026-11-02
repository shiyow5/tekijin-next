"""Search what one organization ingested: dense + BM25 over its chunks (#81, R3).

The older ``HybridRetriever`` reads tables that are not divided by organization, so
a member's question must never go through it. This searches only that member's
organization, and hands the graph ids; the text is read back when an answer is
composed, so a deleted source is not kept alive inside a checkpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.config import get_settings
from tekijin.data.sources import search_chunks
from tekijin.models.tables import Chunk, SourceItem
from tekijin.retrieval.bm25_cache import cached_bm25_index
from tekijin.retrieval.fragments import CitedEvidence, clip_text
from tekijin.retrieval.fusion import rrf

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids agent<-retrieval cycle)
    from tekijin.agent.state import SourceHit

SOURCE_KIND = "source"
# A conversation's title already names its day; its stored time is the ingestion's.
_TITLE_HAS_DATE = "discord"
_PREFIX = "src_"
# Each channel's depth before fusion, and how many chunks survive it.
_POOL = 50
_TOP_CHUNKS = 8
# What one composed answer may read.
_MAX_SOURCES = 6
_MAX_CHARS = 1200


def source_citation_id(item_id: int) -> str:
    """The citation id of a source item, distinct from any older kind's id."""

    return f"{_PREFIX}{item_id}"


@dataclass(frozen=True, slots=True)
class SourceSearchResult:
    hits: list[SourceHit]
    # The best dense similarity. RRF scores carry no absolute meaning, so C5 routes
    # on this instead.
    confidence: float


class ChunkSearch:
    def __init__(self, session: Session, *, rrf_k: int = 60) -> None:
        self._session = session
        self._rrf_k = rrf_k
        self._bm25_weight = get_settings().bm25_weight

    def search(self, query: str, org_id: int, query_vector: Sequence[float]) -> SourceSearchResult:
        dense = search_chunks(self._session, org_id=org_id, query_vec=query_vector, top_k=_POOL)
        rows = self._session.execute(
            select(Chunk.id, Chunk.source_item_id, Chunk.text)
            .where(Chunk.org_id == org_id)
            .order_by(Chunk.id)
        ).all()
        index = cached_bm25_index(f"chunks:{org_id}", ((row.id, row.text) for row in rows))
        fused = rrf(
            [[hit.chunk_id for hit in dense], [id_ for id_, _ in index.search(query, _POOL)]],
            k=self._rrf_k,
            weights=[1.0, self._bm25_weight],
        )[:_TOP_CHUNKS]
        item_of = {row.id: row.source_item_id for row in rows}
        return SourceSearchResult(
            hits=[
                {"chunk_id": id_, "source_item_id": item_of[id_], "score": score}
                for id_, score in cast("list[tuple[int, float]]", fused)
            ],
            confidence=max((hit.similarity for hit in dense), default=0.0),
        )

    def evidence(self, hits: Sequence[SourceHit], org_id: int) -> list[CitedEvidence]:
        """The hits as one piece of evidence per source, best source first.

        A source that was removed after the search, or that is not this
        organization's, is left out."""

        chunk_ids = [hit["chunk_id"] for hit in hits]
        rows = self._session.execute(
            select(Chunk.id, Chunk.text, SourceItem)
            .join(SourceItem, SourceItem.id == Chunk.source_item_id)
            .where(Chunk.id.in_(chunk_ids), Chunk.org_id == org_id)
        ).all()
        found = {row.id: row for row in rows}
        grouped: dict[int, list[str]] = {}
        items: dict[int, SourceItem] = {}
        for chunk_id in chunk_ids:
            if chunk_id not in found:
                continue
            row = found[chunk_id]
            items[row.SourceItem.id] = row.SourceItem
            grouped.setdefault(row.SourceItem.id, []).append(row.text)
        return [
            _evidence(items[item_id], texts)
            for item_id, texts in list(grouped.items())[:_MAX_SOURCES]
        ]


def _evidence(item: SourceItem, texts: list[str]) -> CitedEvidence:
    title = item.title or "資料"
    if item.provider == _TITLE_HAS_DATE:
        label = title
    else:
        label = f"{title}({item.updated_at.date().isoformat()})"
    return CitedEvidence(
        source_id=source_citation_id(item.id),
        kind=SOURCE_KIND,
        text=clip_text(f"{label} {' '.join(texts)}", _MAX_CHARS),
        title=item.title,
        url=item.url,
    )

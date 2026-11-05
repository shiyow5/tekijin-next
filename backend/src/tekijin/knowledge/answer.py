"""Verified knowledge as evidence for the self-answer (#118, R6).

Until #118 the graph answered a member's question from the nearest verified
articles directly, listing their fields without an LLM. Articles that were similar
but did not answer the question were returned too, so this module now only offers
them: :class:`KnowledgeCandidates` finds the verified knowledge of the asker's
organization that the asker may read, and the self-answer composer reads it next
to the documents and conversations and decides what it uses.

An answer that cites an article (``ka_{id}``) also cites what the article was made
from: a document (``src_{id}``, the same id a document answer cites), a Discord
message (a link to it) or a past answer (``qa``). :func:`expand_citations` puts
those right after the article.

The ``knowledge_retrieval_enabled`` flag is read by the caller, not here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.data.knowledge import (
    ArticleDTO,
    ArticleSource,
    article_sources,
    matching_drafts,
    search_verified,
    verified_articles,
)
from tekijin.data.visibility import SourceScope
from tekijin.models.tables import Answer
from tekijin.retrieval.fragments import CitedEvidence, clip_text
from tekijin.retrieval.sources import SOURCE_KIND, source_citation_id

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids agent<-knowledge cycle)
    from tekijin.agent.state import KnowledgeHit

#: Citation id of a knowledge article. Distinct from the older ``ku_`` ids of the
#: company-era knowledge units, which point at a table nothing reads any more.
KNOWLEDGE_CITATION_PREFIX = "ka_"
KNOWLEDGE_KIND = "knowledge"
MESSAGE_KIND = "message"

#: How many verified articles are offered to the composer.
DEFAULT_TOP_K = 3
_MAX_CHARS = 1200
_LABELS = (("issue", "問い"), ("environment", "場面"), ("resolution", "答え"), ("cause", "原因"))


@dataclass(frozen=True, slots=True)
class KnowledgeSearchResult:
    hits: list[KnowledgeHit]
    # The best similarity among ``hits`` (0.0 when there are none).
    confidence: float
    # Drafts that would have answered when no verified article did (the review
    # queue's order).
    draft_ids: list[int] = field(default_factory=list)


def knowledge_citation_id(article_id: int) -> str:
    return f"{KNOWLEDGE_CITATION_PREFIX}{article_id}"


def knowledge_article_id(source_id: str) -> int | None:
    """The article id in a ``ka_{id}`` citation, or ``None`` for any other id."""

    rest = source_id.removeprefix(KNOWLEDGE_CITATION_PREFIX)
    if rest == source_id or not rest.isdigit():
        return None
    return int(rest)


def evidence_text(article: ArticleDTO) -> str:
    """The article's fields, labelled, as the composer reads them."""

    lines = []
    for name, label in _LABELS:
        value = getattr(article, name)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _source_citation(source: ArticleSource, answer_ids: dict[str, str]) -> dict[str, Any] | None:
    if source.source_item_id is not None:
        citation: dict[str, Any] = {
            "source_id": source_citation_id(source.source_item_id),
            "kind": SOURCE_KIND,
        }
        if source.title:
            citation["title"] = source.title
        if source.url:
            citation["url"] = source.url
        return citation
    if source.channel_message_id is not None:
        citation = {"source_id": f"msg_{source.channel_message_id}", "kind": MESSAGE_KIND}
        if source.channel_name:
            citation["title"] = f"#{source.channel_name}"
        if source.url:
            citation["url"] = source.url
        return citation
    if source.question_id is not None and source.question_id in answer_ids:
        return {"source_id": answer_ids[source.question_id], "kind": "qa"}
    return None


def _answer_ids(session: Session, question_ids: set[str]) -> dict[str, str]:
    """The latest recorded answer of each question (what a ``qa`` citation opens)."""

    if not question_ids:
        return {}
    rows = session.execute(
        select(Answer.question_id, Answer.id)
        .where(Answer.question_id.in_(question_ids))
        .order_by(Answer.created_at, Answer.id)
    ).all()
    return {question_id: answer_id for question_id, answer_id in rows}


def _key(citation: Mapping[str, Any]) -> tuple[str, str]:
    return (str(citation["kind"]), str(citation["source_id"]))


def expand_citations(
    citations: Sequence[Mapping[str, Any]], groups: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """``citations`` with each cited article followed by its sources, no duplicates.

    ``groups`` maps an article's citation id to its citation and then those of its
    sources. An article missing from ``groups`` is kept as it was cited."""

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        entries = groups.get(str(citation["source_id"])) or [citation]
        for entry in entries:
            if _key(entry) not in seen:
                seen.add(_key(entry))
                out.append(dict(entry))
    return out


class KnowledgeCandidates:
    """The verified knowledge a member's question may draw on (#118).

    :meth:`search` runs in C4 and hands the graph ids only; :meth:`evidence` and
    :meth:`citations` read the articles again, so an article retired or hidden
    after the search is not used, and its text is never kept in a checkpoint."""

    def __init__(
        self, session: Session, *, min_similarity: float, top_k: int = DEFAULT_TOP_K
    ) -> None:
        self._session = session
        self._min_similarity = min_similarity
        self._top_k = top_k

    def search(
        self, org_id: int, query_vector: Sequence[float], scope: SourceScope
    ) -> KnowledgeSearchResult:
        if not query_vector:
            return KnowledgeSearchResult(hits=[], confidence=0.0)
        found = search_verified(
            self._session,
            org_id=org_id,
            query_vec=query_vector,
            scope=scope,
            top_k=self._top_k,
            min_similarity=self._min_similarity,
        )
        hits: list[KnowledgeHit] = [
            {"article_id": article.id, "similarity": similarity} for article, similarity in found
        ]
        if hits:
            return KnowledgeSearchResult(hits=hits, confidence=max(h["similarity"] for h in hits))
        drafts = matching_drafts(
            self._session,
            org_id=org_id,
            query_vec=query_vector,
            min_similarity=self._min_similarity,
        )
        return KnowledgeSearchResult(hits=[], confidence=0.0, draft_ids=drafts)

    def evidence(
        self, hits: Sequence[KnowledgeHit], org_id: int, scope: SourceScope
    ) -> list[CitedEvidence]:
        ids = [hit["article_id"] for hit in hits]
        articles = verified_articles(self._session, ids, org_id=org_id, scope=scope)
        return [
            CitedEvidence(
                source_id=knowledge_citation_id(article.id),
                kind=KNOWLEDGE_KIND,
                text=clip_text(evidence_text(article), _MAX_CHARS),
                title=article.issue,
            )
            for article_id in ids
            if (article := articles.get(article_id)) is not None
        ]

    def citations(
        self, article_ids: Sequence[int], org_id: int, scope: SourceScope
    ) -> dict[str, list[dict[str, Any]]]:
        """Each article's citation followed by those of its sources, by citation id."""

        articles = verified_articles(self._session, article_ids, org_id=org_id, scope=scope)
        sources = article_sources(self._session, list(articles), org_id=org_id, scope=scope)
        answer_ids = _answer_ids(
            self._session,
            {s.question_id for group in sources.values() for s in group if s.question_id},
        )
        groups: dict[str, list[dict[str, Any]]] = {}
        for article_id in article_ids:
            article = articles.get(article_id)
            if article is None:
                continue
            entries: list[dict[str, Any]] = [
                {
                    "source_id": knowledge_citation_id(article.id),
                    "kind": KNOWLEDGE_KIND,
                    "title": article.issue,
                }
            ]
            for source in sources.get(article.id, []):
                citation = _source_citation(source, answer_ids)
                if citation is not None:
                    entries.append(citation)
            groups[knowledge_citation_id(article.id)] = entries
        return groups

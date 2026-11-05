"""Answer a question from the organization's verified knowledge (#112, R6).

The answer is assembled from the articles' own fields (issue, environment,
resolution, cause), with no LLM, so it says only what an admin verified. It cites
each article (``ka_{id}``) and, after it, what the article was made from: a
document (``src_{id}``, the same id a document answer cites), a Discord message (a
link to it) or a past answer (``qa``).

:func:`answer_from_knowledge` returns ``None`` when nothing verified is close enough,
so the caller routes the question as usual. The ``knowledge_retrieval_enabled`` flag
is read by the caller, not here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.data.knowledge import ArticleDTO, ArticleSource, article_sources, search_verified
from tekijin.data.visibility import SourceScope
from tekijin.models.tables import Answer
from tekijin.retrieval.sources import SOURCE_KIND, source_citation_id

#: Citation id of a knowledge article. Distinct from the older ``ku_`` ids of the
#: company-era knowledge units, which point at a table nothing reads any more.
KNOWLEDGE_CITATION_PREFIX = "ka_"
KNOWLEDGE_KIND = "knowledge"
MESSAGE_KIND = "message"

_INTRO = "確認済みの知識から答えます。"
_LABELS = (("issue", "問い"), ("environment", "場面"), ("resolution", "答え"), ("cause", "原因"))


@dataclass(frozen=True, slots=True)
class KnowledgeAnswer:
    answer: str
    citations: list[dict[str, Any]]
    article_ids: tuple[int, ...]


def knowledge_citation_id(article_id: int) -> str:
    return f"{KNOWLEDGE_CITATION_PREFIX}{article_id}"


def _format(article: ArticleDTO) -> str:
    lines = []
    for field, label in _LABELS:
        value = getattr(article, field)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def compose_knowledge_answer(articles: Sequence[ArticleDTO]) -> str | None:
    """The answer text built from ``articles``, or ``None`` when none has an answer."""

    usable = [a for a in articles if a.resolution]
    if not usable:
        return None
    return _INTRO + "\n\n" + "\n\n".join(_format(a) for a in usable)


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


def knowledge_citations(
    session: Session, articles: Sequence[ArticleDTO], *, org_id: int, scope: SourceScope
) -> list[dict[str, Any]]:
    """Each article's citation, followed by those of its sources (no duplicates)."""

    sources = article_sources(session, [a.id for a in articles], org_id=org_id, scope=scope)
    answer_ids = _answer_ids(
        session,
        {s.question_id for group in sources.values() for s in group if s.question_id},
    )
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for article in articles:
        entries: list[dict[str, Any] | None] = [
            {
                "source_id": knowledge_citation_id(article.id),
                "kind": KNOWLEDGE_KIND,
                "title": article.issue,
            }
        ]
        entries += [_source_citation(s, answer_ids) for s in sources.get(article.id, [])]
        for entry in entries:
            if entry is None:
                continue
            key = (entry["kind"], entry["source_id"])
            if key not in seen:
                seen.add(key)
                citations.append(entry)
    return citations


def answer_from_knowledge(
    session: Session,
    *,
    org_id: int,
    query_vec: Sequence[float],
    scope: SourceScope,
    min_similarity: float,
    top_k: int = 5,
    top_n: int = 3,
) -> KnowledgeAnswer | None:
    """Answer from the best verified articles at or above ``min_similarity``.

    ``None`` means nothing verified answers this, and the question is routed as
    usual. Only knowledge of ``org_id`` that ``scope`` may see is used.
    """

    hits = search_verified(
        session,
        org_id=org_id,
        query_vec=query_vec,
        scope=scope,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    chosen = [a for a, _ in hits if a.resolution][:top_n]
    text = compose_knowledge_answer(chosen)
    if text is None:
        return None
    return KnowledgeAnswer(
        answer=text,
        citations=knowledge_citations(session, chosen, org_id=org_id, scope=scope),
        article_ids=tuple(a.id for a in chosen),
    )

"""Read-only lookup for the company-wide knowledge list (GET /knowledge, #293/#301).

Surfaces accumulated FORMAL knowledge — both past Q&A (an ``answers`` row) and
internal documents — as a single browsable/searchable list. This mirrors the
same two source kinds a self-answer (#291) cites (``schemas.SourceCitation``,
``kind: "qa" | "document"``): ``source_id`` here is exactly the id a citation
carries (``Answer.id`` for ``"qa"``, ``Document.id`` for ``"document"``), so a
chat citation chip and a knowledge-list card can point at the same stable
entity. Deliberately NOT scoped to one asker — the whole point is "someone
else already asked this" or "there's already a document for this"
(#301's "「これに近い話、前にも誰かが聞いてたはず」").

A ``"qa"`` item needs an actual ``answers`` row (not merely an accepted
recommendation) — that is the only place answer TEXT lives, so without it
there is nothing to show as the item's ``summary``.

Named ``knowledge_library`` (not ``knowledge``) to stay clear of
:mod:`tekijin.data.knowledge`, the organization's verified KCS knowledge (#112),
which has its own screen and endpoints (``/articles``).
"""

from __future__ import annotations

import datetime as dt
from itertools import zip_longest
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tekijin.data.dashboard import _self_resolution_rate
from tekijin.models.tables import Answer, Document, Employee, Question


def _qa_items(
    session: Session,
    *,
    q: str | None,
    department: str | None,
    topic: str | None,
    since: dt.date | None,
) -> list[dict[str, Any]]:
    """Past-Q&A knowledge items (``kind="qa"``), newest answer first."""

    stmt = (
        select(
            Answer.id,
            Question.id,
            Question.body,
            Question.topics,
            Question.session_id,
            Answer.body,
            Answer.created_at,
            Employee.name,
            Employee.department,
        )
        .select_from(Answer)
        .join(Question, Question.id == Answer.question_id)
        .join(Employee, Employee.id == Answer.responder_id)
        .order_by(Answer.created_at.desc(), Answer.id.desc())
    )
    if q:
        stmt = stmt.where(Question.body.ilike(f"%{q}%"))
    if department:
        stmt = stmt.where(Employee.department == department)
    if topic:
        stmt = stmt.where(Question.topics.any(topic))  # type: ignore[arg-type]
    if since:
        stmt = stmt.where(Answer.created_at >= since)

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for (
        answer_id,
        qid,
        q_body,
        topics,
        session_id,
        answer_body,
        answered_at,
        name,
        dept,
    ) in session.execute(stmt):
        if qid in seen:  # a question with >1 answers row contributes only the newest
            continue
        seen.add(qid)
        items.append(
            {
                "source_id": answer_id,
                "kind": "qa",
                "title": q_body or "",
                "summary": answer_body or "",
                "topics": list(topics or []),
                "responder_name": name,
                "responder_department": dept,
                "resolved_at": answered_at.isoformat() if answered_at is not None else None,
                "question_id": qid,
                "session_id": session_id,
            }
        )
    return items


def _document_items(
    session: Session, *, q: str | None, since: dt.date | None
) -> list[dict[str, Any]]:
    """Internal-document knowledge items (``kind="document"``), newest first.

    Documents have no department/topic/responder — a ``department`` or ``topic``
    filter is QA-specific, so the caller skips this source entirely when either
    is set rather than matching nothing here.
    """

    stmt = select(Document.id, Document.title, Document.body, Document.updated_at).order_by(
        Document.updated_at.desc(), Document.id.desc()
    )
    if q:
        stmt = stmt.where(Document.title.ilike(f"%{q}%") | Document.body.ilike(f"%{q}%"))
    if since:
        stmt = stmt.where(Document.updated_at >= since)

    return [
        {
            "source_id": doc_id,
            "kind": "document",
            "title": title or "",
            "summary": body or "",
            "topics": [],
            "responder_name": None,
            "responder_department": None,
            "resolved_at": updated_at.isoformat() if updated_at is not None else None,
            "question_id": None,
            "session_id": None,
        }
        for doc_id, title, body, updated_at in session.execute(stmt)
    ]


def list_knowledge(
    session: Session,
    *,
    q: str | None = None,
    department: str | None = None,
    topic: str | None = None,
    since: dt.date | None = None,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Answers + documents, round-robin interleaved (each source newest-first
    within itself), paged, plus a summary.

    ``q`` is a case-insensitive substring match (question body for ``"qa"``,
    title/body for ``"document"``). ``department`` and ``topic`` are QA-specific —
    documents carry neither, so either excludes them. Verified knowledge has its
    own screen (``/articles``, #112). ``since`` bounds each item's own timestamp
    (the ANSWER's for ``"qa"``, ``updated_at`` for ``"document"``) and is the ONLY period
    bound: the matching ``until`` was removed in #394 — no screen ever sent it,
    nothing tested it, and it compared a TIMESTAMP column against a bare date, so
    ``until=<day>`` dropped everything answered after that day's 00:00.
    Re-adding an end bound means a half-open ``< until + 1 day`` (or a timestamp)
    plus the UI that sends it.

    Returns ``(items, total_matching, summary)``: ``total_matching`` is the
    count of items matching the filters above, BEFORE the ``offset``/``limit``
    page cut. ``summary`` reuses the dashboard's self-resolution rate (no new
    aggregation logic for the side panel); ``total_items`` is the GLOBAL count
    of answers + documents (independent of both the
    filters and the page), matching the DoD's "蓄積件数" site-wide stat.
    Per-responder aggregates are deliberately NOT part of this summary — that view
    belongs to ``/dashboard``,
    not a knowledge browser (PR #340 review).
    """

    global_total = (session.scalar(select(func.count(func.distinct(Answer.question_id)))) or 0) + (
        session.scalar(select(func.count()).select_from(Document)) or 0
    )
    summary = {
        "total_items": global_total,
        "self_resolution_rate": _self_resolution_rate(session),
    }

    qa_items = _qa_items(session, q=q, department=department, topic=topic, since=since)
    # Documents carry no department or topic, so either filter excludes them.
    doc_items = [] if (department or topic) else _document_items(session, q=q, since=since)
    matching = _interleave(qa_items, doc_items)
    return matching[offset : offset + limit], len(matching), summary


def _interleave(*sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin the given source lists (qa, document), each source's
    own newest-first order preserved.

    A flat sort by timestamp would bury every document past the first page:
    the seed's documents are all older than its newest answers (docs are
    added far less often than Q&A), so a plain browse of the newest items
    showed only ``kind="qa"`` — no document ever reached the front page
    without a keyword search (PR #340 review follow-up). This mirrors
    ``collect_context_fragments`` (#69) — the
    self-answer composer's own retrieval fragments — which round-robins its
    channels for the same reason: no source should crowd out the others by
    trending newer.
    """

    interleaved: list[dict[str, Any]] = []
    for group in zip_longest(*sources):
        for item in group:
            if item is not None:
                interleaved.append(item)
    return interleaved


def get_qa_detail(session: Session, source_id: str) -> dict[str, Any] | None:
    """Full detail of one past-Q&A knowledge item, keyed by ``Answer.id``.

    The document counterpart already has its own detail viewer at
    ``GET /documents/{doc_id}`` (#143) — this only covers ``kind="qa"``, the
    gap #321's chat citation chip was left non-linked for.
    """

    row = (
        session.execute(
            select(
                Question.id,
                Question.body,
                Question.topics,
                Question.session_id,
                Answer.body,
                Answer.created_at,
                Employee.name,
                Employee.department,
            )
            .select_from(Answer)
            .join(Question, Question.id == Answer.question_id)
            .join(Employee, Employee.id == Answer.responder_id)
            .where(Answer.id == source_id)
        )
    ).first()
    if row is None:
        return None
    qid, q_body, topics, session_id, answer_body, answered_at, name, dept = row
    return {
        "source_id": source_id,
        "kind": "qa",
        "title": q_body or "",
        "summary": answer_body or "",
        "topics": list(topics or []),
        "responder_name": name,
        "responder_department": dept,
        "resolved_at": answered_at.isoformat() if answered_at is not None else None,
        "question_id": qid,
        "session_id": session_id,
    }

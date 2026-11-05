"""An answer written in a Discord request thread, kept as an answer and a case draft (#114).

The requested member answers in the thread. Each of their messages there is stored
as an ``answers`` row, the same row the web's 「回答を書く」 writes (#274), so the
answer is reused like any other. The question and its answers also become one case
draft that cites the question (as the Slack capture does, ``slack_thread.py``). It
waits in the admin's review queue. Until it is checked, a later answer rewrites the
draft's resolution; a checked (verified) or retired article is left alone.

No model is called: the question is the issue and the answers are the resolution.
The admin edits the draft when it needs more shape.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.data.knowledge import (
    REVIEW_STATES,
    SourceRef,
    articles_from_question,
    create_draft,
    update_article,
)
from tekijin.data.writes import create_answer
from tekijin.knowledge.slack_thread import thread_org_id
from tekijin.models.tables import Answer, KnowledgeArticle, Question

#: How much of the answers is kept as the draft's resolution and the source quote.
_RESOLUTION_MAX = 4000
_QUOTE_MAX = 1000


def record_thread_answer(
    session: Session,
    *,
    parties: dict[str, Any],
    body: str,
    embedding: Sequence[float] | None,
    now: dt.datetime,
) -> tuple[str, int | None]:
    """Store ``body`` as the responder's answer and refresh the case draft.

    ``parties`` is what :func:`tekijin.data.messages.thread_parties` returns for the
    accepted hand-off. Returns ``(answer_id, draft_id)``; ``draft_id`` is ``None``
    when no draft was written (no organization to keep it in, or the article was
    already checked).
    """

    question = session.get(Question, parties["question_id"])
    topics = (question.topics if question is not None else None) or []
    answer_id = create_answer(
        session,
        question_id=parties["question_id"],
        responder_id=parties["responder_id"],
        body=body,
        topic=topics[0] if topics else None,
        embedding=embedding,
        created_at=now,
    )
    return answer_id, refresh_case_draft(session, parties=parties, now=now)


def update_thread_answer(
    session: Session,
    *,
    parties: dict[str, Any],
    answer_id: str,
    body: str,
    embedding: Sequence[float] | None,
    now: dt.datetime,
) -> int | None:
    """An answer edited in the thread: replace its text and refresh the draft."""

    answer = session.get(Answer, answer_id)
    if answer is None:
        return None
    answer.body = body
    answer.embedding = list(embedding) if embedding is not None else None
    session.flush()
    return refresh_case_draft(session, parties=parties, now=now)


def refresh_case_draft(
    session: Session, *, parties: dict[str, Any], now: dt.datetime
) -> int | None:
    """Write the responder's answers into the question's case draft, creating it
    if needed. A checked (verified) or retired article is left alone."""

    question_id = parties["question_id"]
    responder_id = parties["responder_id"]
    question = session.get(Question, question_id)
    if question is None or not (question.body or "").strip():
        return None
    org_id = thread_org_id(session, parties["asker_id"])
    if org_id is None:
        return None
    resolution = _resolution(session, question_id, responder_id)
    existing = articles_from_question(session, question_id)
    if not existing and not resolution:
        return None  # every answer was deleted before a draft was made
    if existing:
        article = existing[0]
        # Lock the row before reading its state: a review that verifies it at the
        # same moment must not have this answer slipped into what was verified.
        locked = session.get(
            KnowledgeArticle, article.id, with_for_update=True, populate_existing=True
        )
        if locked is None or locked.state not in REVIEW_STATES:
            return None
        update_article(
            session, article.id, org_id=article.org_id, changes={"resolution": resolution}, now=now
        )
        return article.id
    draft = create_draft(
        session,
        org_id=org_id,
        kind="case",
        issue=question.body or "",
        resolution=resolution,
        author_id=responder_id,
        sources=(SourceRef(question_id=question_id, quote=resolution[:_QUOTE_MAX]),),
        now=now,
    )
    return draft.id


def _resolution(session: Session, question_id: str, responder_id: int) -> str:
    bodies = session.scalars(
        select(Answer.body)
        .where(Answer.question_id == question_id, Answer.responder_id == responder_id)
        .order_by(Answer.created_at, Answer.id)
    ).all()
    text = "\n".join(b.strip() for b in bodies if b and b.strip())
    return text[:_RESOLUTION_MAX]

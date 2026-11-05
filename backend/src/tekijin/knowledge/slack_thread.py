"""A resolved hand-off thread as a record to make knowledge from (#476, #112).

When a participant marks a Slack pair-channel thread solved, the question and its
answer (or the thread's chat) become one record for
:func:`tekijin.knowledge.extract.extract_and_store`. The draft cites the question,
so the article is tied to the hand-off it came from, and re-capturing the same
thread can find the draft it already made.

The organization is the asker's. A Slack account made before organizations existed
has none; it falls back to the organization of the configured Discord server, and
without one there is nowhere to keep the knowledge.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.config import get_settings
from tekijin.data.knowledge import SourceRef
from tekijin.data.messages import thread_parties
from tekijin.knowledge.extract import ExtractionSource
from tekijin.models.tables import Answer, Employee, Message, Organization, Question

#: How much of the thread is kept as the quoted part of the source.
_QUOTE_MAX = 1000


def thread_org_id(session: Session, asker_id: int) -> int | None:
    """The organization a thread's knowledge belongs to, or ``None``."""

    org_id = session.scalar(select(Employee.org_id).where(Employee.id == asker_id))
    if org_id is not None:
        return int(org_id)
    guild_id = get_settings().discord_guild_id
    if not guild_id:
        return None
    found = session.scalar(
        select(Organization.id).where(Organization.discord_guild_id == str(guild_id))
    )
    return int(found) if found is not None else None


def slack_thread_source(
    session: Session, thread_id: int, *, parties: dict | None = None
) -> ExtractionSource | None:
    """Assemble a resolved thread into an :class:`ExtractionSource`, or ``None``.

    ``None`` when there is nothing to read: the thread is not an accepted hand-off
    (``thread_parties`` checks that), the question is gone, no answer or chat exists
    yet, or there is no organization to keep the knowledge in. The text is the
    question plus the captured answers (#274), or the thread's chat when no answer
    was captured. ``parties`` lets a caller that already ran :func:`thread_parties`
    pass the result in.
    """

    if parties is None:
        parties = thread_parties(session, thread_id)
    if parties is None:
        return None
    question_id = parties["question_id"]
    question = session.get(Question, question_id)
    if question is None:
        return None
    question_text = (question.body or "").strip()

    answers = session.scalars(
        select(Answer.body)
        .where(Answer.question_id == question_id, Answer.body.isnot(None))
        .order_by(Answer.created_at)
    ).all()
    parts = [b.strip() for b in answers if b and b.strip()]
    if not parts:
        messages = session.scalars(
            select(Message.body)
            .where(Message.recommendation_id == thread_id)
            .order_by(Message.created_at)
        ).all()
        parts = [m.strip() for m in messages if m and m.strip()]
    if not parts:
        return None

    org_id = thread_org_id(session, parties["asker_id"])
    if org_id is None:
        return None
    answer_text = "\n".join(f"- {p}" for p in parts)
    text = f"質問: {question_text}\n\n回答:\n{answer_text}"
    return ExtractionSource(
        org_id=org_id,
        text=text,
        sources=(SourceRef(question_id=question_id, quote=answer_text[:_QUOTE_MAX]),),
        author_id=parties["responder_id"],
    )

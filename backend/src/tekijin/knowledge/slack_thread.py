"""A resolved hand-off thread as a knowledge-extraction input (#476 Screen 02).

When a participant marks a Slack pair-channel thread solved (a ✅ reaction), the
resolved Q&A becomes a candidate *case*: the question is the 課題, the answer /
follow-up chat is the 打ち手 / 結果. This module assembles that thread into the
same :class:`~tekijin.knowledge.extract.ExtractionSource` the chat pipeline
feeds, so extraction, topic provenance, and idempotent storage are shared
— a Slack thread is just a new ``source_type``.

Provenance-faithful, like the other sources: the unit's topics come from the
QUESTION's precomputed tags (never the model), so the knowledge vocabulary cannot
drift. The source id is derived from the thread id, so re-reacting on the same
thread refreshes the same draft in place rather than creating duplicates.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.data.messages import thread_parties
from tekijin.knowledge.extract import ExtractionSource
from tekijin.models.tables import Answer, Message, Question

#: ``source_type`` stamped on units distilled from a resolved Slack thread.
SLACK_THREAD_SOURCE_TYPE = "slack_thread"


def slack_thread_source(
    session: Session, thread_id: int, *, parties: dict | None = None
) -> ExtractionSource | None:
    """Assemble a resolved thread into an :class:`ExtractionSource`, or ``None``.

    Returns ``None`` when there is nothing to extract — the thread is not an
    accepted hand-off (``thread_parties`` gates on ``outcome == "accepted"`` and
    ``consult_method != "direct"``), the question row is gone, or no answer/chat
    content exists yet. The text is ``質問 …`` + the captured answer bodies (#274),
    falling back to the thread's chat transcript when no answer body was captured,
    so a solved conversation that lives only in chat is still distillable.

    ``parties`` lets a caller that already ran :func:`thread_parties` (e.g. the
    capture gate) pass the result in to avoid a second identical join; when ``None``
    it is fetched here, so a standalone caller still works.
    """

    if parties is None:
        parties = thread_parties(session, thread_id)
    if parties is None:
        return None
    question_id = parties["question_id"]

    row = session.execute(
        select(Question.body, Question.topics).where(Question.id == question_id)
    ).first()
    if row is None:
        return None
    body, topics = row
    question_text = (body or "").strip()

    # The responder's captured answer(s) (#274) are the primary 打ち手/結果 signal.
    answer_bodies = (
        session.execute(
            select(Answer.body)
            .where(Answer.question_id == question_id, Answer.body.isnot(None))
            .order_by(Answer.created_at)
        )
        .scalars()
        .all()
    )
    parts = [b.strip() for b in answer_bodies if b and b.strip()]
    if not parts:
        # No captured answer body — fall back to the thread's chat transcript so a
        # conversation resolved purely in chat is still a candidate case.
        messages = (
            session.execute(
                select(Message.body)
                .where(Message.recommendation_id == thread_id)
                .order_by(Message.created_at)
            )
            .scalars()
            .all()
        )
        parts = [m.strip() for m in messages if m and m.strip()]

    if not parts:
        return None

    answer_text = "\n".join(f"- {p}" for p in parts)
    text = f"質問: {question_text}\n\n回答:\n{answer_text}"
    return ExtractionSource(
        source_type=SLACK_THREAD_SOURCE_TYPE,
        source_id=f"slack_thread_{thread_id}",
        text=text,
        topics=tuple(topics or ()),
    )

"""Read-only lookup for the responder inbox (GET /inbox).

Lists the questions where an employee is the *currently handed-off* responder and
the handoff is still pending. The pending handoff itself lives in the graph
checkpointer keyed by ``session_id`` (never in SQL), so the SQL proxy for
"responder X is being asked and hasn't answered" is a **rank-1 recommendation
with ``outcome IS NULL``**. The question carries the ``session_id`` (persisted at
/ask, #123), so each item can deep-link into ``/answer/{session_id}``.

Rows whose question predates session tracking (``session_id IS NULL`` — seeded
history) are skipped: they have no live handoff to open.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.data.discord_requests import sent_to_responder
from tekijin.models.tables import Employee, Question, Recommendation


def pending_handoffs_for_responder(session: Session, responder_id: int) -> list[dict[str, Any]]:
    """Return the pending handoffs awaiting ``responder_id``, newest first.

    Each item: ``session_id``, ``question_id``, ``question`` (body), ``topics``,
    ``asker`` (int id + name/dept), ``created_at`` (ISO 8601). Deduped by
    ``session_id`` — if a question somehow has more than one pending rank-1 row,
    the most recent one wins.
    """

    stmt = (
        select(
            Recommendation.created_at,
            Question.session_id,
            Question.id,
            Question.body,
            Question.topics,
            Question.asker_id,
            Question.consult_method,
            Employee.name,
            Employee.department,
        )
        .join(Question, Recommendation.question_id == Question.id)
        .join(Employee, Employee.id == Question.asker_id)
        .where(
            Recommendation.employee_id == responder_id,
            Recommendation.rank == 1,
            Recommendation.outcome.is_(None),
            sent_to_responder(),
            Question.session_id.is_not(None),
        )
        .order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
    )

    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for (
        created_at,
        session_id,
        qid,
        body,
        topics,
        asker_id,
        consult_method,
        asker_name,
        asker_dept,
    ) in session.execute(stmt):
        if session_id in seen:
            continue
        seen.add(session_id)
        items.append(
            {
                "session_id": session_id,
                "question_id": qid,
                "question": body or "",
                "topics": list(topics or []),
                "asker_id": asker_id,
                "asker_name": asker_name,
                "asker_dept": asker_dept,
                # The RAW column value — NULL (never chosen) and anything else the
                # bare VARCHAR(32) happens to hold. Normalising here would only be a
                # PARTIAL guard (`or "chat"` catches NULL but not an unknown non-NULL
                # value), and a second, weaker copy of a rule that already lives at
                # the API boundary: `schemas.normalize_consult_method`, applied in
                # routes.py where this dict becomes an `InboxItem` (#427).
                # The responder needs this BEFORE accepting (#245): "直接相談"
                # never opens a chat thread, so it changes what accepting means.
                "consult_method": consult_method,
                "created_at": created_at.isoformat() if created_at is not None else None,
            }
        )
    return items

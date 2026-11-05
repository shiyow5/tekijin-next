"""Read-only lookups backing the notification bell (GET /notifications).

No generic notification table: every notification kind is derived directly
from ``Recommendation`` rows, each durably identified by a specific
``(rank, outcome)`` shape (see each function's docstring). "Seen" state lives
on a dedicated ``Recommendation.<kind>_seen_at`` column per kind (set by the
matching ``data.writes.ack_*_notifications`` function).

Two audiences share this module:

- The **asker** side (``notifications_for_asker``): declined (#E7) and
  accepted (#509) outcomes on their own questions.
- The **responder** side (``pending_request_notifications_for_responder``,
  #509): new incoming requests still awaiting their decision — the same
  "pending" shape ``data.inbox.pending_handoffs_for_responder`` lists, plus
  the "not yet seen in the bell" filter.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tekijin.data.discord_requests import sent_to_responder
from tekijin.models.tables import Employee, Question, Recommendation


def pending_decline_notifications_for_asker(
    session: Session, asker_id: int
) -> list[dict[str, Any]]:
    """Declines the asker hasn't seen yet (newest first).

    Each item: ``kind`` ("declined"), ``id`` (the declined ``Recommendation``
    row's id — also the ack target), ``question_id``, ``session_id``
    (deep-link target, may be ``None`` for pre-session-tracking rows),
    ``message`` (ready-to-render text), ``declined_person_name``,
    ``created_at`` (ISO 8601).

    A decline event is durably identified as a ``rank == 1``,
    ``outcome == 'declined'`` row — ``set_recommendation_outcome`` only ever
    writes the primary recommendation's outcome, and a reroute's next
    candidate gets a FRESH ``rank == 1`` row (see ``agent/nodes.py::reroute``
    / ``c6_score``), so this predicate selects each historical decline event
    exactly once.
    """

    stmt = (
        select(
            Recommendation.id,
            Recommendation.created_at,
            Employee.name,
            Question.id,
            Question.session_id,
        )
        .join(Question, Recommendation.question_id == Question.id)
        .join(Employee, Employee.id == Recommendation.employee_id)
        .where(
            Question.asker_id == asker_id,
            Recommendation.rank == 1,
            Recommendation.outcome == "declined",
            Recommendation.declined_seen_at.is_(None),
        )
        .order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
    )

    items: list[dict[str, Any]] = []
    for rec_id, created_at, declined_name, question_id, session_id in session.execute(stmt):
        items.append(
            {
                "kind": "declined",
                "id": rec_id,
                "question_id": question_id,
                "session_id": session_id,
                "message": f"{declined_name}さんに断られたので次の候補に依頼してください",
                "declined_person_name": declined_name,
                "created_at": created_at.isoformat() if created_at is not None else None,
            }
        )
    return items


def pending_accepted_notifications_for_asker(
    session: Session, asker_id: int
) -> list[dict[str, Any]]:
    """Acceptances the asker hasn't seen yet (newest first, #509).

    Each item: ``kind`` ("accepted"), ``id`` (the accepted ``Recommendation``
    row's id — also the ack target AND the chat thread id, see ``Message``),
    ``question_id``, ``session_id``, ``message``, ``responder_name``,
    ``consult_method`` (raw column value; ``None``/anything but ``"direct"``
    means chat — see ``schemas.normalize_consult_method``), ``created_at``
    (the acceptance time, falling back to the recommendation creation time for
    legacy rows without ``Question.resolved_at``).

    At most one ``Recommendation`` per question ever reaches
    ``outcome == 'accepted'`` (``set_recommendation_outcome`` is a guarded,
    idempotent first-write), so no ``rank == 1`` restriction is needed here
    the way decline/request-received need it to exclude superseded rows.
    """

    stmt = (
        select(
            Recommendation.id,
            func.coalesce(Question.resolved_at, Recommendation.created_at).label("accepted_at"),
            Employee.name,
            Question.id,
            Question.session_id,
            Question.consult_method,
        )
        .join(Question, Recommendation.question_id == Question.id)
        .join(Employee, Employee.id == Recommendation.employee_id)
        .where(
            Question.asker_id == asker_id,
            Recommendation.outcome == "accepted",
            Recommendation.accepted_seen_at.is_(None),
        )
        .order_by(
            func.coalesce(Question.resolved_at, Recommendation.created_at).desc(),
            Recommendation.id.desc(),
        )
    )

    items: list[dict[str, Any]] = []
    for (
        rec_id,
        created_at,
        responder_name,
        question_id,
        session_id,
        consult_method,
    ) in session.execute(stmt):
        items.append(
            {
                "kind": "accepted",
                "id": rec_id,
                "question_id": question_id,
                "session_id": session_id,
                "message": f"{responder_name}さんが依頼を受け取りました",
                "responder_name": responder_name,
                "consult_method": consult_method,
                "created_at": created_at.isoformat() if created_at is not None else None,
            }
        )
    return items


def notifications_for_asker(session: Session, asker_id: int) -> list[dict[str, Any]]:
    """Declined + accepted notifications for the asker, merged newest-first."""

    items = [
        *pending_decline_notifications_for_asker(session, asker_id),
        *pending_accepted_notifications_for_asker(session, asker_id),
    ]
    items.sort(key=lambda item: item["created_at"] or "", reverse=True)
    return items


def pending_request_notifications_for_responder(
    session: Session, responder_id: int
) -> list[dict[str, Any]]:
    """Incoming requests the responder hasn't seen in the bell yet (#509).

    Same "still pending" shape ``data.inbox.pending_handoffs_for_responder``
    lists (``rank == 1``, ``outcome IS NULL``, a live ``session_id``), plus
    ``request_seen_at IS NULL`` — including that function's ``session_id``
    dedup (see its docstring): if a question somehow has more than one
    pending rank-1 row, only the most recent one is surfaced here, so the
    bell's count never exceeds what ``/inbox`` would show for the same
    question. Each item: ``kind`` ("request_received"), ``id`` (ack target),
    ``question_id``, ``session_id`` (deep-links to the inbox item),
    ``message``, ``asker_name``, ``created_at``.
    """

    stmt = (
        select(
            Recommendation.id,
            Recommendation.created_at,
            Employee.name,
            Question.id,
            Question.session_id,
        )
        .join(Question, Recommendation.question_id == Question.id)
        .join(Employee, Employee.id == Question.asker_id)
        .where(
            Recommendation.employee_id == responder_id,
            Recommendation.rank == 1,
            Recommendation.outcome.is_(None),
            sent_to_responder(),
            Recommendation.request_seen_at.is_(None),
            Question.session_id.is_not(None),
        )
        .order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
    )

    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for rec_id, created_at, asker_name, question_id, session_id in session.execute(stmt):
        if session_id in seen:
            continue
        seen.add(session_id)
        items.append(
            {
                "kind": "request_received",
                "id": rec_id,
                "question_id": question_id,
                "session_id": session_id,
                "message": f"{asker_name}さんから新しい依頼が届きました",
                "asker_name": asker_name,
                "created_at": created_at.isoformat() if created_at is not None else None,
            }
        )
    return items

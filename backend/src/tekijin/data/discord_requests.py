"""Hand-offs sent from Discord, and what the bot still has to post (#114).

The API process never talks to Discord: the bot token stays with the bot. Whatever
must appear in a request thread is queued in ``discord_outbox`` and the bot fetches
and delivers it (``GET /internal/discord/outbox``), then reports the result.

* :func:`create_request` records that a recommendation was sent from Discord and
  queues the thread for it. It happens once per recommendation.
* :func:`enqueue_post` queues one message for a request's thread. A post is offered
  to the bot only after the thread exists, so it cannot overtake the thread.
* :func:`record_delivery` stores the bot's result. A request's result names the
  thread, which is how messages written there are matched back to the hand-off.

Callers own the transaction.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, exists, not_, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tekijin.models.tables import (
    DiscordOutbox,
    DiscordRequest,
    DiscordThreadMessage,
    Question,
    Recommendation,
)

#: Every session ``/ask`` starts on Discord has this prefix (``api/discord_routes.py``).
DISCORD_SESSION_PREFIX = "discord_"

REQUEST = "request"
POST = "post"

#: How often a delivery may fail before the item is no longer offered.
MAX_ATTEMPTS = 5


def sent_to_responder() -> ColumnElement[bool]:
    """The pending rank-1 recommendation was actually sent (#114).

    ``/ask`` on Discord keeps its hand-off paused until the asker presses
    「依頼する」, so until then the recommended member must not see it as a request
    (the inbox and the request notifications use this). A hand-off asked on the web
    counts as sent, as before.
    """

    return or_(
        not_(Question.session_id.startswith(DISCORD_SESSION_PREFIX, autoescape=True)),
        exists().where(DiscordRequest.recommendation_id == Recommendation.id),
    )


@dataclass(frozen=True, slots=True)
class OutboxItem:
    id: int
    kind: str
    recommendation_id: int
    #: The request's thread; ``None`` for a ``request`` (the bot creates it).
    thread_id: str | None
    payload: dict[str, Any]


def create_request(
    session: Session, *, recommendation_id: int, session_id: str, payload: dict[str, Any]
) -> bool:
    """Record a request and queue its thread. ``False`` when it already exists."""

    if session.get(DiscordRequest, recommendation_id) is not None:
        return False
    try:
        with session.begin_nested():
            session.add(DiscordRequest(recommendation_id=recommendation_id, session_id=session_id))
            session.flush()
            session.add(
                DiscordOutbox(kind=REQUEST, recommendation_id=recommendation_id, payload=payload)
            )
            session.flush()
    except IntegrityError:
        return False  # a concurrent press created it first
    return True


def request_for_recommendation(session: Session, recommendation_id: int) -> DiscordRequest | None:
    return session.get(DiscordRequest, recommendation_id)


def question_was_sent(session: Session, recommendation_id: int) -> bool:
    """True when another recommendation of the same question was sent from Discord."""

    question_id = session.scalar(
        select(Recommendation.question_id).where(Recommendation.id == recommendation_id)
    )
    if question_id is None:
        return False
    return (
        session.scalar(
            select(DiscordRequest.recommendation_id)
            .join(Recommendation, Recommendation.id == DiscordRequest.recommendation_id)
            .where(Recommendation.question_id == question_id)
            .limit(1)
        )
        is not None
    )


def request_for_thread(session: Session, thread_id: str) -> DiscordRequest | None:
    return session.scalars(
        select(DiscordRequest).where(DiscordRequest.thread_id == thread_id)
    ).first()


def seen_message(session: Session, message_id: str) -> DiscordThreadMessage | None:
    return session.get(DiscordThreadMessage, message_id)


def captured_message_ids(session: Session, recommendation_id: int) -> list[str]:
    """The thread messages of a request that are kept as chat messages or answers."""

    return list(
        session.scalars(
            select(DiscordThreadMessage.message_id).where(
                DiscordThreadMessage.recommendation_id == recommendation_id,
                or_(
                    DiscordThreadMessage.message_row_id.is_not(None),
                    DiscordThreadMessage.answer_id.is_not(None),
                ),
            )
        )
    )


def mark_message_seen(
    session: Session,
    message_id: str,
    recommendation_id: int,
    *,
    message_row_id: int | None = None,
    answer_id: str | None = None,
) -> bool:
    """Record a thread message as handled. ``False`` when it already was."""

    try:
        with session.begin_nested():
            session.add(
                DiscordThreadMessage(
                    message_id=message_id,
                    recommendation_id=recommendation_id,
                    message_row_id=message_row_id,
                    answer_id=answer_id,
                )
            )
            session.flush()
    except IntegrityError:
        return False
    return True


def mark_accepted(session: Session, recommendation_id: int, now: dt.datetime) -> None:
    """Stamp when the request was accepted (first stamp wins)."""

    request = session.get(DiscordRequest, recommendation_id)
    if request is not None and request.accepted_at is None:
        request.accepted_at = now
        session.flush()


def enqueue_post(session: Session, recommendation_id: int, text: str) -> bool:
    """Queue ``text`` for the request's thread. ``False`` when there is no request."""

    if session.get(DiscordRequest, recommendation_id) is None:
        return False
    session.add(
        DiscordOutbox(kind=POST, recommendation_id=recommendation_id, payload={"text": text})
    )
    session.flush()
    return True


def pending_outbox(session: Session, *, limit: int) -> list[OutboxItem]:
    """Undelivered items the bot can deliver now, oldest first."""

    rows = session.execute(
        select(DiscordOutbox, DiscordRequest.thread_id)
        .join(DiscordRequest, DiscordRequest.recommendation_id == DiscordOutbox.recommendation_id)
        .where(
            DiscordOutbox.delivered_at.is_(None),
            DiscordOutbox.attempts < MAX_ATTEMPTS,
            or_(DiscordOutbox.kind == REQUEST, DiscordRequest.thread_id.is_not(None)),
        )
        .order_by(DiscordOutbox.id)
        .limit(limit)
    ).all()
    return [
        OutboxItem(
            id=item.id,
            kind=item.kind,
            recommendation_id=item.recommendation_id,
            thread_id=thread_id if item.kind == POST else None,
            payload=dict(item.payload or {}),
        )
        for item, thread_id in rows
    ]


def record_delivery(
    session: Session,
    outbox_id: int,
    *,
    ok: bool,
    now: dt.datetime,
    channel_id: str | None = None,
    thread_id: str | None = None,
) -> bool:
    """Store the bot's result for one item. ``False`` when it was not applied.

    A delivered ``request`` must name its channel and thread. A failure counts an
    attempt; after :data:`MAX_ATTEMPTS` the item is no longer offered.
    """

    item = session.get(DiscordOutbox, outbox_id)
    if item is None or item.delivered_at is not None:
        return False
    if not ok:
        item.attempts = (item.attempts or 0) + 1
        session.flush()
        return True
    if item.kind == REQUEST:
        if not channel_id or not thread_id:
            return False
        request = session.get(DiscordRequest, item.recommendation_id)
        if request is None:  # pragma: no cover - the outbox row cascades with it
            return False
        request.channel_id = channel_id
        request.thread_id = thread_id
    item.delivered_at = now
    session.flush()
    return True

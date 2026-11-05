"""The Discord side of :class:`~tekijin.channels.notifier.ChannelNotifier` (#114).

A hand-off sent from Discord is received in a thread of the one output channel
(``discord_output_channel_id``). This notifier keeps that thread and the web chat
showing the same conversation: a message sent on the web is copied into the
thread, and an acceptance made on the web is announced there.

It never talks to Discord itself. It queues posts in ``discord_outbox`` and the bot
delivers them (``data/discord_requests.py``), so the bot token stays with the bot
and every post passes the bot's one output gate (``discord_bot/output.py``).

Only hand-offs that started on Discord get a thread. A question asked on the web
is never posted to Discord: the asker did not choose to show it there.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session, sessionmaker

from tekijin.config import get_settings
from tekijin.data.db import session_scope
from tekijin.data.discord_requests import (
    DISCORD_SESSION_PREFIX,
    create_request,
    enqueue_post,
    mark_accepted,
    question_was_sent,
)
from tekijin.data.members import discord_user_id_for
from tekijin.models.tables import Employee, Question, Recommendation

#: Discord rejects a message longer than this. The bot clips again; this keeps the
#: queued payload bounded.
_POST_MAX = 1900


def _enabled() -> bool:
    return get_settings().discord_output_enabled()


def relay_text(sender_name: str, body: str) -> str:
    """A web chat message as it appears in the thread."""

    return _clip(f"{sender_name}さん(Webから):\n{body}")


def accepted_text(responder_name: str) -> str:
    return (
        f"{responder_name}さんが依頼を受けました。"
        "回答はこのスレッドに書いてください。"
        "依頼された人が書いた内容は回答として記録され、知識の下書きになります。"
    )


def _clip(text: str) -> str:
    return text if len(text) <= _POST_MAX else text[: _POST_MAX - 1] + "…"


def request_payload(session: Session, *, recommendation_id: int, draft: str) -> dict | None:
    """What the bot needs to open the request thread, or ``None`` when the
    recommendation is gone."""

    rec = session.get(Recommendation, recommendation_id)
    if rec is None:
        return None
    question = session.get(Question, rec.question_id)
    responder = session.get(Employee, rec.employee_id)
    asker = session.get(Employee, question.asker_id) if question is not None else None
    guild = str(get_settings().discord_guild_id or "")
    return {
        "recommendation_id": recommendation_id,
        "question": (question.body if question is not None else None) or "",
        "draft": draft,
        "asker_name": asker.name if asker is not None else "",
        "asker_discord_id": discord_user_id_for(session, guild, asker.id) if asker else None,
        "responder_name": responder.name if responder is not None else "",
        "responder_discord_id": (
            discord_user_id_for(session, guild, responder.id) if responder else None
        ),
    }


def send_request(session: Session, *, session_id: str, recommendation_id: int, draft: str) -> bool:
    """Queue the request thread for a Discord hand-off. ``False`` when it was not
    queued: Discord output is off, the hand-off did not start on Discord, the
    recommendation is gone, or the request was already sent."""

    if not _enabled() or not session_id.startswith(DISCORD_SESSION_PREFIX):
        return False
    payload = request_payload(session, recommendation_id=recommendation_id, draft=draft)
    if payload is None:
        return False
    return create_request(
        session, recommendation_id=recommendation_id, session_id=session_id, payload=payload
    )


def pass_on_request(
    session_factory: sessionmaker[Session],
    *,
    session_id: str,
    recommendation_id: int,
    draft: str,
) -> bool:
    """A Discord hand-off that was sent moved on to a new recommendation (a decline
    or a referral, pressed in Discord or on the web): send it to the new person
    too. A hand-off never sent (「依頼する」 not pressed yet) stays unsent."""

    if not _enabled() or not session_id.startswith(DISCORD_SESSION_PREFIX):
        return False
    with session_scope(session_factory) as session:
        if not question_was_sent(session, recommendation_id):
            return False
        return send_request(
            session, session_id=session_id, recommendation_id=recommendation_id, draft=draft
        )


class DiscordNotifier:
    """Mirrors web-side events into Discord request threads (via the outbox)."""

    def relay_message(
        self,
        session_factory: sessionmaker[Session],
        *,
        employee_a: int,
        employee_b: int,
        sender_name: str,
        body: str,
        thread_id: int | None = None,
    ) -> None:
        if not _enabled() or thread_id is None:
            return
        with session_scope(session_factory) as session:
            enqueue_post(session, thread_id, relay_text(sender_name, body))

    def announce_handoff_request(
        self,
        session_factory: sessionmaker[Session],
        *,
        session_id: str,
        recommendation_id: int,
        thread_id: int,
        parties: dict,
        draft: str,
    ) -> None:
        if not _enabled() or not session_id.startswith(DISCORD_SESSION_PREFIX):
            return
        with session_scope(session_factory) as session:
            send_request(
                session, session_id=session_id, recommendation_id=recommendation_id, draft=draft
            )

    def announce_handoff_accepted(
        self,
        session_factory: sessionmaker[Session],
        *,
        thread_id: int,
        parties: dict,
        draft: str,
    ) -> None:
        if not _enabled():
            return
        with session_scope(session_factory) as session:
            mark_accepted(session, thread_id, dt.datetime.now(dt.UTC).replace(tzinfo=None))
            enqueue_post(session, thread_id, accepted_text(parties.get("responder_name") or ""))

"""Internal API for sending hand-offs from Discord and answering them there (#114).

The flow, all behind the bot's shared secret and only while Discord output is on
(``Settings.discord_output_enabled``):

1. ``/ask`` ends at a recommendation and the reply carries a 「依頼する」 button.
   ``POST /request`` sends the hand-off: the same draft save the web does
   (``consult_method="chat"``), and a request thread queued for the bot.
2. The thread holds 承諾 / 辞退 / 「自分より適任がいる」. ``POST /request/action``
   runs them through the same service functions as the web's ``/answer`` and
   ``/handoff/refer``. A decline or referral passes the request to the next person
   with a new thread.
3. ``POST /thread-message``: what the parties write in the thread becomes the
   web chat, and the requested member's writing is kept as their answer and as a
   case draft (``knowledge/request_answer.py``).
4. ``GET /outbox`` / ``POST /outbox/{id}``: the bot fetches what to post and
   reports the thread it created.

Everything the bot sends is checked here: a button's ids come from the Discord
client, so the pressing user must be the asker (「依頼する」) or the requested member
(the thread's buttons), resolved through their Discord identity, never created.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text
from sqlalchemy.orm import Session

from tekijin.api.discord_routes import DiscordUser, _guild_for, require_bot_secret
from tekijin.api.service import AgentService, HandoffNotFound, SessionConflict, SessionInvalid
from tekijin.channels.ask_summary import summarize_events
from tekijin.channels.discord_notifier import send_request
from tekijin.config import Settings
from tekijin.data.db import session_scope
from tekijin.data.discord_requests import (
    REQUEST,
    enqueue_post,
    mark_message_seen,
    pending_outbox,
    record_delivery,
    request_for_recommendation,
    request_for_thread,
    seen_message,
)
from tekijin.data.members import find_discord_member
from tekijin.data.messages import create_message, thread_parties
from tekijin.knowledge.request_answer import (
    record_thread_answer,
    refresh_case_draft,
    update_thread_answer,
)
from tekijin.models.tables import Answer, DiscordOutbox, Message
from tekijin.retrieval.embedding import PASSAGE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/discord", include_in_schema=False)

_DISCORD_ID = r"^[0-9]{1,20}$"
# Discord ids carry their creation time in milliseconds since 2015-01-01 UTC.
_DISCORD_EPOCH_MS = 1420070400000
# Discord caps a message at 2000 characters (4000 with Nitro).
_MAX_THREAD_MESSAGE = 4000


class DiscordSendRequest(DiscordUser):
    session_id: str = Field(pattern=r"^discord_[0-9a-f]{32}$")


class DiscordSendResponse(BaseModel):
    status: Literal["requested", "already", "forbidden", "gone", "disabled", "unlinked"]
    responder_name: str | None = None


class DiscordRequestAction(DiscordUser):
    recommendation_id: int = Field(ge=1)
    action: Literal["accept", "decline", "refer"]
    referred_discord_user_id: str | None = Field(default=None, pattern=_DISCORD_ID)


class DiscordActionResponse(BaseModel):
    status: Literal[
        "accepted",
        "declined",
        "referred",
        "forbidden",
        "stale",
        "gone",
        "disabled",
        "unlinked",
        "refer_unknown",
        "refer_invalid",
    ]
    # After a decline or referral: who the request went to next (None: nobody).
    next_name: str | None = None


class DiscordThreadMessage(DiscordUser):
    thread_id: str = Field(pattern=_DISCORD_ID)
    message_id: str = Field(pattern=_DISCORD_ID)
    body: str = Field(min_length=1, max_length=_MAX_THREAD_MESSAGE)
    # The message was edited in Discord; it corrects what it already became.
    edited: bool = False

    @field_validator("body")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("body must not be blank")
        return value.strip()


class DiscordThreadResponse(BaseModel):
    status: Literal[
        "stored",
        "answered",
        "updated",
        "ignored",
        "not_accepted",
        "duplicate",
        "disabled",
        "unlinked",
    ]


class OutboxItemResponse(BaseModel):
    id: int
    kind: str
    thread_id: str | None
    payload: dict


class OutboxResponse(BaseModel):
    channel_id: str
    items: list[OutboxItemResponse]


class OutboxResult(BaseModel):
    ok: bool
    channel_id: str | None = Field(default=None, pattern=_DISCORD_ID)
    thread_id: str | None = Field(default=None, pattern=_DISCORD_ID)


class OutboxResultResponse(BaseModel):
    status: Literal["ok", "refused"]


def _service(request: Request) -> AgentService:
    return request.app.state.agent_service


def _presser(service: AgentService, settings: Settings, user: DiscordUser) -> int | None:
    """The member who pressed the button, or ``None`` (another server, or someone
    TEKIJIN has never heard from)."""

    guild_id = _guild_for(settings, user)
    if guild_id is None:
        return None
    with session_scope(service.session_factory) as session:
        return find_discord_member(session, guild_id, user.discord_user_id)


def _now() -> dt.datetime:
    return dt.datetime.now()  # noqa: DTZ005 - naive, like messages.created_at (routes.py)


def _post(service: AgentService, recommendation_id: int, message: str) -> None:
    try:
        with session_scope(service.session_factory) as session:
            enqueue_post(session, recommendation_id, message)
    except Exception:  # noqa: BLE001 - a status line must not fail the action itself
        logger.exception("could not queue a post for request %s", recommendation_id)


def _drain(service: AgentService, session_id: str) -> None:
    """Run the queued resume to its next pause or end (no one reads /events here)."""

    events = [
        (e.event or "", json.loads(e.data or "{}")) for e in service.stream_events(session_id)
    ]
    if summarize_events(events).kind == "error":
        logger.warning("the run for discord session %s ended in an error", session_id)


@router.post("/request", response_model=DiscordSendResponse)
def discord_send_request(
    req: DiscordSendRequest,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> DiscordSendResponse:
    """「依頼する」: the asker sends the hand-off ``/ask`` ended at."""

    if not settings.discord_output_enabled():
        return DiscordSendResponse(status="disabled")
    if _guild_for(settings, req) is None:
        return DiscordSendResponse(status="unlinked")
    service = _service(request)
    presser = _presser(service, settings, req)
    asker_id, _ = service.session_participants(req.session_id)
    if asker_id is None:
        return DiscordSendResponse(status="gone")
    if presser is None or presser != asker_id:
        return DiscordSendResponse(status="forbidden")
    try:
        handoff = service.get_handoff(req.session_id)
    except (HandoffNotFound, SessionConflict):
        return DiscordSendResponse(status="gone")
    rec_id = handoff.recommendation_id
    name = handoff.responder.name if handoff.responder is not None else None
    if rec_id is None:
        return DiscordSendResponse(status="gone")
    with session_scope(service.session_factory) as session:
        if request_for_recommendation(session, rec_id) is not None:
            return DiscordSendResponse(status="already", responder_name=name)
    draft = handoff.draft.strip() or handoff.question
    try:
        service.save_handoff_draft(req.session_id, draft, "chat", actor_id=presser)
    except (HandoffNotFound, SessionConflict):
        return DiscordSendResponse(status="gone")
    with session_scope(service.session_factory) as session:
        sent = send_request(
            session, session_id=req.session_id, recommendation_id=rec_id, draft=draft
        )
    return DiscordSendResponse(status="requested" if sent else "already", responder_name=name)


@router.post("/request/action", response_model=DiscordActionResponse)
def discord_request_action(
    req: DiscordRequestAction,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> DiscordActionResponse:
    """承諾 / 辞退 / 「自分より適任がいる」, pressed in a request thread."""

    if not settings.discord_output_enabled():
        return DiscordActionResponse(status="disabled")
    guild_id = _guild_for(settings, req)
    if guild_id is None:
        return DiscordActionResponse(status="unlinked")
    service = _service(request)
    with session_scope(service.session_factory) as session:
        found = request_for_recommendation(session, req.recommendation_id)
        session_id = found.session_id if found is not None else None
    if session_id is None:
        return DiscordActionResponse(status="gone")
    try:
        handoff = service.get_handoff(session_id)
    except (HandoffNotFound, SessionConflict):
        return DiscordActionResponse(status="gone")
    if handoff.recommendation_id != req.recommendation_id:
        return DiscordActionResponse(status="stale")
    _, responder_id = service.session_participants(session_id)
    presser = _presser(service, settings, req)
    if presser is None or responder_id is None or presser != responder_id:
        return DiscordActionResponse(status="forbidden")
    name = handoff.responder.name if handoff.responder is not None else ""
    try:
        if req.action == "accept":
            service.submit_resume(
                session_id, outcome="accepted", recommendation_id=req.recommendation_id
            )
            _drain(service, session_id)
            return DiscordActionResponse(status="accepted")
        if req.action == "decline":
            service.submit_resume(
                session_id, outcome="declined", recommendation_id=req.recommendation_id
            )
            _post(service, req.recommendation_id, f"{name}さんは、この依頼を辞退しました。")
            return DiscordActionResponse(
                status="declined", next_name=_pass_on(service, session_id, req.recommendation_id)
            )
        return _refer(service, req, guild_id, session_id, presser, name)
    except SessionConflict:
        return DiscordActionResponse(status="stale")
    except HandoffNotFound:
        return DiscordActionResponse(status="gone")


def _refer(
    service: AgentService,
    req: DiscordRequestAction,
    guild_id: str,
    session_id: str,
    presser: int,
    name: str,
) -> DiscordActionResponse:
    if req.referred_discord_user_id is None:
        return DiscordActionResponse(status="refer_unknown")
    with session_scope(service.session_factory) as session:
        referred = find_discord_member(session, guild_id, req.referred_discord_user_id)
    if referred is None:
        return DiscordActionResponse(status="refer_unknown")
    try:
        service.refer_handoff_target(session_id, referred, actor_id=presser)
    except SessionInvalid:
        return DiscordActionResponse(status="refer_invalid")
    _post(service, req.recommendation_id, f"{name}さんが、より適任の人を挙げました。")
    return DiscordActionResponse(
        status="referred", next_name=_pass_on(service, session_id, req.recommendation_id)
    )


def _pass_on(service: AgentService, session_id: str, previous: int) -> str | None:
    """After a decline or referral: run the reroute, send the request on to the
    next person, and say so in the old thread. Returns their name, or ``None``."""

    _drain(service, session_id)
    try:
        handoff = service.get_handoff(session_id)
    except (HandoffNotFound, SessionConflict):
        handoff = None
    rec_id = handoff.recommendation_id if handoff is not None else None
    if handoff is None or rec_id is None or rec_id == previous:
        _post(service, previous, "ほかに依頼できる人が見つかりませんでした。")
        return None
    draft = handoff.draft.strip() or handoff.question
    with session_scope(service.session_factory) as session:
        send_request(session, session_id=session_id, recommendation_id=rec_id, draft=draft)
    name = handoff.responder.name if handoff.responder is not None else ""
    _post(service, previous, f"次の候補の{name}さんに依頼しました。")
    return name


@router.post("/thread-message", response_model=DiscordThreadResponse)
def discord_thread_message(
    req: DiscordThreadMessage,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> DiscordThreadResponse:
    """A message written (or edited) in a request thread. Only the two parties'
    messages written after the request was accepted are kept; the requested
    member's are also their answer. Each Discord message is handled once; an edit
    of one already kept corrects what it became."""

    if not settings.discord_output_enabled():
        return DiscordThreadResponse(status="disabled")
    guild_id = _guild_for(settings, req)
    if guild_id is None:
        return DiscordThreadResponse(status="unlinked")
    service = _service(request)
    with session_scope(service.session_factory) as session:
        found = request_for_thread(session, req.thread_id)
        if found is None:
            return DiscordThreadResponse(status="ignored")
        rec_id = found.recommendation_id
        accepted_at = found.accepted_at
        seen = seen_message(session, req.message_id)
        seen_rows = (seen.message_row_id, seen.answer_id) if seen is not None else None
        sender = find_discord_member(session, guild_id, req.discord_user_id)
        parties = thread_parties(session, rec_id)
    if seen_rows is not None:
        if req.edited and parties is not None:
            return _apply_edit(service, req, parties, *seen_rows)
        return DiscordThreadResponse(status="duplicate")
    status: Literal["ignored", "not_accepted"] | None = None
    if sender is None:
        status = "ignored"
    elif parties is None or (
        accepted_at is not None and snowflake_time(req.message_id) < accepted_at
    ):
        status = "not_accepted"
    elif sender not in (parties["asker_id"], parties["responder_id"]):
        status = "ignored"
    if status is not None:
        # Remembered, so the bot sending it again (after a restart, once the
        # request is accepted) does not turn it into an answer.
        with session_scope(service.session_factory) as session:
            _lock_message(session, req.message_id)
            mark_message_seen(session, req.message_id, rec_id)
        return DiscordThreadResponse(status=status)
    assert parties is not None and sender is not None
    is_answer = sender == parties["responder_id"]
    embedding = _embed(service, req.body) if is_answer else None
    now = _now()
    # Stored at the time it was written, so a message recovered late still sorts
    # in the conversation (and the draft) where it was written.
    written_at = snowflake_local_time(req.message_id)
    with session_scope(service.session_factory) as session:
        _lock_message(session, req.message_id)
        if seen_message(session, req.message_id) is not None:
            return DiscordThreadResponse(status="duplicate")
        row = create_message(session, rec_id, sender, req.body, written_at)
        answer_id = None
        if is_answer:
            # One draft per question even when two answers arrive together.
            session.execute(
                text("SELECT pg_advisory_xact_lock(:cls, :obj)"), {"cls": 114, "obj": rec_id}
            )
            answer_id, _ = record_thread_answer(
                session,
                parties=parties,
                body=req.body,
                embedding=embedding,
                now=now,
                written_at=written_at,
            )
        if not mark_message_seen(
            session, req.message_id, rec_id, message_row_id=row["id"], answer_id=answer_id
        ):
            raise HTTPException(status_code=409, detail="message already handled")
    if not is_answer:
        return DiscordThreadResponse(status="stored")
    # The draft has no embedding yet (or lost it to the new resolution); without
    # one it cannot be matched to new questions or rise in the review queue.
    service.schedule_knowledge_embed()
    return DiscordThreadResponse(status="answered")


class DiscordThreadDelete(BaseModel):
    guild_id: str = Field(pattern=_DISCORD_ID)
    message_ids: list[str] = Field(min_length=1, max_length=100)
    # The thread the messages were in, when the bot knows it: ids never received
    # are then remembered as deleted, so a late send of one is refused.
    thread_id: str | None = Field(default=None, pattern=_DISCORD_ID)

    @field_validator("message_ids")
    @classmethod
    def _ids(cls, value: list[str]) -> list[str]:
        if not all(v.isdigit() and len(v) <= 20 for v in value):
            raise ValueError("message ids are Discord ids")
        return value


class DiscordThreadDeleteResponse(BaseModel):
    status: Literal["ok", "disabled", "unlinked"]
    removed: int = 0


@router.post("/thread-message/delete", response_model=DiscordThreadDeleteResponse)
def discord_thread_message_delete(
    req: DiscordThreadDelete,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> DiscordThreadDeleteResponse:
    """Messages deleted from a request thread: remove the chat messages and answers
    they became, and rewrite the still-reviewable draft without them."""

    if not settings.discord_output_enabled():
        return DiscordThreadDeleteResponse(status="disabled")
    if str(settings.discord_guild_id) != req.guild_id:
        return DiscordThreadDeleteResponse(status="unlinked")
    service = _service(request)
    removed = 0
    touched: set[int] = set()
    now = _now()
    with session_scope(service.session_factory) as session:
        thread = request_for_thread(session, req.thread_id) if req.thread_id else None
        for message_id in sorted(set(req.message_ids)):
            # Waits for a capture of the same message still in flight, so its
            # rows are seen (and removed) here rather than surviving the deletion.
            _lock_message(session, message_id)
            seen = seen_message(session, message_id)
            if seen is None:
                if thread is not None:
                    mark_message_seen(session, message_id, thread.recommendation_id)
                continue
            if seen.message_row_id is None and seen.answer_id is None:
                continue
            if seen.answer_id is not None:
                answer = session.get(Answer, seen.answer_id)
                if answer is not None:
                    session.delete(answer)
                touched.add(seen.recommendation_id)
            if seen.message_row_id is not None:
                message = session.get(Message, seen.message_row_id)
                if message is not None:
                    session.delete(message)
            seen.answer_id = None
            seen.message_row_id = None
            removed += 1
        session.flush()
        for rec_id in touched:
            parties = thread_parties(session, rec_id)
            if parties is None:
                continue
            session.execute(
                text("SELECT pg_advisory_xact_lock(:cls, :obj)"), {"cls": 114, "obj": rec_id}
            )
            refresh_case_draft(session, parties=parties, now=now)
    if touched:
        service.schedule_knowledge_embed()
    return DiscordThreadDeleteResponse(status="ok", removed=removed)


def _apply_edit(
    service: AgentService,
    req: DiscordThreadMessage,
    parties: dict,
    message_row_id: int | None,
    answer_id: str | None,
) -> DiscordThreadResponse:
    """Correct the chat message and answer an edited Discord message became."""

    if message_row_id is None:
        return DiscordThreadResponse(status="duplicate")  # it was ignored; still is
    embedding = _embed(service, req.body) if answer_id is not None else None
    now = _now()
    with session_scope(service.session_factory) as session:
        message = session.get(Message, message_row_id)
        if message is not None:
            message.body = req.body
        if answer_id is not None:
            session.execute(
                text("SELECT pg_advisory_xact_lock(:cls, :obj)"),
                {"cls": 114, "obj": message.recommendation_id if message else 0},
            )
            update_thread_answer(
                session,
                parties=parties,
                answer_id=answer_id,
                body=req.body,
                embedding=embedding,
                now=now,
            )
    if answer_id is not None:
        service.schedule_knowledge_embed()
    return DiscordThreadResponse(status="updated")


def _lock_message(session: Session, message_id: str) -> None:
    """Serialize everything done for one Discord message (capture, deletion)."""

    session.execute(
        text("SELECT pg_advisory_xact_lock(:cls, hashtext(:obj))"),
        {"cls": 1141, "obj": message_id},
    )


def snowflake_local_time(snowflake: str) -> dt.datetime:
    """When a Discord message was written, as naive local time like the other
    ``messages.created_at`` (routes.py stamps ``dt.datetime.now()``)."""

    ms = (int(snowflake) >> 22) + _DISCORD_EPOCH_MS
    return dt.datetime.fromtimestamp(ms / 1000)  # noqa: DTZ006 - naive local on purpose


def snowflake_time(snowflake: str) -> dt.datetime:
    """When a Discord message was written (naive UTC), read from its id."""

    ms = (int(snowflake) >> 22) + _DISCORD_EPOCH_MS
    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).replace(tzinfo=None)


def _embed(service: AgentService, body: str) -> list[float] | None:
    try:
        return service.embedder.encode([body], kind=PASSAGE)[0]
    except Exception:  # noqa: BLE001 - still reusable through BM25 without it
        logger.exception("failed to embed a discord answer; storing it without one")
        return None


@router.get("/outbox", response_model=OutboxResponse)
def discord_outbox(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    settings: Settings = Depends(require_bot_secret),
) -> OutboxResponse:
    """What the bot should post now. Nothing while Discord output is off."""

    if not settings.discord_output_enabled():
        raise HTTPException(status_code=404, detail="Discord output is off")
    with session_scope(_service(request).session_factory) as session:
        items = pending_outbox(session, limit=limit)
    return OutboxResponse(
        channel_id=str(settings.discord_output_channel_id),
        items=[
            OutboxItemResponse(id=i.id, kind=i.kind, thread_id=i.thread_id, payload=i.payload)
            for i in items
        ],
    )


@router.post("/outbox/{outbox_id}", response_model=OutboxResultResponse)
def discord_outbox_result(
    outbox_id: int,
    result: OutboxResult,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> OutboxResultResponse:
    """The bot's result for one item. A thread must be in the output channel."""

    if not settings.discord_output_enabled():
        raise HTTPException(status_code=404, detail="Discord output is off")
    with session_scope(_service(request).session_factory) as session:
        item = session.get(DiscordOutbox, outbox_id)
        if (
            result.ok
            and item is not None
            and item.kind == REQUEST
            and result.channel_id != str(settings.discord_output_channel_id)
        ):
            return OutboxResultResponse(status="refused")
        applied = record_delivery(
            session,
            outbox_id,
            ok=result.ok,
            channel_id=result.channel_id,
            thread_id=result.thread_id,
            now=_now(),
        )
    return OutboxResultResponse(status="ok" if applied else "refused")

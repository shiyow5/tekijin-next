"""Internal API for the Discord bot process (Discord D1 #60, members #66).

The bot (``python -m tekijin.discord_bot``) is a separate, stateless process. It calls
these endpoints with a shared secret instead of a user token: the bot vouches for
which Discord user typed the command. That user becomes a member of the server's
organization the first time TEKIJIN hears from them (``data/members.py``).

``/tekijin join`` hands back a one-time login URL for the web UI. It points at the
frontend with the token in the URL fragment; the frontend redeems it at
``POST /auth/discord/redeem`` for a normal bearer token.

OFF by default: with ``discord_bot_secret`` unset the internal endpoints answer 404,
as if they did not exist.
"""

from __future__ import annotations

import datetime as dt
import hmac
import json
import logging
import threading
import uuid
from typing import Annotated, Literal

import jwt
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from tekijin.api import schemas
from tekijin.api.service import AgentService, AskerNotFound, ServiceBusy
from tekijin.auth.principal import Principal
from tekijin.auth.tokens import create_access_token
from tekijin.channels.ask_summary import AskOutcome, summarize_events
from tekijin.config import Settings, get_settings
from tekijin.data.db import session_scope
from tekijin.data.documents import get_document
from tekijin.data.members import (
    get_or_create_org_for_guild,
    mark_joined,
    resolve_discord_member,
    set_opted_out,
)
from tekijin.data.sources import MANUAL, manual_external_id, upsert_source_item
from tekijin.models.tables import Employee

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/discord", include_in_schema=False)
login_router = APIRouter(prefix="/auth/discord")

_LOGIN_PURPOSE = "discord_login"
_LOGIN_TTL = dt.timedelta(minutes=10)
_LOGIN_INVALID = "ログインURLが無効か、期限が切れています。"

# How much of a cited document's body the reply quotes.
_EXCERPT_LENGTH = 300

# Discord caps a slash-command string option at 6000 characters; the web /ask has no
# cap, but a chat command should not be a way to push arbitrarily large inputs.
_MAX_QUESTION_LENGTH = 2000


class DiscordUser(BaseModel):
    discord_user_id: str = Field(pattern=r"^[0-9]{1,20}$")
    # None for a DM; the configured server is used then.
    guild_id: str | None = Field(default=None, pattern=r"^[0-9]{1,20}$")
    display_name: str = Field(default="", max_length=255)


class DiscordAskRequest(DiscordUser):
    question: str = Field(min_length=1, max_length=_MAX_QUESTION_LENGTH)

    @field_validator("question")
    @classmethod
    def _trim_nonempty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("question must not be blank")
        return trimmed


class DiscordAskResponse(AskOutcome):
    web_url: str


class DiscordMemberRequest(DiscordUser):
    action: Literal["join", "optout", "optin"]


class DiscordMemberResponse(BaseModel):
    status: Literal["joined", "opted_out", "opted_in", "unlinked"]
    # ``join`` only: open within 10 minutes, once.
    login_url: str | None = None


# ``/tekijin add`` (#70). Discord's option cap is 6000; a longer document is better
# added from the web, which takes a file.
_MAX_ADD_LENGTH = 4000


class DiscordAddRequest(DiscordUser):
    text: str = Field(min_length=1, max_length=_MAX_ADD_LENGTH)
    title: str | None = Field(default=None, max_length=200)
    url: str | None = Field(default=None, max_length=2000, pattern=r"^https?://\S+$")

    @field_validator("text")
    @classmethod
    def _text_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    @field_validator("title")
    @classmethod
    def _title_or_none(cls, value: str | None) -> str | None:
        return (value or "").strip() or None


class DiscordAddResponse(BaseModel):
    status: Literal["added", "updated", "unchanged", "not_joined", "unlinked"]
    chunks: int = 0


class DiscordRedeemRequest(BaseModel):
    token: str = Field(min_length=1, max_length=2048)


def require_bot_secret(
    x_tekijin_bot_secret: Annotated[str | None, Header()] = None,
) -> Settings:
    settings = get_settings()
    if not settings.discord_bot_secret:
        raise HTTPException(status_code=404, detail="Not Found")
    given = (x_tekijin_bot_secret or "").encode()
    if not hmac.compare_digest(given, settings.discord_bot_secret.encode()):
        raise HTTPException(status_code=401, detail="invalid bot secret")
    return settings


@router.post("/ask", response_model=DiscordAskResponse)
def discord_ask(
    req: DiscordAskRequest,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> DiscordAskResponse:
    """Run one question to the end of its first segment and return the outcome.

    The run is the same one ``POST /ask`` + ``GET /events`` drive for the web client;
    this just consumes the events server-side. D1 cannot continue a run that pauses
    (a hand-off waiting to be sent, or a clarification), so the run is discarded
    once read: the unsent recommendation must not reach the responder's inbox, and
    a paused session would otherwise never be swept. The reply points the asker to
    the web to ask again.
    """

    web_url = settings.discord_web_url
    service: AgentService = request.app.state.agent_service
    asker_id = _member_id(service, settings, req)
    if asker_id is None:
        return DiscordAskResponse(kind="unlinked", web_url=web_url)

    session_id = f"discord_{uuid.uuid4().hex}"
    try:
        service.start_question(session_id, asker_id, req.question)
    except ServiceBusy as exc:
        raise HTTPException(
            status_code=503,
            detail="現在混雑しています。少し待ってから、もう一度お試しください。",
            headers={"Retry-After": "5"},
        ) from exc
    except AskerNotFound:
        logger.warning("discord user %s maps to unknown member %s", req.discord_user_id, asker_id)
        return DiscordAskResponse(kind="unlinked", web_url=web_url)
    try:
        events = [
            (e.event or "", json.loads(e.data or "{}")) for e in service.stream_events(session_id)
        ]
    finally:
        _discard(service, session_id)
    outcome = _with_document_preview(service, summarize_events(events))
    return DiscordAskResponse(**outcome.model_dump(), web_url=web_url)


def _with_document_preview(service: AgentService, outcome: AskOutcome) -> AskOutcome:
    """Add the cited document's title and opening lines (the web opens the document
    instead). A lookup failure only loses the preview, never the answer."""

    if not outcome.doc_id:
        return outcome
    try:
        with session_scope(service.session_factory) as session:
            doc = get_document(session, outcome.doc_id)
    except Exception:  # noqa: BLE001 - the preview is optional
        logger.exception("failed to load document %s for a discord reply", outcome.doc_id)
        return outcome
    if doc is None:
        return outcome
    body = " ".join((doc.get("body") or "").split())
    excerpt = body if len(body) <= _EXCERPT_LENGTH else body[: _EXCERPT_LENGTH - 1] + "…"
    return outcome.model_copy(
        update={"document_title": doc.get("title"), "document_excerpt": excerpt or None}
    )


def _discard(service: AgentService, session_id: str) -> None:
    try:
        service.discard_session(session_id)
    except Exception:  # noqa: BLE001 - the answer is already computed; do not fail it
        logger.exception("failed to discard discord session %s", session_id)


@router.post("/member", response_model=DiscordMemberResponse)
def discord_member(
    req: DiscordMemberRequest,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> DiscordMemberResponse:
    """``/tekijin join`` / ``optout`` / ``optin`` for the Discord user."""

    service: AgentService = request.app.state.agent_service
    guild_id = _guild_for(settings, req)
    if guild_id is None:
        return DiscordMemberResponse(status="unlinked")
    now = service.now()
    with session_scope(service.session_factory) as session:
        org = get_or_create_org_for_guild(session, guild_id, name=f"Discord {guild_id}")
        member = resolve_discord_member(
            session, org, req.discord_user_id, display_name=req.display_name, now=now
        )
        if req.action == "join":
            mark_joined(member, now)
            member_id = member.id
        else:
            set_opted_out(member, req.action == "optout", now)
            return DiscordMemberResponse(
                status="opted_out" if req.action == "optout" else "opted_in"
            )
    token = _login_token(settings, member_id)
    return DiscordMemberResponse(
        status="joined",
        login_url=f"{settings.discord_web_url.rstrip('/')}/login#discord_login={token}",
    )


@router.post("/add", response_model=DiscordAddResponse)
def discord_add(
    req: DiscordAddRequest,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> DiscordAddResponse:
    """``/tekijin add``: store a note a joined member sends (#70).

    With a URL, the note stands for the document there, so sending the same URL
    again updates it. Without one, every add is a new note.
    """

    service: AgentService = request.app.state.agent_service
    guild_id = _guild_for(settings, req)
    if guild_id is None:
        return DiscordAddResponse(status="unlinked")
    now = service.now()
    with session_scope(service.session_factory) as session:
        org = get_or_create_org_for_guild(session, guild_id, name=f"Discord {guild_id}")
        member = resolve_discord_member(
            session, org, req.discord_user_id, display_name=req.display_name, now=now
        )
        if member.joined_at is None:
            return DiscordAddResponse(status="not_joined")
        result = upsert_source_item(
            session,
            service.embedder,
            org_id=org.id,
            provider=MANUAL,
            external_id=manual_external_id(req.url),
            title=req.title,
            body=req.text,
            url=req.url,
            author_id=member.id,
            now=now,
        )
    return DiscordAddResponse(status=result.status, chunks=result.chunks)


@login_router.post("/redeem", response_model=schemas.LoginResponse)
def redeem_discord_login(req: DiscordRedeemRequest, request: Request) -> schemas.LoginResponse:
    """Trade a ``/tekijin join`` login token for a bearer token, once."""

    settings = get_settings()
    try:
        payload = jwt.decode(req.token, settings.auth_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=_LOGIN_INVALID) from exc
    if payload.get("purpose") != _LOGIN_PURPOSE or not _spend(request, str(payload.get("jti"))):
        raise HTTPException(status_code=401, detail=_LOGIN_INVALID)

    service: AgentService = request.app.state.agent_service
    with service.session_factory() as session:
        member = session.get(Employee, int(payload["sub"]))
        if member is None or member.joined_at is None:
            raise HTTPException(status_code=401, detail=_LOGIN_INVALID)
        principal = Principal(
            employee_id=member.id, name=member.name, dept=member.department, is_admin=False
        )
    token = create_access_token(
        principal, secret=settings.auth_secret, ttl_hours=settings.auth_token_ttl_hours
    )
    return schemas.LoginResponse(
        access_token=token,
        principal=schemas.PrincipalResponse(
            id=schemas.format_employee_id(member.id),
            name=principal.name,
            dept=principal.dept,
            is_admin=False,
        ),
    )


def _guild_for(settings: Settings, user: DiscordUser) -> str | None:
    """The server the request belongs to, or ``None`` when it is not ours.

    TEKIJIN serves the configured server only (one organization for now). With no
    server configured nothing is ours: otherwise anyone in any server the bot was
    added to could ``/tekijin join`` and sign in. A DM carries no server, so it
    falls back to the configured one.
    """

    if not settings.discord_guild_id:
        return None
    configured = str(settings.discord_guild_id)
    if user.guild_id is not None and user.guild_id != configured:
        return None
    return configured


def _member_id(service: AgentService, settings: Settings, user: DiscordUser) -> int | None:
    guild_id = _guild_for(settings, user)
    if guild_id is None:
        return None
    with session_scope(service.session_factory) as session:
        org = get_or_create_org_for_guild(session, guild_id, name=f"Discord {guild_id}")
        member = resolve_discord_member(
            session, org, user.discord_user_id, display_name=user.display_name, now=service.now()
        )
        return member.id


def _login_token(settings: Settings, member_id: int) -> str:
    now = dt.datetime.now(dt.UTC)
    return jwt.encode(
        {
            "purpose": _LOGIN_PURPOSE,
            "sub": str(member_id),
            "jti": uuid.uuid4().hex,
            "iat": int(now.timestamp()),
            "exp": int((now + _LOGIN_TTL).timestamp()),
        },
        settings.auth_secret,
        algorithm="HS256",
    )


_spent_guard = threading.Lock()


def _spend(request: Request, jti: str) -> bool:
    """True the first time ``jti`` is seen. The API runs as a single worker, so
    process memory is enough; a restart only re-opens tokens that expire in
    minutes anyway."""

    with _spent_guard:
        spent: set[str] = getattr(request.app.state, "discord_login_spent", set())
        request.app.state.discord_login_spent = spent
        if jti in spent:
            return False
        spent.add(jti)
        return True

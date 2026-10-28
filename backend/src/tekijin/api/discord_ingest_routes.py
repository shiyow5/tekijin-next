"""Internal API through which the bot forwards channel messages (#76, R2).

Same shared-secret guard as ``api/discord_routes.py``; separate only to keep that
file to the slash commands. The bot sends messages in batches, so one request may
rebuild several conversations (``data/conversations.py``). Only the configured
server is accepted: a batch from any other server is ignored, not stored.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from tekijin.api.discord_routes import require_bot_secret
from tekijin.api.service import AgentService
from tekijin.config import Settings
from tekijin.data.conversations import (
    IncomingMessage,
    channel_cursors,
    delete_messages,
    ingest_messages,
)
from tekijin.data.db import session_scope
from tekijin.data.members import get_or_create_org_for_guild

router = APIRouter(prefix="/internal/discord", include_in_schema=False)

_SNOWFLAKE = r"^[0-9]{1,20}$"
# The bot flushes well below these; they bound what one request can make the
# backend re-embed.
_MAX_MESSAGES = 200
_MAX_DELETIONS = 1000


class DiscordMessageIn(BaseModel):
    message_id: int = Field(ge=0)
    channel_id: int = Field(ge=0)
    channel_name: str = Field(max_length=255)
    thread_id: int | None = Field(default=None, ge=0)
    thread_name: str | None = Field(default=None, max_length=255)
    author_id: str = Field(pattern=_SNOWFLAKE)
    author_name: str = Field(default="", max_length=255)
    # Discord allows 4000 characters with Nitro.
    content: str = Field(min_length=1, max_length=6000)
    created_at: dt.datetime
    edited_at: dt.datetime | None = None


class DiscordMessagesRequest(BaseModel):
    guild_id: str = Field(pattern=_SNOWFLAKE)
    messages: list[DiscordMessageIn] = Field(max_length=_MAX_MESSAGES)


class DiscordMessagesResponse(BaseModel):
    status: Literal["ok", "unlinked"]
    stored: int = 0
    conversations: int = 0


class DiscordDeleteRequest(BaseModel):
    guild_id: str = Field(pattern=_SNOWFLAKE)
    message_ids: list[int] = Field(max_length=_MAX_DELETIONS)


class DiscordDeleteResponse(BaseModel):
    status: Literal["ok", "unlinked"]
    deleted: int = 0


class DiscordCursorsResponse(BaseModel):
    status: Literal["ok", "unlinked"]
    # Channel or thread id -> the last stored message id there.
    cursors: dict[str, int] = Field(default_factory=dict)


def _ours(settings: Settings, guild_id: str) -> bool:
    return bool(settings.discord_guild_id) and guild_id == str(settings.discord_guild_id)


def _utc_naive(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(dt.UTC).replace(tzinfo=None)


def _incoming(message: DiscordMessageIn) -> IncomingMessage:
    return IncomingMessage(
        message_id=message.message_id,
        channel_id=message.channel_id,
        channel_name=message.channel_name,
        thread_id=message.thread_id,
        thread_name=message.thread_name,
        author_discord_id=message.author_id,
        author_name=message.author_name,
        content=message.content,
        created_at=_utc_naive(message.created_at),
        edited_at=_utc_naive(message.edited_at) if message.edited_at else None,
    )


def _service(request: Request) -> AgentService:
    service: AgentService = request.app.state.agent_service
    return service


@router.post("/messages", response_model=DiscordMessagesResponse)
def discord_messages(
    req: DiscordMessagesRequest,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> DiscordMessagesResponse:
    """Store new or edited messages and rebuild their conversations."""

    if not _ours(settings, req.guild_id):
        return DiscordMessagesResponse(status="unlinked")
    service = _service(request)
    with session_scope(service.session_factory) as session:
        org = get_or_create_org_for_guild(session, req.guild_id, name=f"Discord {req.guild_id}")
        result = ingest_messages(
            session,
            service.embedder,
            org=org,
            guild_id=req.guild_id,
            messages=[_incoming(m) for m in req.messages],
            now=service.now(),
        )
    return DiscordMessagesResponse(
        status="ok", stored=result.stored, conversations=result.conversations
    )


@router.post("/messages/delete", response_model=DiscordDeleteResponse)
def discord_delete_messages(
    req: DiscordDeleteRequest,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> DiscordDeleteResponse:
    """Forget messages deleted in Discord and rebuild their conversations."""

    if not _ours(settings, req.guild_id):
        return DiscordDeleteResponse(status="unlinked")
    service = _service(request)
    with session_scope(service.session_factory) as session:
        org = get_or_create_org_for_guild(session, req.guild_id, name=f"Discord {req.guild_id}")
        deleted = delete_messages(
            session,
            service.embedder,
            org_id=org.id,
            guild_id=req.guild_id,
            message_ids=req.message_ids,
            now=service.now(),
        )
    return DiscordDeleteResponse(status="ok", deleted=deleted)


@router.get("/cursors", response_model=DiscordCursorsResponse)
def discord_cursors(
    request: Request,
    guild_id: str = Query(pattern=_SNOWFLAKE),
    settings: Settings = Depends(require_bot_secret),
) -> DiscordCursorsResponse:
    """Where each channel's history left off, so the bot reads only what is new."""

    if not _ours(settings, guild_id):
        return DiscordCursorsResponse(status="unlinked")
    with session_scope(_service(request).session_factory) as session:
        org = get_or_create_org_for_guild(session, guild_id, name=f"Discord {guild_id}")
        cursors = channel_cursors(session, org_id=org.id)
    return DiscordCursorsResponse(
        status="ok", cursors={str(stream): last for stream, last in cursors.items()}
    )

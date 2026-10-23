"""Internal API for the Discord bot process (Discord D1, #60).

The bot (``python -m tekijin.discord_bot``) is a separate, stateless process. It calls
``POST /internal/discord/ask`` with a shared secret instead of a user token: the bot
vouches for which Discord user typed ``/ask``, and this endpoint maps that user to an
employee through ``Settings.discord_user_map``. Real account linking replaces the map
in D2.

OFF by default: with ``discord_bot_secret`` unset the endpoint answers 404, as if it
did not exist.
"""

from __future__ import annotations

import hmac
import json
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from tekijin.api.service import AgentService, AskerNotFound, ServiceBusy
from tekijin.channels.ask_summary import AskOutcome, summarize_events
from tekijin.config import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/discord", include_in_schema=False)

# Discord caps a slash-command string option at 6000 characters; the web /ask has no
# cap, but a chat command should not be a way to push arbitrarily large inputs.
_MAX_QUESTION_LENGTH = 2000


class DiscordAskRequest(BaseModel):
    discord_user_id: str = Field(pattern=r"^[0-9]{1,20}$")
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
    asker_id = settings.discord_user_map.get(req.discord_user_id)
    if asker_id is None:
        return DiscordAskResponse(kind="unlinked", web_url=web_url)

    service: AgentService = request.app.state.agent_service
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
        logger.warning("discord user %s maps to unknown employee %s", req.discord_user_id, asker_id)
        return DiscordAskResponse(kind="unlinked", web_url=web_url)
    try:
        events = [
            (e.event or "", json.loads(e.data or "{}")) for e in service.stream_events(session_id)
        ]
    finally:
        _discard(service, session_id)
    outcome = summarize_events(events)
    return DiscordAskResponse(**outcome.model_dump(), web_url=web_url)


def _discard(service: AgentService, session_id: str) -> None:
    try:
        service.discard_session(session_id)
    except Exception:  # noqa: BLE001 - the answer is already computed; do not fail it
        logger.exception("failed to discard discord session %s", session_id)

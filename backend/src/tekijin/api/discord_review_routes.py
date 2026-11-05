"""Internal API for ``/tekijin review``: checking knowledge drafts from Discord (#114).

Only the Discord users listed in ``discord_admin_user_ids`` may list the review
queue or press 確認 / 差し戻し, and the check is made here on every call: the
article id on a button comes from the Discord client. The actions are the same
store functions as the web's knowledge screen (``api/knowledge_routes.py``), and a
checked article is embedded at once so it can answer the next question.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from tekijin.api.discord_routes import DiscordUser, _guild_for, require_bot_secret
from tekijin.api.service import AgentService
from tekijin.config import Settings
from tekijin.data.db import session_scope
from tekijin.data.knowledge import (
    REVIEW_STATES,
    ArticleNotFound,
    ArticleStateError,
    list_articles,
    send_back_article,
    verify_article,
)
from tekijin.data.members import find_discord_member, get_or_create_org_for_guild
from tekijin.data.visibility import EVERYTHING
from tekijin.knowledge.index import embed_articles

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/discord", include_in_schema=False)

#: Discord allows 5 rows of components; one row of buttons per draft.
REVIEW_PAGE = 5
_ISSUE_MAX = 200
_RESOLUTION_MAX = 300


class ReviewArticle(BaseModel):
    id: int
    kind: str
    state: str
    issue: str
    resolution: str | None
    demand_count: int


class DiscordReviewResponse(BaseModel):
    status: Literal["ok", "forbidden", "unlinked"]
    articles: list[ReviewArticle] = Field(default_factory=list)
    total: int = 0


class DiscordReviewAction(DiscordUser):
    article_id: int = Field(ge=1)
    action: Literal["verify", "send_back"]


class DiscordReviewActionResponse(BaseModel):
    status: Literal["verified", "sent_back", "invalid", "gone", "forbidden", "unlinked"]


def _clip(value: str | None, limit: int) -> str | None:
    if value is None or len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _admin_guild(settings: Settings, user: DiscordUser) -> tuple[str | None, bool]:
    """``(guild_id, is_admin)`` for the user. ``guild_id`` is ``None`` for another
    server."""

    guild_id = _guild_for(settings, user)
    return guild_id, guild_id is not None and user.discord_user_id in settings.discord_admin_ids()


@router.post("/review", response_model=DiscordReviewResponse)
def discord_review(
    req: DiscordUser,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> DiscordReviewResponse:
    """The drafts waiting for review, most wanted first."""

    guild_id, is_admin = _admin_guild(settings, req)
    if guild_id is None:
        return DiscordReviewResponse(status="unlinked")
    if not is_admin:
        return DiscordReviewResponse(status="forbidden")
    service: AgentService = request.app.state.agent_service
    with session_scope(service.session_factory) as session:
        org = get_or_create_org_for_guild(session, guild_id, name=f"Discord {guild_id}")
        articles, total = list_articles(
            session, org_id=org.id, states=REVIEW_STATES, scope=EVERYTHING, limit=REVIEW_PAGE
        )
    return DiscordReviewResponse(
        status="ok",
        total=total,
        articles=[
            ReviewArticle(
                id=a.id,
                kind=a.kind,
                state=a.state,
                issue=_clip(a.issue, _ISSUE_MAX) or "",
                resolution=_clip(a.resolution, _RESOLUTION_MAX),
                demand_count=a.demand_count,
            )
            for a in articles
        ],
    )


@router.post("/review/action", response_model=DiscordReviewActionResponse)
def discord_review_action(
    req: DiscordReviewAction,
    request: Request,
    settings: Settings = Depends(require_bot_secret),
) -> DiscordReviewActionResponse:
    """確認 (verify) or 差し戻し (send back) one draft."""

    guild_id, is_admin = _admin_guild(settings, req)
    if guild_id is None:
        return DiscordReviewActionResponse(status="unlinked")
    if not is_admin:
        return DiscordReviewActionResponse(status="forbidden")
    service: AgentService = request.app.state.agent_service
    now = dt.datetime.now()  # noqa: DTZ005 - naive, like the web's review (knowledge_routes)
    try:
        with session_scope(service.session_factory) as session:
            org = get_or_create_org_for_guild(session, guild_id, name=f"Discord {guild_id}")
            if req.action == "send_back":
                send_back_article(session, req.article_id, org_id=org.id, now=now)
                return DiscordReviewActionResponse(status="sent_back")
            verifier = find_discord_member(session, guild_id, req.discord_user_id)
            verify_article(session, req.article_id, org_id=org.id, verifier_id=verifier, now=now)
            try:
                with session.begin_nested():
                    embed_articles(session, service.embedder, article_ids=[req.article_id])
            except Exception:  # noqa: BLE001 - the check itself must not be lost
                logger.warning(
                    "failed to embed knowledge article %s", req.article_id, exc_info=True
                )
    except ArticleNotFound:
        return DiscordReviewActionResponse(status="gone")
    except ArticleStateError:
        return DiscordReviewActionResponse(status="invalid")
    return DiscordReviewActionResponse(status="verified")

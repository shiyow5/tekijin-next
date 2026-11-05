"""The web's knowledge screen: read verified knowledge, review drafts (#112, R6).

Every member reads their organization's verified knowledge that they may see. The
organization's administrator (the admin account, see :mod:`tekijin.api.org_access`)
also lists the drafts, most wanted first, and edits, verifies, sends back or
retires them. An edited or verified article is embedded at once, so it can answer
the next question.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.api.org_access import org_id_for, scope_for
from tekijin.api.service import AgentService
from tekijin.auth.deps import require_principal
from tekijin.auth.principal import Principal
from tekijin.data.db import session_scope
from tekijin.data.knowledge import (
    REVIEW_STATES,
    VERIFIED,
    ArticleDTO,
    ArticleNotFound,
    ArticleSource,
    ArticleStateError,
    article_sources,
    get_article,
    list_articles,
    retire_article,
    send_back_article,
    update_article,
    verify_article,
)
from tekijin.knowledge.index import embed_articles
from tekijin.models.tables import Employee

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/articles")

_MAX_FIELD = 2000
_REVIEW_ONLY = "知識の確認は管理者だけが行えます。"
_NOT_FOUND = "知識が見つかりません。"


class ArticleItem(BaseModel):
    id: int
    kind: str
    issue: str
    environment: str | None
    resolution: str | None
    cause: str | None
    state: str
    reuse_count: int
    demand_count: int
    last_used_at: dt.datetime | None
    superseded_by: int | None
    author_name: str | None
    verifier_name: str | None
    created_at: dt.datetime
    updated_at: dt.datetime
    verified_at: dt.datetime | None


class ArticleSourceItem(BaseModel):
    kind: Literal["source", "message", "question", "lost"]
    source_item_id: int | None = None
    title: str | None = None
    url: str | None = None
    quote: str | None = None


class ArticleDetail(ArticleItem):
    sources: list[ArticleSourceItem]
    can_review: bool


class ArticleListResponse(BaseModel):
    items: list[ArticleItem]
    total: int
    can_review: bool


class ArticleEditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["case", "procedure", "fact", "decision"] | None = None
    issue: str | None = Field(default=None, min_length=1, max_length=_MAX_FIELD)
    environment: str | None = Field(default=None, max_length=_MAX_FIELD)
    resolution: str | None = Field(default=None, max_length=_MAX_FIELD)
    cause: str | None = Field(default=None, max_length=_MAX_FIELD)


class ArticleRetireRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    superseded_by: int | None = None


def _service(request: Request) -> AgentService:
    service: AgentService = request.app.state.agent_service
    return service


def _require_reviewer(principal: Principal) -> None:
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail=_REVIEW_ONLY)


def _names(session: Session, articles: list[ArticleDTO]) -> dict[int, str]:
    ids = {a.author_id for a in articles} | {a.verifier_id for a in articles}
    ids.discard(None)
    if not ids:
        return {}
    rows = session.execute(select(Employee.id, Employee.name).where(Employee.id.in_(ids))).all()
    return {member_id: name for member_id, name in rows}


def _item(article: ArticleDTO, names: dict[int, str]) -> ArticleItem:
    return ArticleItem(
        id=article.id,
        kind=article.kind,
        issue=article.issue,
        environment=article.environment,
        resolution=article.resolution,
        cause=article.cause,
        state=article.state,
        reuse_count=article.reuse_count,
        demand_count=article.demand_count,
        last_used_at=article.last_used_at,
        superseded_by=article.superseded_by,
        author_name=names.get(article.author_id) if article.author_id is not None else None,
        verifier_name=names.get(article.verifier_id) if article.verifier_id is not None else None,
        created_at=article.created_at,
        updated_at=article.updated_at,
        verified_at=article.verified_at,
    )


def _source_item(source: ArticleSource) -> ArticleSourceItem:
    if source.source_item_id is not None:
        return ArticleSourceItem(
            kind="source",
            source_item_id=source.source_item_id,
            title=source.title,
            url=source.url,
            quote=source.quote,
        )
    if source.channel_message_id is not None:
        title = f"#{source.channel_name}" if source.channel_name else None
        return ArticleSourceItem(kind="message", title=title, url=source.url, quote=source.quote)
    if source.question_id is not None:
        return ArticleSourceItem(kind="question", title="依頼への回答", quote=source.quote)
    return ArticleSourceItem(kind="lost", title="削除された出典", quote=source.quote)


def _detail(session: Session, article: ArticleDTO, principal: Principal) -> ArticleDetail:
    scope = scope_for(principal)
    sources = article_sources(session, [article.id], org_id=article.org_id, scope=scope)
    item = _item(article, _names(session, [article]))
    return ArticleDetail(
        **item.model_dump(),
        sources=[_source_item(s) for s in sources.get(article.id, [])],
        can_review=principal.is_admin,
    )


def _embed(service: AgentService, session: Session, article_id: int) -> None:
    """Embed one article now; a failure leaves it unsearchable but keeps the edit."""

    try:
        with session.begin_nested():
            embed_articles(session, service.embedder, article_ids=[article_id])
    except Exception:  # noqa: BLE001 - the review itself must not be lost
        logger.warning("failed to embed knowledge article %s", article_id, exc_info=True)


@router.get("", response_model=ArticleListResponse)
def list_knowledge_articles(
    request: Request,
    view: Literal["verified", "review"] = "verified",
    q: str | None = Query(default=None, min_length=1, max_length=200),
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    principal: Principal = Depends(require_principal),
) -> ArticleListResponse:
    if view == "review":
        _require_reviewer(principal)
    states = REVIEW_STATES if view == "review" else (VERIFIED,)
    with _service(request).session_factory() as session:
        org_id = org_id_for(session, principal)
        items, total = list_articles(
            session,
            org_id=org_id,
            states=states,
            scope=scope_for(principal),
            q=q,
            offset=offset,
            limit=limit,
        )
        names = _names(session, items)
    return ArticleListResponse(
        items=[_item(a, names) for a in items], total=total, can_review=principal.is_admin
    )


@router.get("/{article_id}", response_model=ArticleDetail)
def knowledge_article(
    article_id: int, request: Request, principal: Principal = Depends(require_principal)
) -> ArticleDetail:
    with _service(request).session_factory() as session:
        org_id = org_id_for(session, principal)
        article = get_article(
            session,
            article_id,
            org_id=org_id,
            scope=scope_for(principal),
            states=None if principal.is_admin else (VERIFIED,),
        )
        if article is None:
            raise HTTPException(status_code=404, detail=_NOT_FOUND)
        return _detail(session, article, principal)


def _review(request: Request, principal: Principal, article_id: int, action) -> ArticleDetail:
    """Run one review action in a transaction and return the article after it."""

    _require_reviewer(principal)
    service = _service(request)
    try:
        with session_scope(service.session_factory) as session:
            org_id = org_id_for(session, principal)
            article = action(session, org_id, service.now())
            if article.state != "retired":
                _embed(service, session, article.id)
            return _detail(session, article, principal)
    except ArticleNotFound as exc:
        raise HTTPException(status_code=404, detail=_NOT_FOUND) from exc
    except ArticleStateError as exc:
        raise HTTPException(
            status_code=409, detail="答えが空の知識は確認済みにできません。"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.patch("/{article_id}", response_model=ArticleDetail)
def edit_knowledge_article(
    article_id: int,
    req: ArticleEditRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> ArticleDetail:
    changes = req.model_dump(exclude_unset=True)
    if changes.get("kind", "") is None or ("issue" in changes and changes["issue"] is None):
        raise HTTPException(status_code=422, detail="kind and issue cannot be empty")
    return _review(
        request,
        principal,
        article_id,
        lambda s, org_id, now: update_article(
            s, article_id, org_id=org_id, changes=changes, now=now
        ),
    )


@router.post("/{article_id}/verify", response_model=ArticleDetail)
def verify_knowledge_article(
    article_id: int, request: Request, principal: Principal = Depends(require_principal)
) -> ArticleDetail:
    return _review(
        request,
        principal,
        article_id,
        lambda s, org_id, now: verify_article(
            s, article_id, org_id=org_id, verifier_id=principal.employee_id, now=now
        ),
    )


@router.post("/{article_id}/send-back", response_model=ArticleDetail)
def send_back_knowledge_article(
    article_id: int, request: Request, principal: Principal = Depends(require_principal)
) -> ArticleDetail:
    return _review(
        request,
        principal,
        article_id,
        lambda s, org_id, now: send_back_article(s, article_id, org_id=org_id, now=now),
    )


@router.post("/{article_id}/retire", response_model=ArticleDetail)
def retire_knowledge_article(
    article_id: int,
    request: Request,
    req: ArticleRetireRequest | None = None,
    principal: Principal = Depends(require_principal),
) -> ArticleDetail:
    superseded_by = req.superseded_by if req is not None else None
    return _review(
        request,
        principal,
        article_id,
        lambda s, org_id, now: retire_article(
            s, article_id, org_id=org_id, superseded_by=superseded_by, now=now
        ),
    )

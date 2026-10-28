"""The web's source endpoints: list, add and remove documents (#72, R2).

A signed-in member works on their own organization's sources. The admin belongs to
no organization (it is a settings account, not a member), so it works on the
organization of the configured Discord server. Anyone in the organization can read
the list; only the member who added a source, or the admin, can delete it.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.api.service import AgentService
from tekijin.auth.deps import require_principal
from tekijin.auth.principal import Principal
from tekijin.config import get_settings
from tekijin.data.db import session_scope
from tekijin.data.sources import (
    MANUAL,
    SourceOwnedElsewhere,
    count_source_items,
    SourceItemSummary,
    delete_source_item,
    get_source_item,
    list_source_items,
    manual_external_id,
    upsert_source_item,
)
from tekijin.models.tables import Employee, Organization, SourceItem

router = APIRouter(prefix="/sources")

# A hand-over document or a manual fits; anything longer should be split up.
_MAX_BODY_LENGTH = 20000


class SourceCreateRequest(BaseModel):
    body: str = Field(min_length=1, max_length=_MAX_BODY_LENGTH)
    title: str | None = Field(default=None, max_length=200)
    url: str | None = Field(default=None, max_length=2000, pattern=r"^https?://\S+$")

    @field_validator("body")
    @classmethod
    def _body_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("body must not be blank")
        return value

    @field_validator("title")
    @classmethod
    def _title_or_none(cls, value: str | None) -> str | None:
        return (value or "").strip() or None


class SourceCreateResponse(BaseModel):
    status: str
    id: int
    chunks: int


class SourceListItem(BaseModel):
    id: int
    provider: str
    title: str | None
    url: str | None
    author_name: str | None
    updated_at: dt.datetime
    chunks: int
    can_delete: bool


class SourceDetail(SourceListItem):
    body: str


class SourceListResponse(BaseModel):
    items: list[SourceListItem]
    # All of the organization's sources, so the client can page past ``limit``.
    total: int


class SourceDeleteResponse(BaseModel):
    status: Literal["deleted"] = "deleted"


def _service(request: Request) -> AgentService:
    service: AgentService = request.app.state.agent_service
    return service


def _org_id(session: Session, principal: Principal) -> int:
    if principal.employee_id is not None:
        member = session.get(Employee, principal.employee_id)
        if member is None or member.org_id is None:
            raise HTTPException(
                status_code=403, detail="組織に属していないため、資料を扱えません。"
            )
        return member.org_id
    guild_id = get_settings().discord_guild_id
    org = (
        session.scalars(
            select(Organization).where(Organization.discord_guild_id == str(guild_id))
        ).first()
        if guild_id
        else None
    )
    if org is None:
        raise HTTPException(
            status_code=409,
            detail="組織がまだありません。Discordのサーバーで一度TEKIJINを使うと作られます。",
        )
    return org.id


@router.get("", response_model=SourceListResponse)
def list_sources(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    principal: Principal = Depends(require_principal),
) -> SourceListResponse:
    with _service(request).session_factory() as session:
        org_id = _org_id(session, principal)
        total = count_source_items(session, org_id=org_id)
        items = list_source_items(session, org_id=org_id, offset=offset, limit=limit)
        names = _author_names(session, items)
    return SourceListResponse(
        items=[_list_item(i, names, principal) for i in items], total=total
    )


@router.get("/{item_id}", response_model=SourceDetail)
def source_detail(
    item_id: int, request: Request, principal: Principal = Depends(require_principal)
) -> SourceDetail:
    with _service(request).session_factory() as session:
        found = get_source_item(session, item_id=item_id, org_id=_org_id(session, principal))
        if found is None:
            raise HTTPException(status_code=404, detail="資料が見つかりません。")
        summary, body = found
        names = _author_names(session, [summary])
    return SourceDetail(**_list_item(summary, names, principal).model_dump(), body=body)


def _author_names(session: Session, items: list[SourceItemSummary]) -> dict[int, str]:
    author_ids = {i.author_id for i in items if i.author_id is not None}
    rows = session.execute(
        select(Employee.id, Employee.name).where(Employee.id.in_(author_ids))
    ).all()
    return {member_id: name for member_id, name in rows}


def _list_item(
    item: SourceItemSummary, names: dict[int, str], principal: Principal
) -> SourceListItem:
    return SourceListItem(
        id=item.id,
        provider=item.provider,
        title=item.title,
        url=item.url,
        author_name=names.get(item.author_id) if item.author_id is not None else None,
        updated_at=item.updated_at,
        chunks=item.chunks,
        can_delete=principal.is_admin
        or (item.author_id is not None and item.author_id == principal.employee_id),
    )


@router.post("", response_model=SourceCreateResponse, status_code=201)
def add_source(
    req: SourceCreateRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> SourceCreateResponse:
    service = _service(request)
    with session_scope(service.session_factory) as session:
        org_id = _org_id(session, principal)
        try:
            result = upsert_source_item(
                session,
                service.embedder,
                org_id=org_id,
                provider=MANUAL,
                external_id=manual_external_id(req.url),
                title=req.title,
                body=req.body,
                url=req.url,
                author_id=principal.employee_id,
                now=service.now(),
                override_owner=principal.is_admin,
            )
        except SourceOwnedElsewhere as exc:
            raise HTTPException(
                status_code=403,
                detail="このURLの資料はほかのメンバーが追加したもので、その人だけが更新できます。",
            ) from exc
    return SourceCreateResponse(status=result.status, id=result.item_id, chunks=result.chunks)


@router.delete("/{item_id}", response_model=SourceDeleteResponse)
def delete_source(
    item_id: int, request: Request, principal: Principal = Depends(require_principal)
) -> SourceDeleteResponse:
    with session_scope(_service(request).session_factory) as session:
        org_id = _org_id(session, principal)
        item = session.get(SourceItem, item_id)
        if item is None or item.org_id != org_id:
            raise HTTPException(status_code=404, detail="資料が見つかりません。")
        if not principal.is_admin and item.author_id != principal.employee_id:
            raise HTTPException(status_code=403, detail="追加した本人だけが削除できます。")
        delete_source_item(session, item_id=item_id, org_id=org_id)
    return SourceDeleteResponse()

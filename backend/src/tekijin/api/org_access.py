"""Which organization a web request works on, and what it may see (#72, #112).

A signed-in member works on their own organization. The admin is a settings
account, not a member, so it works on the organization of the configured Discord
server, and it is that organization's administrator. The web does not know a
member's Discord permissions, so a member sees what everyone in the server can
read; the admin sees everything.
"""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.auth.principal import Principal
from tekijin.config import get_settings
from tekijin.data.visibility import EVERYTHING, PUBLIC, SourceScope
from tekijin.models.tables import Employee, Organization


def org_id_for(session: Session, principal: Principal) -> int:
    """The organization ``principal`` works on, or an HTTP error saying why not."""

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


def scope_for(principal: Principal) -> SourceScope:
    """What ``principal`` may see of the organization's conversations."""

    return EVERYTHING if principal.is_admin else PUBLIC

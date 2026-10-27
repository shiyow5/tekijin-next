"""Organizations, members and their external accounts (#66, ADR-0012).

A Discord user becomes a member the first time TEKIJIN hears from them (``/ask``,
``/tekijin join`` ...): there is no roster to sync against first. The member is an
``employees`` row (renamed in R3) plus an ``identities`` row that maps the Discord
user id to it within the organization.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.models.tables import Employee, Identity, Organization

DISCORD = "discord"


def get_or_create_org_for_guild(session: Session, guild_id: str, *, name: str) -> Organization:
    """The organization for a Discord server, created on first use."""

    org = session.scalars(
        select(Organization).where(Organization.discord_guild_id == guild_id)
    ).first()
    if org is None:
        org = Organization(name=name, discord_guild_id=guild_id)
        session.add(org)
        session.flush()
    return org


def resolve_discord_member(
    session: Session,
    org: Organization,
    discord_user_id: str,
    *,
    display_name: str,
    now: dt.datetime,
) -> Employee:
    """The member behind a Discord user, created on first sight.

    The display name is refreshed on every call so the name shown on hand-offs
    follows what the person is called in the server today.
    """

    identity = session.scalars(
        select(Identity).where(
            Identity.org_id == org.id,
            Identity.provider == DISCORD,
            Identity.external_id == discord_user_id,
        )
    ).first()
    if identity is not None:
        member = session.get(Employee, identity.employee_id)
        if member is not None:
            if display_name and identity.display_name != display_name:
                identity.display_name = display_name
                member.name = display_name
            return member

    member = Employee(name=display_name or f"Discord {discord_user_id}", org_id=org.id)
    session.add(member)
    session.flush()
    session.add(
        Identity(
            org_id=org.id,
            employee_id=member.id,
            provider=DISCORD,
            external_id=discord_user_id,
            display_name=display_name or None,
            linked_at=now,
        )
    )
    session.flush()
    return member


def mark_joined(member: Employee, now: dt.datetime) -> None:
    """``/tekijin join``: first registration time wins."""

    if member.joined_at is None:
        member.joined_at = now


def set_opted_out(member: Employee, opted_out: bool, now: dt.datetime) -> None:
    """``/tekijin optout`` / ``optin``. Opting out keeps the earliest time."""

    if not opted_out:
        member.opted_out_at = None
    elif member.opted_out_at is None:
        member.opted_out_at = now

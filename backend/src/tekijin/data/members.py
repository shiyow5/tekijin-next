"""Organizations, members and their external accounts (#66, ADR-0012).

A Discord user becomes a member the first time TEKIJIN hears from them (``/ask``,
``/tekijin join`` ...): there is no roster to sync against first. The member is an
``employees`` row (renamed in R3) plus an ``identities`` row that maps the Discord
user id to it within the organization.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tekijin.models.tables import Employee, Identity, Organization

DISCORD = "discord"


def get_or_create_org_for_guild(session: Session, guild_id: str, *, name: str) -> Organization:
    """The organization for a Discord server, created on first use.

    Two first requests can race to create it; the loser's insert hits the unique
    guild id, is rolled back to a savepoint, and both end up with the winner's row.
    """

    org = _org_for_guild(session, guild_id)
    if org is not None:
        return org
    try:
        with session.begin_nested():
            session.add(Organization(name=name, discord_guild_id=guild_id))
    except IntegrityError:
        pass
    org = _org_for_guild(session, guild_id)
    assert org is not None  # inserted above, or by the concurrent winner
    return org


def _org_for_guild(session: Session, guild_id: str) -> Organization | None:
    return session.scalars(
        select(Organization).where(Organization.discord_guild_id == guild_id)
    ).first()


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
    follows what the person is called in the server today. Overlapping first
    commands from one user race like :func:`get_or_create_org_for_guild`: the
    loser's member and identity are rolled back together to a savepoint.
    """

    identity = _discord_identity(session, org, discord_user_id)
    if identity is None:
        try:
            with session.begin_nested():
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
            return member
        except IntegrityError:
            identity = _discord_identity(session, org, discord_user_id)
            assert identity is not None  # created by the concurrent winner

    member = session.get(Employee, identity.employee_id)
    assert member is not None  # identities.employee_id is a foreign key
    if display_name and identity.display_name != display_name:
        identity.display_name = display_name
        member.name = display_name
    return member


def _discord_identity(session: Session, org: Organization, discord_user_id: str) -> Identity | None:
    return session.scalars(
        select(Identity).where(
            Identity.org_id == org.id,
            Identity.provider == DISCORD,
            Identity.external_id == discord_user_id,
        )
    ).first()


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

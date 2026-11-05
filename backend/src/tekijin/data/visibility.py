"""Which ingested Discord conversations a reader may see (#85).

Every channel is ingested, so a private channel's conversation must not reach a
member who cannot read that channel. The bot reports which channels everyone can
read and which threads are private; a question from Discord also carries the
asker's own readable channels. A conversation whose channel or thread is unknown
is hidden: the failure mode is a missing answer, never a leak.

CLI (fills the channel and thread of conversations stored before #85)::

    python -m tekijin.data.visibility backfill
"""

from __future__ import annotations

import datetime as dt
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, and_, exists, false, or_, select, true, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from tekijin.models.tables import ChannelMessage, DiscordChannel, DiscordThread, SourceItem

# Same value as ``conversations.DISCORD``; that module imports this one's users.
_DISCORD = "discord"


@dataclass(frozen=True, slots=True)
class SourceScope:
    """What one reader may see.

    ``channels`` are the channels the reader can read; ``None`` means only those
    everyone can read. A private thread is shown only in ``thread_channels``, where
    the reader may manage threads (who was added to a private thread is not known).
    ``everything`` is for the admin and for tools.
    """

    channels: frozenset[int] | None = None
    thread_channels: frozenset[int] = frozenset()
    everything: bool = False


PUBLIC = SourceScope()
EVERYTHING = SourceScope(everything=True)


def visible(org_id: int, scope: SourceScope) -> ColumnElement[bool]:
    """A condition on ``source_items`` that keeps what ``scope`` may see."""

    if scope.everything:
        return true()
    return or_(
        SourceItem.provider != _DISCORD,
        stream_visible(org_id, scope, SourceItem.discord_channel_id, SourceItem.discord_thread_id),
    )


def stream_visible(
    org_id: int, scope: SourceScope, channel_id: Any, thread_id: Any
) -> ColumnElement[bool]:
    """A condition that keeps rows said in a channel (and thread) ``scope`` may read.

    ``channel_id`` and ``thread_id`` are the columns holding where a row was said,
    so other tables than ``source_items`` (self-introductions, #106) follow the
    same rule. A row whose channel is unknown is hidden.
    """

    if scope.everything:
        return true()
    if scope.channels is None:
        readable: Any = channel_id.in_(
            select(DiscordChannel.channel_id).where(
                DiscordChannel.org_id == org_id, DiscordChannel.public.is_(True)
            )
        )
    elif scope.channels:
        readable = channel_id.in_(sorted(scope.channels))
    else:
        readable = false()
    public_thread = exists().where(
        DiscordThread.org_id == org_id,
        DiscordThread.thread_id == thread_id,
        DiscordThread.private.is_(False),
    )
    thread_ok = [thread_id.is_(None), public_thread]
    if scope.thread_channels:
        thread_ok.append(channel_id.in_(sorted(scope.thread_channels)))
    return and_(readable, or_(*thread_ok))


def scope_to_state(scope: SourceScope) -> dict[str, Any] | None:
    """A checkpoint-safe form of ``scope``. ``PUBLIC`` is ``None``."""

    if scope == PUBLIC:
        return None
    return {
        "channels": None if scope.channels is None else sorted(scope.channels),
        "thread_channels": sorted(scope.thread_channels),
        "everything": scope.everything,
    }


def scope_from_state(value: Mapping[str, Any] | None) -> SourceScope:
    if not value:
        return PUBLIC
    channels = value.get("channels")
    return SourceScope(
        channels=None if channels is None else frozenset(int(c) for c in channels),
        thread_channels=frozenset(int(c) for c in value.get("thread_channels") or ()),
        everything=bool(value.get("everything")),
    )


def record_visibility(
    session: Session,
    org_id: int,
    *,
    channels: Mapping[int, bool],
    threads: Mapping[int, bool],
    now: dt.datetime,
) -> None:
    """Store what the bot reported: channel -> everyone can read it, thread ->
    it is private. A later report replaces an earlier one."""

    if channels:
        stmt = insert(DiscordChannel).values(
            [
                {"org_id": org_id, "channel_id": c, "public": p, "updated_at": now}
                for c, p in channels.items()
            ]
        )
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=[DiscordChannel.org_id, DiscordChannel.channel_id],
                set_={"public": stmt.excluded.public, "updated_at": now},
            )
        )
    if threads:
        stmt = insert(DiscordThread).values(
            [
                {"org_id": org_id, "thread_id": t, "private": p, "updated_at": now}
                for t, p in threads.items()
            ]
        )
        session.execute(
            stmt.on_conflict_do_update(
                index_elements=[DiscordThread.org_id, DiscordThread.thread_id],
                set_={"private": stmt.excluded.private, "updated_at": now},
            )
        )


def backfill_discord_streams(session: Session) -> int:
    """Fill the channel and thread of conversations stored without them. Returns
    how many were filled. The caller commits."""

    rows = session.execute(
        select(SourceItem.id, SourceItem.org_id, SourceItem.external_id).where(
            SourceItem.provider == _DISCORD, SourceItem.discord_channel_id.is_(None)
        )
    ).all()
    filled = 0
    for item_id, org_id, key in rows:
        stream = _stream_of(session, org_id, key)
        if stream is None:
            continue
        session.execute(
            update(SourceItem)
            .where(SourceItem.id == item_id)
            .values(discord_channel_id=stream[0], discord_thread_id=stream[1])
        )
        filled += 1
    return filled


def _stream_of(session: Session, org_id: int, key: str) -> tuple[int, int | None] | None:
    kind, _, rest = key.partition(":")
    if kind == "channel":
        return int(rest.partition(":")[0]), None
    if kind == "thread":
        channel = session.scalar(
            select(ChannelMessage.channel_id)
            .where(ChannelMessage.org_id == org_id, ChannelMessage.thread_id == int(rest))
            .limit(1)
        )
        return None if channel is None else (int(channel), int(rest))
    return None


def main(argv: list[str]) -> int:  # pragma: no cover - exercised via CLI
    from tekijin.data.db import get_engine, get_sessionmaker, session_scope

    if argv != ["backfill"]:
        sys.stderr.write("usage: python -m tekijin.data.visibility backfill\n")
        return 2
    with session_scope(get_sessionmaker(get_engine())) as session:
        filled = backfill_discord_streams(session)
    sys.stdout.write(f"filled {filled} conversations\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main(sys.argv[1:]))

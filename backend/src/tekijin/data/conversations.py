"""Discord channel messages, grouped into searchable conversations (#76, R2).

One message is too short to retrieve on its own: "守衛室です" means nothing without
the question before it. Messages are therefore stored one by one in
``channel_messages`` and grouped into conversations — a whole thread, or one
channel's day in Japan time — each kept as a ``source_items`` row (provider
``discord``) with embedded chunks. Whenever a message is added, edited or deleted,
its conversation is rebuilt; a conversation left without messages is removed.

The author of each message becomes a member the first time they are seen, as with
``/ask``. Using messages as evidence of who knows what is R5, behind the
organization's ``expertise_from_messages`` setting.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tekijin.data.members import resolve_discord_member
from tekijin.data.sources import upsert_source_item
from tekijin.models.tables import ChannelMessage, Employee, Organization, SourceItem
from tekijin.retrieval.embedding import Embedder

DISCORD = "discord"
JST = dt.timezone(dt.timedelta(hours=9), "JST")
_UNKNOWN_AUTHOR = "(不明)"


@dataclass(frozen=True, slots=True)
class IncomingMessage:
    """A message as the bot saw it. Times are naive UTC."""

    message_id: int
    channel_id: int
    channel_name: str
    thread_id: int | None
    thread_name: str | None
    author_discord_id: str
    author_name: str
    content: str
    created_at: dt.datetime
    edited_at: dt.datetime | None


@dataclass(frozen=True, slots=True)
class IngestResult:
    stored: int
    # Conversations whose stored text changed.
    conversations: int


def conversation_key(channel_id: int, thread_id: int | None, created_at: dt.datetime) -> str:
    """The ``external_id`` of the conversation a message belongs to."""

    if thread_id is not None:
        return f"thread:{thread_id}"
    return f"channel:{channel_id}:{_jst_date(created_at).isoformat()}"


def _jst_date(utc_naive: dt.datetime) -> dt.date:
    return utc_naive.replace(tzinfo=dt.UTC).astimezone(JST).date()


def _key_of(row: ChannelMessage) -> str:
    return conversation_key(row.channel_id, row.thread_id, row.created_at)


def ingest_messages(
    session: Session,
    embedder: Embedder,
    *,
    org: Organization,
    guild_id: str,
    messages: Sequence[IncomingMessage],
    now: dt.datetime,
) -> IngestResult:
    """Store new or edited messages and rebuild the conversations they touch.

    Re-sending a message unchanged touches nothing, so replaying history after a
    restart is cheap. The caller commits.
    """

    changed: set[str] = set()
    for message in messages:
        member = resolve_discord_member(
            session, org, message.author_discord_id, display_name=message.author_name, now=now
        )
        row = _find(session, org.id, message.message_id)
        if row is None:
            row = _insert(session, org.id, message.message_id)
        elif _same(row, message, member.id):
            continue
        else:
            changed.add(_key_of(row))
        row.channel_id, row.channel_name = message.channel_id, message.channel_name
        row.thread_id, row.thread_name = message.thread_id, message.thread_name
        row.author_id, row.content = member.id, message.content
        row.created_at, row.edited_at = message.created_at, message.edited_at
        changed.add(_key_of(row))
    session.flush()
    rebuilt = sum(_rebuild(session, embedder, org.id, guild_id, key, now) for key in changed)
    return IngestResult(stored=len(messages), conversations=rebuilt)


def _find(session: Session, org_id: int, message_id: int) -> ChannelMessage | None:
    return session.scalars(
        select(ChannelMessage).where(
            ChannelMessage.org_id == org_id, ChannelMessage.message_id == message_id
        )
    ).first()


def _insert(session: Session, org_id: int, message_id: int) -> ChannelMessage:
    """A new row the caller fills in. A concurrent insert of the same message is
    absorbed at a savepoint, and the winner's row is returned instead."""

    try:
        with session.begin_nested():
            created = ChannelMessage(
                org_id=org_id,
                message_id=message_id,
                channel_id=0,
                channel_name="",
                content="",
                created_at=dt.datetime.min,
            )
            session.add(created)
        return created
    except IntegrityError:
        winner = _find(session, org_id, message_id)
        assert winner is not None  # inserted by the concurrent winner
        return winner


def _same(row: ChannelMessage, message: IncomingMessage, author_id: int) -> bool:
    return (
        row.channel_id,
        row.channel_name,
        row.thread_id,
        row.thread_name,
        row.author_id,
        row.content,
        row.created_at,
        row.edited_at,
    ) == (
        message.channel_id,
        message.channel_name,
        message.thread_id,
        message.thread_name,
        author_id,
        message.content,
        message.created_at,
        message.edited_at,
    )


def delete_messages(
    session: Session,
    embedder: Embedder,
    *,
    org_id: int,
    guild_id: str,
    message_ids: Iterable[int],
    now: dt.datetime,
) -> int:
    """Forget deleted messages and rebuild their conversations. Unknown ids are
    ignored. Returns how many stored messages were deleted."""

    rows = session.scalars(
        select(ChannelMessage).where(
            ChannelMessage.org_id == org_id, ChannelMessage.message_id.in_(list(message_ids))
        )
    ).all()
    keys = {_key_of(row) for row in rows}
    for row in rows:
        session.delete(row)
    session.flush()
    for key in keys:
        _rebuild(session, embedder, org_id, guild_id, key, now)
    return len(rows)


def delete_streams(
    session: Session,
    embedder: Embedder,
    *,
    org_id: int,
    guild_id: str,
    stream_ids: Iterable[int],
    now: dt.datetime,
) -> int:
    """A channel or thread was deleted in Discord: forget its messages and the
    conversations made from them (a deleted channel takes its threads with it).
    Returns how many stored messages were deleted."""

    ids = list(stream_ids)
    rows = session.scalars(
        select(ChannelMessage).where(
            ChannelMessage.org_id == org_id,
            (ChannelMessage.channel_id.in_(ids)) | (ChannelMessage.thread_id.in_(ids)),
        )
    ).all()
    return delete_messages(
        session,
        embedder,
        org_id=org_id,
        guild_id=guild_id,
        message_ids=[row.message_id for row in rows],
        now=now,
    )


def delete_conversation(session: Session, *, org_id: int, external_id: str) -> int:
    """Delete a conversation's stored messages along with the conversation itself.

    Deleting only the ``source_items`` row would not last: the next message on
    that day (or in that thread) rebuilds the conversation from every stored
    message. Returns how many messages were deleted.
    """

    rows = session.scalars(
        select(ChannelMessage).where(ChannelMessage.org_id == org_id, *_key_filter(external_id))
    ).all()
    for row in rows:
        session.delete(row)
    session.execute(
        delete(SourceItem).where(
            SourceItem.org_id == org_id,
            SourceItem.provider == DISCORD,
            SourceItem.external_id == external_id,
        )
    )
    return len(rows)


def channel_cursors(session: Session, *, org_id: int) -> dict[int, int]:
    """The last stored message id per channel or thread, where history resumes."""

    stream = func.coalesce(ChannelMessage.thread_id, ChannelMessage.channel_id)
    rows = session.execute(
        select(stream, func.max(ChannelMessage.message_id))
        .where(ChannelMessage.org_id == org_id)
        .group_by(stream)
    ).all()
    return {int(stream_id): int(last) for stream_id, last in rows}


def _rebuild(
    session: Session, embedder: Embedder, org_id: int, guild_id: str, key: str, now: dt.datetime
) -> bool:
    """Rewrite one conversation from its stored messages. True when it changed."""

    rows = session.execute(
        select(ChannelMessage, Employee.name)
        .outerjoin(Employee, Employee.id == ChannelMessage.author_id)
        .where(ChannelMessage.org_id == org_id, *_key_filter(key))
        .order_by(ChannelMessage.created_at, ChannelMessage.message_id)
    ).all()
    if not rows:
        removed = session.execute(
            delete(SourceItem).where(
                SourceItem.org_id == org_id,
                SourceItem.provider == DISCORD,
                SourceItem.external_id == key,
            )
        )
        return bool(removed.rowcount)  # type: ignore[attr-defined]

    first: ChannelMessage = rows[0][0]
    if first.thread_id is not None:
        title = f"#{first.channel_name} › {first.thread_name or ''}".rstrip()
        stream_id = first.thread_id
    else:
        title = f"#{first.channel_name} {_jst_date(first.created_at).isoformat()}"
        stream_id = first.channel_id
    body = "\n".join(f"{name or _UNKNOWN_AUTHOR}: {row.content}" for row, name in rows)
    result = upsert_source_item(
        session,
        embedder,
        org_id=org_id,
        provider=DISCORD,
        external_id=key,
        title=title,
        body=body,
        url=f"https://discord.com/channels/{guild_id}/{stream_id}/{first.message_id}",
        author_id=None,
        now=now,
        # A conversation has many authors and belongs to none of them.
        override_owner=True,
    )
    return result.status != "unchanged"


def _key_filter(key: str) -> tuple:
    kind, _, rest = key.partition(":")
    if kind == "thread":
        return (ChannelMessage.thread_id == int(rest),)
    channel, _, day = rest.partition(":")
    start = dt.datetime.combine(dt.date.fromisoformat(day), dt.time(), JST)
    start_utc = start.astimezone(dt.UTC).replace(tzinfo=None)
    return (
        ChannelMessage.thread_id.is_(None),
        ChannelMessage.channel_id == int(channel),
        ChannelMessage.created_at >= start_utc,
        ChannelMessage.created_at < start_utc + dt.timedelta(days=1),
    )

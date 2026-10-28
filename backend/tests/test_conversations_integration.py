"""Integration tests for ingesting Discord channel messages as conversations (#76, R2).

Messages are stored one by one in ``channel_messages`` and grouped into
conversations — a whole thread, or one channel's day in Japan time — which are
``source_items`` with embedded chunks. Any change to a message rebuilds its
conversation; a conversation without messages is removed.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select, text

from tekijin.data.conversations import (
    IncomingMessage,
    channel_cursors,
    delete_messages,
    ingest_messages,
)
from tekijin.data.db import get_sessionmaker
from tekijin.models.tables import ChannelMessage, Employee, Organization, SourceItem

GUILD = "424242"
NOW = dt.datetime(2026, 10, 28, 12, 0, 0)


@pytest.fixture
def org_id(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        org = Organization(name="サークル", discord_guild_id=GUILD)
        session.add(org)
        session.commit()
        org_id = org.id
    yield org_id
    with factory() as session:
        session.execute(text("DELETE FROM chunks"))
        session.execute(text("DELETE FROM source_items"))
        session.execute(text("DELETE FROM channel_messages"))
        session.execute(text("DELETE FROM identities"))
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


def _msg(message_id: int, content: str, *, at: dt.datetime, author="1", name="アリス", **kw):
    return IncomingMessage(
        message_id=message_id,
        channel_id=kw.get("channel_id", 10),
        channel_name=kw.get("channel_name", "雑談"),
        thread_id=kw.get("thread_id"),
        thread_name=kw.get("thread_name"),
        author_discord_id=author,
        author_name=name,
        content=content,
        created_at=at,
        edited_at=kw.get("edited_at"),
    )


def _ingest(session, embedder, org_id, messages):
    org = session.get(Organization, org_id)
    return ingest_messages(session, embedder, org=org, guild_id=GUILD, messages=messages, now=NOW)


def _delete(session, embedder, org_id, ids):
    return delete_messages(
        session, embedder, org_id=org_id, guild_id=GUILD, message_ids=ids, now=NOW
    )


def _items(session) -> dict[str, SourceItem]:
    return {i.external_id: i for i in session.scalars(select(SourceItem))}


# 03:00 UTC is 12:00 in Japan.
NOON = dt.datetime(2026, 10, 28, 3, 0, 0)


def test_one_channel_day_becomes_one_conversation(engine, fake_embedder, org_id) -> None:
    with get_sessionmaker(engine)() as session:
        result = _ingest(
            session,
            fake_embedder,
            org_id,
            [
                _msg(1002, "守衛室 です", at=NOON + dt.timedelta(minutes=1), author="2", name="ボブ"),
                _msg(1001, "部室 の 鍵 は？", at=NOON),
            ],
        )
        session.commit()
        assert (result.stored, result.conversations) == (2, 1)
        item = _items(session)["channel:10:2026-10-28"]
        assert item.provider == "discord"
        assert item.title == "#雑談 2026-10-28"
        assert item.body == "アリス: 部室 の 鍵 は？\nボブ: 守衛室 です"
        assert item.url == f"https://discord.com/channels/{GUILD}/10/1001"
        # Authors became members of the organization.
        names = set(session.scalars(select(Employee.name).where(Employee.org_id == org_id)))
        assert names == {"アリス", "ボブ"}


def test_the_day_follows_japan_time(engine, fake_embedder, org_id) -> None:
    late = dt.datetime(2026, 10, 28, 14, 30)  # 23:30 in Japan
    after_midnight = dt.datetime(2026, 10, 28, 15, 30)  # 00:30 the next day
    with get_sessionmaker(engine)() as session:
        _ingest(
            session,
            fake_embedder,
            org_id,
            [_msg(2001, "おやすみ", at=late), _msg(2002, "おはよう", at=after_midnight)],
        )
        session.commit()
        assert set(_items(session)) == {"channel:10:2026-10-28", "channel:10:2026-10-29"}


def test_a_thread_is_one_conversation(engine, fake_embedder, org_id) -> None:
    thread = {"thread_id": 77, "thread_name": "合宿の宿"}
    with get_sessionmaker(engine)() as session:
        _ingest(
            session,
            fake_embedder,
            org_id,
            [
                _msg(3001, "宿 どこ？", at=NOON, **thread),
                _msg(3002, "去年 と 同じ", at=NOON + dt.timedelta(days=2), **thread),
            ],
        )
        session.commit()
        item = _items(session)["thread:77"]
        assert item.title == "#雑談 › 合宿の宿"
        assert item.body == "アリス: 宿 どこ？\nアリス: 去年 と 同じ"
        assert item.url == f"https://discord.com/channels/{GUILD}/77/3001"


def test_an_edit_rebuilds_and_a_resend_changes_nothing(engine, fake_embedder, org_id) -> None:
    with get_sessionmaker(engine)() as session:
        _ingest(session, fake_embedder, org_id, [_msg(4001, "鍵 は 守衛室", at=NOON)])
        session.commit()
        same = _ingest(session, fake_embedder, org_id, [_msg(4001, "鍵 は 守衛室", at=NOON)])
        assert same.conversations == 0
        edited = _msg(4001, "鍵 は 事務室", at=NOON, edited_at=NOON + dt.timedelta(hours=1))
        _ingest(session, fake_embedder, org_id, [edited])
        session.commit()
        assert _items(session)["channel:10:2026-10-28"].body == "アリス: 鍵 は 事務室"
        stored = session.scalars(select(ChannelMessage)).one()
        assert stored.edited_at == NOON + dt.timedelta(hours=1)


def test_deleting_messages_rebuilds_then_removes_the_conversation(
    engine, fake_embedder, org_id
) -> None:
    with get_sessionmaker(engine)() as session:
        _ingest(
            session,
            fake_embedder,
            org_id,
            [_msg(5001, "一つ目", at=NOON), _msg(5002, "二つ目", at=NOON + dt.timedelta(minutes=1))],
        )
        session.commit()
        assert _delete(session, fake_embedder, org_id, [5001]) == 1
        session.commit()
        assert _items(session)["channel:10:2026-10-28"].body == "アリス: 二つ目"
        # Unknown ids are ignored.
        assert _delete(session, fake_embedder, org_id, [5002, 9999]) == 1
        session.commit()
        assert _items(session) == {}
        assert session.scalars(select(ChannelMessage)).all() == []


def test_cursors_give_the_last_message_per_channel_or_thread(
    engine, fake_embedder, org_id
) -> None:
    with get_sessionmaker(engine)() as session:
        _ingest(
            session,
            fake_embedder,
            org_id,
            [
                _msg(6001, "a", at=NOON),
                _msg(6005, "b", at=NOON),
                _msg(6003, "c", at=NOON, channel_id=11, channel_name="運営"),
                _msg(6004, "d", at=NOON, thread_id=78, thread_name="t"),
            ],
        )
        session.commit()
        assert channel_cursors(session, org_id=org_id) == {10: 6005, 11: 6003, 78: 6004}

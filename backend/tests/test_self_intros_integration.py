"""Self-introductions stored, replaced, deleted and searched (#106, R5).

Every test runs inside one transaction that is rolled back, so nothing it writes
outlives it and no shared table needs cleaning.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select

from tekijin.data.conversations import (
    IncomingMessage,
    delete_conversation,
    delete_messages,
    ingest_messages,
)
from tekijin.data.db import get_sessionmaker
from tekijin.data.members import resolve_discord_member, set_opted_out
from tekijin.data.self_intros import (
    IntroHit,
    delete_member_intros,
    search_self_intros,
    set_join_intro,
)
from tekijin.data.sources import delete_source_item, upsert_source_item
from tekijin.data.visibility import EVERYTHING, PUBLIC, SourceScope, record_visibility
from tekijin.models.tables import Employee, Organization, SelfIntro
from tekijin.retrieval.bm25_cache import clear_bm25_cache

NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
GUILD = "106106106"
INTRO_CHANNEL, CHAT_CHANNEL, SECRET_CHANNEL = 501, 502, 503
SLIDES = "スライド1 山田 太郎\n会計 補助金 申請\n\nスライド2 鈴木 一郎\n望遠鏡 赤道儀 修理\n"


@pytest.fixture
def session(engine):
    clear_bm25_cache()
    factory = get_sessionmaker(engine)
    with factory() as session:
        yield session
        session.rollback()
    clear_bm25_cache()


@pytest.fixture
def org(session) -> Organization:
    org = Organization(name="自己紹介のテスト", discord_guild_id=GUILD)
    session.add(org)
    session.flush()
    return org


def _member(session, org: Organization, discord_id: str, name: str) -> Employee:
    return resolve_discord_member(session, org, discord_id, display_name=name, now=NOW)


def _doc(session, embedder, org: Organization, *, body=SLIDES, title="2026年度 自己紹介"):
    return upsert_source_item(
        session,
        embedder,
        org_id=org.id,
        provider="manual",
        external_id="url:https://example.test/intro",
        title=title,
        body=body,
        url="https://example.test/intro",
        author_id=None,
        now=NOW,
        override_owner=True,
    )


def _intros(session, org: Organization) -> list[SelfIntro]:
    return list(
        session.scalars(
            select(SelfIntro).where(SelfIntro.org_id == org.id).order_by(SelfIntro.id)
        ).all()
    )


def _texts(session, org: Organization) -> dict[tuple[int, str], str]:
    return {(row.member_id, row.origin): row.text for row in _intros(session, org)}


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def test_an_intro_document_gives_each_named_member_a_row(session, fake_embedder, org) -> None:
    yamada = _member(session, org, "1", "山田 太郎")
    suzuki = _member(session, org, "2", "鈴木一郎")
    _member(session, org, "3", "佐々木 花")  # not on the slides

    result = _doc(session, fake_embedder, org)

    rows = _intros(session, org)
    assert {(r.member_id, r.origin, r.source_item_id) for r in rows} == {
        (yamada.id, "slides", result.item_id),
        (suzuki.id, "slides", result.item_id),
    }
    assert all(r.embedding is not None for r in rows)
    assert "補助金" in _texts(session, org)[(yamada.id, "slides")]


def test_a_document_not_titled_intro_gives_none(session, fake_embedder, org) -> None:
    _member(session, org, "1", "山田 太郎")

    _doc(session, fake_embedder, org, title="会計の引継ぎ書")

    assert _intros(session, org) == []


def test_updating_the_document_replaces_its_rows(session, fake_embedder, org) -> None:
    yamada = _member(session, org, "1", "山田 太郎")
    suzuki = _member(session, org, "2", "鈴木一郎")
    _doc(session, fake_embedder, org)
    suzuki_row = next(r for r in _intros(session, org) if r.member_id == suzuki.id)
    suzuki_vector = list(suzuki_row.embedding)

    _doc(session, fake_embedder, org, body="スライド2 鈴木 一郎\n望遠鏡 赤道儀 修理\n")

    rows = _intros(session, org)
    assert [r.member_id for r in rows] == [suzuki.id]
    # Unchanged text keeps its row and vector.
    assert rows[0].id == suzuki_row.id and list(rows[0].embedding) == suzuki_vector
    assert (yamada.id, "slides") not in _texts(session, org)


def test_retitling_away_from_intro_removes_the_rows(session, fake_embedder, org) -> None:
    _member(session, org, "1", "山田 太郎")
    _doc(session, fake_embedder, org)

    _doc(session, fake_embedder, org, title="会計の資料")

    assert _intros(session, org) == []


def test_an_unchanged_document_fills_in_a_member_who_arrived_later(
    session, fake_embedder, org
) -> None:
    _member(session, org, "1", "山田 太郎")
    _doc(session, fake_embedder, org)
    suzuki = _member(session, org, "2", "鈴木一郎")

    assert _doc(session, fake_embedder, org).status == "unchanged"

    assert (suzuki.id, "slides") in _texts(session, org)


def test_deleting_the_document_deletes_its_rows(session, fake_embedder, org) -> None:
    _member(session, org, "1", "山田 太郎")
    item_id = _doc(session, fake_embedder, org).item_id

    delete_source_item(session, item_id=item_id, org_id=org.id)
    session.flush()
    session.expire_all()

    assert _intros(session, org) == []


# --------------------------------------------------------------------------- #
# /tekijin join and /tekijin intro
# --------------------------------------------------------------------------- #
def test_the_join_intro_is_one_row_that_is_replaced(session, fake_embedder, org) -> None:
    member = _member(session, org, "1", "山田 太郎")

    assert set_join_intro(session, fake_embedder, member=member, text=" 会計です ", now=NOW)
    assert set_join_intro(session, fake_embedder, member=member, text="写真を撮ります", now=NOW)

    rows = _intros(session, org)
    assert [(r.origin, r.text) for r in rows] == [("join", "写真を撮ります")]


def test_an_opted_out_member_cannot_set_an_intro(session, fake_embedder, org) -> None:
    member = _member(session, org, "1", "山田 太郎")
    set_opted_out(member, True, NOW)

    assert not set_join_intro(session, fake_embedder, member=member, text="会計です", now=NOW)
    assert _intros(session, org) == []


# --------------------------------------------------------------------------- #
# Discord introduction channel
# --------------------------------------------------------------------------- #
def _msg(message_id: int, content: str, *, author="1", name="山田 太郎", **kw) -> IncomingMessage:
    return IncomingMessage(
        message_id=message_id,
        channel_id=kw.get("channel_id", INTRO_CHANNEL),
        channel_name=kw.get("channel_name", "自己紹介"),
        thread_id=kw.get("thread_id"),
        thread_name=kw.get("thread_name"),
        author_discord_id=author,
        author_name=name,
        content=content,
        created_at=kw.get("at", dt.datetime(2026, 11, 1, 3, 0)),
        edited_at=kw.get("edited_at"),
    )


def _ingest(session, embedder, org, messages) -> None:
    ingest_messages(session, embedder, org=org, guild_id=GUILD, messages=messages, now=NOW)


def test_posts_in_the_intro_channel_become_intros(session, fake_embedder, org) -> None:
    _ingest(
        session,
        fake_embedder,
        org,
        [
            _msg(1, "会計を担当しています"),
            _msg(2, "望遠鏡の修理が得意です", author="2", name="鈴木一郎"),
            _msg(3, "今日は晴れ", channel_id=CHAT_CHANNEL, channel_name="雑談"),
            # A reply in a thread under the intro channel is not an introduction...
            _msg(4, "よろしく!", author="2", name="鈴木一郎", thread_id=900, thread_name="x"),
            # ...but a forum post's first message is.
            _msg(901, "写真班です", author="3", name="高橋", thread_id=901, thread_name="高橋"),
        ],
    )

    rows = {r.message_id: r for r in _intros(session, org)}
    assert set(rows) == {1, 2, 901}
    assert rows[1].origin == "discord" and rows[1].text == "会計を担当しています"
    assert (rows[901].discord_channel_id, rows[901].discord_thread_id) == (INTRO_CHANNEL, 901)


def test_an_edit_updates_and_a_delete_removes_the_intro(session, fake_embedder, org) -> None:
    _ingest(session, fake_embedder, org, [_msg(1, "会計です"), _msg(2, "写真班です")])

    _ingest(session, fake_embedder, org, [_msg(1, "会計と補助金の申請", edited_at=NOW)])
    delete_messages(session, fake_embedder, org_id=org.id, guild_id=GUILD, message_ids=[2], now=NOW)

    assert [(r.message_id, r.text) for r in _intros(session, org)] == [(1, "会計と補助金の申請")]


def test_hiding_the_conversation_removes_its_intros(session, fake_embedder, org) -> None:
    _ingest(session, fake_embedder, org, [_msg(1, "会計です")])
    key = f"channel:{INTRO_CHANNEL}:2026-11-01"

    delete_conversation(session, org_id=org.id, external_id=key, now=NOW)
    session.flush()
    # Replaying the hidden message does not bring it back.
    _ingest(session, fake_embedder, org, [_msg(1, "会計です")])

    assert _intros(session, org) == []


def test_replaying_history_fills_in_intros_of_stored_posts(session, fake_embedder, org) -> None:
    _ingest(session, fake_embedder, org, [_msg(1, "会計です")])
    for row in _intros(session, org):
        session.delete(row)
    session.flush()

    _ingest(session, fake_embedder, org, [_msg(1, "会計です")])

    assert [r.message_id for r in _intros(session, org)] == [1]


# --------------------------------------------------------------------------- #
# optout
# --------------------------------------------------------------------------- #
def test_optout_deletes_every_intro_and_blocks_new_ones(session, fake_embedder, org) -> None:
    yamada = _member(session, org, "1", "山田 太郎")
    _doc(session, fake_embedder, org)
    set_join_intro(session, fake_embedder, member=yamada, text="会計です", now=NOW)
    _ingest(session, fake_embedder, org, [_msg(1, "会計を担当しています")])
    assert {o for m, o in _texts(session, org) if m == yamada.id} == {"slides", "join", "discord"}

    set_opted_out(yamada, True, NOW)
    assert delete_member_intros(session, member_id=yamada.id) == 3
    # Neither the next upsert of the document nor a new post brings them back.
    suzuki = _member(session, org, "2", "鈴木一郎")
    _doc(session, fake_embedder, org)
    _ingest(session, fake_embedder, org, [_msg(2, "まだ書きます")])

    assert {member_id for member_id, _ in _texts(session, org)} == {suzuki.id}


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def _search(session, embedder, org, query: str, scope: SourceScope = PUBLIC) -> list[IntroHit]:
    vector = embedder.encode([query], kind="query")[0]
    return search_self_intros(session, org_id=org.id, query=query, query_vector=vector, scope=scope)


def test_search_ranks_the_matching_member_first(session, fake_embedder, org) -> None:
    _member(session, org, "1", "山田 太郎")
    suzuki = _member(session, org, "2", "鈴木一郎")
    _doc(session, fake_embedder, org)

    hits = _search(session, fake_embedder, org, "望遠鏡 赤道儀 修理")

    assert hits[0].member_id == suzuki.id and hits[0].origin == "slides"
    assert hits[0].cosine > hits[-1].cosine or len(hits) == 1
    assert hits[0].score >= hits[-1].score
    assert 0.0 <= hits[0].cosine <= 1.0 + 1e-6


def test_search_leaves_out_opted_out_and_inactive_members(session, fake_embedder, org) -> None:
    yamada = _member(session, org, "1", "山田 太郎")
    suzuki = _member(session, org, "2", "鈴木一郎")
    _doc(session, fake_embedder, org)

    yamada.opted_out_at = NOW  # opted out without the rows being deleted yet
    suzuki.is_active = False
    session.flush()

    assert _search(session, fake_embedder, org, "望遠鏡 会計") == []


def test_search_shows_discord_intros_only_to_readers(session, fake_embedder, org) -> None:
    _ingest(
        session,
        fake_embedder,
        org,
        [
            _msg(1, "望遠鏡 赤道儀 修理"),
            _msg(
                2,
                "望遠鏡 赤道儀 修理",
                author="2",
                name="鈴木一郎",
                channel_id=SECRET_CHANNEL,
                channel_name="自己紹介-幹部",
            ),
        ],
    )
    record_visibility(
        session, org.id, channels={INTRO_CHANNEL: True, SECRET_CHANNEL: False}, threads={}, now=NOW
    )
    session.flush()

    def members(scope: SourceScope) -> set[int]:
        return {h.member_id for h in _search(session, fake_embedder, org, "望遠鏡", scope)}

    yamada = _member(session, org, "1", "山田 太郎")
    suzuki = _member(session, org, "2", "鈴木一郎")
    assert members(PUBLIC) == {yamada.id}
    assert members(SourceScope(channels=frozenset({SECRET_CHANNEL}))) == {suzuki.id}
    assert members(EVERYTHING) == {yamada.id, suzuki.id}


def test_slides_and_join_intros_are_visible_to_everyone(session, fake_embedder, org) -> None:
    yamada = _member(session, org, "1", "山田 太郎")
    _member(session, org, "2", "鈴木一郎")
    _doc(session, fake_embedder, org)
    set_join_intro(session, fake_embedder, member=yamada, text="望遠鏡も少し触ります", now=NOW)

    origins = {
        h.origin
        for h in _search(session, fake_embedder, org, "望遠鏡", SourceScope(channels=frozenset()))
    }

    assert origins == {"slides", "join"}


def test_search_is_limited_to_the_organization(session, fake_embedder, org) -> None:
    _member(session, org, "1", "山田 太郎")
    _doc(session, fake_embedder, org)
    other = Organization(name="ほか", discord_guild_id="106106107")
    session.add(other)
    session.flush()

    vector = fake_embedder.encode(["会計"], kind="query")[0]
    assert (
        search_self_intros(
            session, org_id=other.id, query="会計", query_vector=vector, scope=PUBLIC
        )
        == []
    )

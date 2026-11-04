"""Members see only the Discord conversations their channel permissions allow (#85)."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select, text, update

from tekijin.data.conversations import IncomingMessage, ingest_messages
from tekijin.data.db import get_sessionmaker
from tekijin.data.sources import (
    count_source_items,
    get_source_item,
    list_source_items,
    search_chunks,
    upsert_source_item,
)
from tekijin.data.visibility import (
    EVERYTHING,
    PUBLIC,
    SourceScope,
    backfill_discord_streams,
    record_visibility,
    scope_from_state,
    scope_to_state,
)
from tekijin.models.tables import Organization, SourceItem
from tekijin.retrieval.sources import ChunkSearch

NOW = dt.datetime(2026, 11, 4, 12, 0, 0)
GUILD = "777"
OPEN, CLOSED = 101, 102
PUBLIC_THREAD, PRIVATE_THREAD = 201, 202
WORDS = "部室 鍵 守衛室 借りる"


@pytest.fixture
def org_id(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        org = Organization(name="テストサークル", discord_guild_id=GUILD)
        session.add(org)
        session.commit()
        value = org.id
    yield value
    with factory() as session:
        for table in (
            "chunks",
            "source_items",
            "channel_messages",
            "discord_channels",
            "discord_threads",
            "identities",
        ):
            session.execute(text(f"DELETE FROM {table}"))
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


def _message(message_id, channel_id, thread_id=None, content=WORDS):
    return IncomingMessage(
        message_id=message_id,
        channel_id=channel_id,
        channel_name=f"c{channel_id}",
        thread_id=thread_id,
        thread_name=f"t{thread_id}" if thread_id else None,
        author_discord_id="5001",
        author_name="会計係",
        content=content,
        created_at=NOW,
        edited_at=None,
    )


def _seed(session, embedder, org_id):
    org = session.get(Organization, org_id)
    ingest_messages(
        session,
        embedder,
        org=org,
        guild_id=GUILD,
        messages=[
            _message(1, OPEN),
            _message(2, CLOSED),
            _message(3, OPEN, PUBLIC_THREAD),
            _message(4, OPEN, PRIVATE_THREAD),
        ],
        now=NOW,
    )
    upsert_source_item(
        session,
        embedder,
        org_id=org_id,
        provider="manual",
        external_id="note:1",
        title="部室",
        body=WORDS,
        url=None,
        author_id=None,
        now=NOW,
    )
    record_visibility(
        session,
        org_id,
        channels={OPEN: True, CLOSED: False},
        threads={PUBLIC_THREAD: False, PRIVATE_THREAD: True},
        now=NOW,
    )
    session.flush()


def _seen(session, org_id, scope):
    return {
        (item.provider, item.title)
        for item in list_source_items(session, org_id=org_id, scope=scope)
    }


def test_a_conversation_remembers_its_channel_and_thread(engine, fake_embedder, org_id):
    with get_sessionmaker(engine)() as session:
        _seed(session, fake_embedder, org_id)
        rows = session.execute(
            select(SourceItem.discord_channel_id, SourceItem.discord_thread_id).where(
                SourceItem.provider == "discord"
            )
        ).all()

    assert set(rows) == {
        (OPEN, None),
        (CLOSED, None),
        (OPEN, PUBLIC_THREAD),
        (OPEN, PRIVATE_THREAD),
    }


def test_the_public_scope_shows_what_everyone_can_read(engine, fake_embedder, org_id):
    with get_sessionmaker(engine)() as session:
        _seed(session, fake_embedder, org_id)
        seen = _seen(session, org_id, PUBLIC)
        total = count_source_items(session, org_id=org_id, scope=PUBLIC)

    titles = {title for _, title in seen}
    assert ("manual", "部室") in seen
    assert any(title.startswith(f"#c{OPEN} 2026") for title in titles)
    assert f"#c{OPEN} › t{PUBLIC_THREAD}" in titles
    assert f"#c{OPEN} › t{PRIVATE_THREAD}" not in titles
    assert not any(title.startswith(f"#c{CLOSED}") for title in titles)
    assert total == len(seen) == 3


def test_a_member_sees_the_channels_they_can_read(engine, fake_embedder, org_id):
    reader = SourceScope(channels=frozenset({CLOSED}))
    manager = SourceScope(channels=frozenset({OPEN}), thread_channels=frozenset({OPEN}))
    with get_sessionmaker(engine)() as session:
        _seed(session, fake_embedder, org_id)
        as_reader = {title for _, title in _seen(session, org_id, reader)}
        as_manager = {title for _, title in _seen(session, org_id, manager)}

    assert any(title.startswith(f"#c{CLOSED}") for title in as_reader)
    assert not any(title.startswith(f"#c{OPEN}") for title in as_reader)
    assert f"#c{OPEN} › t{PRIVATE_THREAD}" in as_manager


def test_unknown_permissions_hide_the_conversation(engine, fake_embedder, org_id):
    with get_sessionmaker(engine)() as session:
        _seed(session, fake_embedder, org_id)
        org = session.get(Organization, org_id)
        ingest_messages(
            session,
            fake_embedder,
            org=org,
            guild_id=GUILD,
            messages=[_message(5, 103), _message(6, OPEN, 203)],
            now=NOW,
        )
        session.flush()
        titles = {title for _, title in _seen(session, org_id, PUBLIC)}
        everything = {title for _, title in _seen(session, org_id, EVERYTHING)}

    assert not any("c103" in title or "t203" in title for title in titles)
    assert {"#c101 › t203"} <= everything


def test_detail_follows_the_same_rule(engine, fake_embedder, org_id):
    with get_sessionmaker(engine)() as session:
        _seed(session, fake_embedder, org_id)
        closed = session.scalars(
            select(SourceItem.id).where(SourceItem.discord_channel_id == CLOSED)
        ).one()
        hidden = get_source_item(session, item_id=closed, org_id=org_id, scope=PUBLIC)
        shown = get_source_item(session, item_id=closed, org_id=org_id, scope=EVERYTHING)

    assert hidden is None
    assert shown is not None


def test_search_and_evidence_follow_the_scope(engine, fake_embedder, org_id):
    with get_sessionmaker(engine)() as session:
        _seed(session, fake_embedder, org_id)
        vector = fake_embedder.encode([WORDS])[0]
        dense = search_chunks(session, org_id=org_id, query_vec=vector, scope=PUBLIC, top_k=20)
        search = ChunkSearch(session)
        public = search.search(WORDS, org_id, vector, PUBLIC)
        everything = search.search(WORDS, org_id, vector, EVERYTHING)
        evidence = search.evidence(everything.hits, org_id, PUBLIC)
        visible_ids = {i.id for i in list_source_items(session, org_id=org_id, scope=PUBLIC)}

    assert {h.source_item_id for h in dense} == visible_ids
    assert {h["source_item_id"] for h in public.hits} == visible_ids
    assert len({h["source_item_id"] for h in everything.hits}) == 5
    assert {e.source_id for e in evidence} == {f"src_{i}" for i in visible_ids}


def test_keyword_hits_are_filtered_too(engine, fake_embedder, org_id):
    with get_sessionmaker(engine)() as session:
        org = session.get(Organization, org_id)
        ingest_messages(
            session,
            fake_embedder,
            org=org,
            guild_id=GUILD,
            messages=[_message(7, CLOSED, content="EM200 赤道儀")],
            now=NOW,
        )
        record_visibility(session, org_id, channels={CLOSED: False}, threads={}, now=NOW)
        session.flush()
        far = fake_embedder.encode(["まったく 別の 話"])[0]
        result = ChunkSearch(session).search("EM200", org_id, far, PUBLIC)

    assert result.hits == []


def test_recording_again_replaces_the_previous_state(engine, fake_embedder, org_id):
    with get_sessionmaker(engine)() as session:
        _seed(session, fake_embedder, org_id)
        record_visibility(session, org_id, channels={CLOSED: True}, threads={}, now=NOW)
        session.flush()
        titles = {title for _, title in _seen(session, org_id, PUBLIC)}

    assert any(title.startswith(f"#c{CLOSED}") for title in titles)


def test_backfill_fills_conversations_stored_before_the_columns(engine, fake_embedder, org_id):
    with get_sessionmaker(engine)() as session:
        _seed(session, fake_embedder, org_id)
        session.execute(update(SourceItem).values(discord_channel_id=None, discord_thread_id=None))
        filled = backfill_discord_streams(session)
        rows = set(
            session.execute(
                select(SourceItem.discord_channel_id, SourceItem.discord_thread_id).where(
                    SourceItem.provider == "discord"
                )
            ).all()
        )

    assert filled == 4
    assert (OPEN, PRIVATE_THREAD) in rows and (CLOSED, None) in rows


def test_scope_survives_the_checkpoint() -> None:
    scope = SourceScope(channels=frozenset({1, 2}), thread_channels=frozenset({2}))
    assert scope_from_state(scope_to_state(scope)) == scope
    assert scope_from_state(None) == PUBLIC
    assert scope_from_state(scope_to_state(PUBLIC)) == PUBLIC

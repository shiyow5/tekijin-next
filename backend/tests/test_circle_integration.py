"""Seeding the fictional circle into the database (#79)."""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest
from sqlalchemy import func, select, text

from tekijin.data.circle import GUILD_ID, external_id, load_circle, seed_circle
from tekijin.data.db import get_sessionmaker
from tekijin.models.tables import ChannelMessage, Chunk, Employee, Organization, SourceItem

NOW = dt.datetime(2026, 11, 1, 12, 0, 0)


@pytest.fixture
def session(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        yield session
        session.rollback()
    with factory() as cleanup:
        cleanup.execute(text("DELETE FROM chunks"))
        cleanup.execute(text("DELETE FROM source_items"))
        cleanup.execute(text("DELETE FROM channel_messages"))
        cleanup.execute(text("DELETE FROM identities"))
        cleanup.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        cleanup.execute(text("DELETE FROM organizations"))
        cleanup.commit()


def _count(session, model, org_id: int) -> int:
    return session.scalar(select(func.count()).select_from(model).where(model.org_id == org_id))


def test_seed_stores_the_whole_circle(session, fake_embedder) -> None:
    circle = load_circle()
    result = seed_circle(session, fake_embedder, circle, now=NOW)
    session.flush()

    org = session.scalars(
        select(Organization).where(Organization.discord_guild_id == GUILD_ID)
    ).one()
    assert result.org_id == org.id
    assert _count(session, Employee, org.id) == len(circle.members)
    lines = sum(len(c["lines"]) for c in circle.conversations)
    assert _count(session, ChannelMessage, org.id) == lines
    assert _count(session, SourceItem, org.id) == len(circle.conversations) + len(circle.documents)
    assert _count(session, Chunk, org.id) >= _count(session, SourceItem, org.id)


def test_seed_twice_changes_nothing(session, fake_embedder) -> None:
    circle = load_circle()
    seed_circle(session, fake_embedder, circle, now=NOW)
    session.flush()
    chunks = session.scalars(select(Chunk.id).order_by(Chunk.id)).all()

    again = seed_circle(session, fake_embedder, circle, now=NOW + dt.timedelta(days=1))
    session.flush()

    assert again.conversations_changed == 0
    assert again.documents_changed == 0
    assert session.scalars(select(Chunk.id).order_by(Chunk.id)).all() == chunks


def test_every_expected_source_exists(session, fake_embedder) -> None:
    circle = load_circle()
    org_id = seed_circle(session, fake_embedder, circle, now=NOW).org_id
    session.flush()
    stored = set(session.scalars(select(SourceItem.external_id).where(SourceItem.org_id == org_id)))
    refs = {r for q in circle.questions for r in q.get("sources", [])}
    assert {external_id(circle, ref) for ref in refs} <= stored


def test_what_the_fixtures_dropped_is_forgotten(session, fake_embedder) -> None:
    circle = load_circle()
    org_id = seed_circle(session, fake_embedder, circle, now=NOW).org_id
    session.flush()
    gone_conversation = external_id(circle, "conv:chat-lunch")
    gone_document = external_id(circle, "doc:camp-handover")
    conversations = [c for c in circle.conversations if c["key"] != "chat-lunch"]
    focus = next(c for c in conversations if c["key"] == "focus")
    shorter = {**focus, "lines": focus["lines"][:-1]}
    smaller = dataclasses.replace(
        circle,
        conversations=[shorter if c is focus else c for c in conversations],
        documents=[d for d in circle.documents if d["key"] != "camp-handover"],
    )

    seed_circle(session, fake_embedder, smaller, now=NOW)
    session.flush()

    stored = set(session.scalars(select(SourceItem.external_id).where(SourceItem.org_id == org_id)))
    assert gone_conversation not in stored
    assert gone_document not in stored
    lines = sum(len(c["lines"]) for c in smaller.conversations)
    assert _count(session, ChannelMessage, org_id) == lines


def test_other_organizations_are_left_alone(session, fake_embedder) -> None:
    other = Organization(name="ほかの組織", discord_guild_id="123")
    session.add(other)
    session.flush()

    seed_circle(session, fake_embedder, load_circle(), now=NOW)
    session.flush()

    assert _count(session, SourceItem, other.id) == 0
    assert _count(session, Employee, other.id) == 0

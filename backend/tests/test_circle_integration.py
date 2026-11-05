"""Seeding the fictional circle into the database (#79)."""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest
from sqlalchemy import func, select, text

from tekijin.agent.graph import build_agent
from tekijin.agent.protocols import IntentResult
from tekijin.data.circle import GUILD_ID, external_id, load_circle, seed_circle
from tekijin.data.circle import _discord_id as circle_discord_id
from tekijin.data.db import get_sessionmaker
from tekijin.data.sources import count_source_items
from tekijin.data.visibility import PUBLIC
from tekijin.models.tables import (
    ChannelMessage,
    Chunk,
    Employee,
    Identity,
    Organization,
    SelfIntro,
    SourceItem,
)
from tekijin.retrieval.experts import FLOORS, DbExpertFinder
from tekijin.retrieval.sources import ChunkSearch

NOW = dt.datetime(2026, 11, 1, 12, 0, 0)
NO_FLOOR = dict.fromkeys(FLOORS, 0.0)


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
        cleanup.execute(text("DELETE FROM discord_channels"))
        cleanup.execute(text("DELETE FROM discord_threads"))
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


def test_everyone_can_see_the_whole_circle(session, fake_embedder) -> None:
    circle = load_circle()
    org_id = seed_circle(session, fake_embedder, circle, now=NOW).org_id
    session.flush()

    assert count_source_items(session, org_id=org_id, scope=PUBLIC) == _count(
        session, SourceItem, org_id
    )


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


def test_the_intro_slides_give_twelve_self_intros(session, fake_embedder) -> None:
    """#106: one ``slides`` row per slide, each the member the slide names."""

    circle = load_circle()
    org_id = seed_circle(session, fake_embedder, circle, now=NOW).org_id
    session.flush()

    rows = session.scalars(select(SelfIntro).where(SelfIntro.org_id == org_id)).all()
    assert len(rows) == 12
    assert {row.origin for row in rows} == {"slides"}
    names = {session.get(Employee, row.member_id).name for row in rows}
    assert "岩田 颯太" in names and "野口 大和" not in names

    seed_circle(session, fake_embedder, circle, now=NOW + dt.timedelta(days=1))
    session.flush()
    assert _count(session, SelfIntro, org_id) == 12


# --------------------------------------------------------------------------- #
# #107: the circle's person questions reach the right member
# --------------------------------------------------------------------------- #
def _member_ids(session, org_id: int) -> dict[str, int]:
    """Fixture member key ("m04") -> stored member id."""

    by_discord = {
        row.external_id: row.employee_id
        for row in session.scalars(select(Identity).where(Identity.org_id == org_id))
    }
    return {m["key"]: by_discord[circle_discord_id(m["key"])] for m in load_circle().members}


def _experts(session, embedder, org_id: int, question: str, *, from_messages: bool):
    vector = embedder.encode([question])[0]
    found = ChunkSearch(session).search(question, org_id, vector, PUBLIC)
    finder = DbExpertFinder(session, floors=NO_FLOOR, expertise_from_messages=from_messages)
    return finder.find(
        org_id=org_id,
        question=question,
        query_vector=vector,
        scope=PUBLIC,
        source_hits=found.hits,
        asker_id=None,
    )


class _Bigrams:
    """Character-bigram embedder. The shared fake embedder splits on whitespace,
    which unspaced Japanese has none of, so its vectors say almost nothing here."""

    def __init__(self, words) -> None:
        self._words = words

    def encode(self, texts, *, kind: str = "passage") -> list[list[float]]:
        return self._words.encode([_bigrams(t) for t in texts], kind=kind)


def _bigrams(passage: str) -> str:
    squashed = "".join(passage.split())
    return " ".join(squashed[i : i + 2] for i in range(len(squashed) - 1)) or squashed


@pytest.fixture
def bigrams(fake_embedder) -> _Bigrams:
    return _Bigrams(fake_embedder)


@pytest.mark.parametrize("from_messages", [False, True])
def test_person_questions_find_the_right_member(session, bigrams, from_messages) -> None:
    """Recall@3 on the circle's 8 person questions, with no floors (the floors are
    calibrated to the real embedder on evo-x2, #107)."""

    circle = load_circle()
    org_id = seed_circle(session, bigrams, circle, now=NOW).org_id
    session.flush()
    key_of = {v: k for k, v in _member_ids(session, org_id).items()}

    hits = 0
    misses: list[tuple[str, list[str]]] = []
    for q in (q for q in circle.questions if q["kind"] == "person"):
        result = _experts(session, bigrams, org_id, q["question"], from_messages=from_messages)
        top = [key_of[e.member_id] for e in result.experts[:3]]
        hits += bool(set(top) & set(q["members"]))
        misses.extend([] if set(top) & set(q["members"]) else [(q["id"], top)])
        for expert in result.experts:
            assert 1 <= len(expert.reasons) <= 3
            if not from_messages:
                assert all(r["type"] != "conversation" for r in expert.reasons)
    # 5/8 with character bigrams and no floors; Nemotron with the floors gets 8/8.
    assert hits >= 5, misses


class _Confident:
    def analyze(self, question, asker, *, context=None) -> IntentResult:
        return IntentResult(confidence=0.9)


def test_a_circle_person_question_reaches_the_hand_off(session, bigrams) -> None:
    circle = load_circle()
    org_id = seed_circle(session, bigrams, circle, now=NOW).org_id
    session.flush()
    ids = _member_ids(session, org_id)
    question = next(q for q in circle.questions if q["id"] == "q29")

    agent = build_agent(
        bigrams,
        session,
        intent_model=_Confident(),
        source_search=ChunkSearch(session),
        expert_finder=DbExpertFinder(session, floors=NO_FLOOR),
    )
    config = {"configurable": {"thread_id": "circle-q29"}}
    state = agent.invoke(
        {
            "question": question["question"],
            "asker": {"id": ids["m01"]},
            "org_id": org_id,
            "now": NOW,
        },
        config,
    )

    assert agent.get_state(config).next == ("send",)
    recs = {r["person_id"]: r for r in state["recommendations"]}
    assert ids["m04"] in recs
    assert ids["m01"] not in recs
    assert {r["type"] for r in recs[ids["m04"]]["reasons"]} == {"self_intro", "document"}

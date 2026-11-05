"""The graph answers a member's question from verified knowledge (#112, R6).

The knowledge step runs after C3 and before C4. It answers only from verified
knowledge of the asker's organization that the asker may read, never takes a
question that asks for a person, and otherwise lets the run go on as before.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import pytest
from sqlalchemy import text

from tekijin.agent.graph import build_agent
from tekijin.agent.protocols import IntentResult
from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import SourceRef, create_draft, verify_article
from tekijin.data.visibility import SourceScope, scope_to_state
from tekijin.knowledge.index import embed_articles
from tekijin.models.tables import ChannelMessage, DiscordChannel, Employee, Organization
from tekijin.retrieval.experts import SELF_INTRO, Evidence, Expert, ExpertResult, rank_experts
from tekijin.retrieval.fragments import CitedEvidence
from tekijin.retrieval.sources import SourceSearchResult

NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
QUESTION = "部室 の 鍵 は どこ に あり ます か"
PRIVATE = 8802


class _Intent:
    def __init__(self, seeks: bool = False) -> None:
        self.seeks = seeks

    def analyze(self, question, asker, *, context=None) -> IntentResult:
        return IntentResult(confidence=0.9, seeks_person=self.seeks)


class _Sources:
    """Nothing in the documents: a question the knowledge does not answer goes to a person."""

    def search(self, query, org_id, query_vector, scope) -> SourceSearchResult:
        return SourceSearchResult(hits=[], confidence=0.0)

    def evidence(self, hits, org_id, scope) -> list[CitedEvidence]:
        return []


class _Finder:
    def __init__(self, member: int) -> None:
        self.member = member

    def find(self, **kwargs: Any) -> ExpertResult:
        evidence = [
            Evidence(
                kind=SELF_INTRO,
                member_id=self.member,
                rank=1,
                cosine=0.3,
                reason={"type": SELF_INTRO, "detail": "自己紹介"},
            )
        ]
        return rank_experts(evidence, {self.member: ("アリス", None)}, floors={SELF_INTRO: 0.0})

    def describe(self, org_id: int, member_ids: Sequence[int]) -> list[Expert]:
        return [
            Expert(
                member_id=m, name="アリス", dept=None, raw_score=0.0, reasons=(), kinds=frozenset()
            )
            for m in dict.fromkeys(member_ids)
            if m == self.member
        ]


class _MustNotRun:
    def search(self, *args, **kwargs):
        raise AssertionError("a member's question reached the company retriever")

    def rank(self, *args, **kwargs):
        raise AssertionError("a member's question reached the company scorer")


@pytest.fixture
def org(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        org = Organization(name="天文サークル", discord_guild_id="771")
        other = Organization(name="ほか", discord_guild_id="772")
        session.add_all([org, other])
        session.flush()
        alice = Employee(name="アリス", org_id=org.id)
        session.add(alice)
        session.add(DiscordChannel(org_id=org.id, channel_id=PRIVATE, public=False, updated_at=NOW))
        session.flush()
        secret = ChannelMessage(
            org_id=org.id,
            message_id=1,
            channel_id=PRIVATE,
            channel_name="運営",
            author_id=alice.id,
            content="鍵",
            created_at=NOW,
        )
        session.add(secret)
        session.commit()
        ids = {"org": org.id, "other": other.id, "alice": alice.id, "secret": secret.id}
    yield ids
    with factory() as session:
        for table in (
            "knowledge_sources",
            "knowledge_articles",
            "channel_messages",
            "discord_channels",
        ):
            session.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed names
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


@pytest.fixture
def db(engine, org):
    factory = get_sessionmaker(engine)
    with factory() as session:
        yield session
        session.rollback()


def _article(db, fake_embedder, org_id, *, verify=True, sources=()) -> int:
    article = create_draft(
        db,
        org_id=org_id,
        kind="procedure",
        issue=QUESTION,
        resolution="部室 の 箱 の 中",
        sources=sources,
        now=NOW,
    )
    if verify:
        verify_article(db, article.id, org_id=org_id, verifier_id=None, now=NOW)
    embed_articles(db, fake_embedder)
    return article.id


def _agent(db, fake_embedder, org, *, seeks=False, floor: float | None = 0.3):
    return build_agent(
        fake_embedder,
        db,
        intent_model=_Intent(seeks),
        retriever=_MustNotRun(),
        scorer=_MustNotRun(),
        source_search=_Sources(),
        expert_finder=_Finder(org["alice"]),
        knowledge_answer_min_similarity=floor,
    )


def _ask(agent, thread, org_id, scope: SourceScope | None = None):
    config = {"configurable": {"thread_id": thread}}
    state = agent.invoke(
        {
            "question": QUESTION,
            "asker": {"id": 999},
            "org_id": org_id,
            "now": NOW,
            "source_scope": scope_to_state(scope) if scope is not None else None,
        },
        config,
    )
    return state, config


def test_a_members_question_is_answered_from_verified_knowledge(db, fake_embedder, org) -> None:
    article_id = _article(db, fake_embedder, org["org"])
    agent = _agent(db, fake_embedder, org)
    state, config = _ask(agent, "kb-hit", org["org"])

    assert not agent.get_state(config).next  # finished, no hand-off
    assert state["self_answer_grounded"] is True
    assert "答え: 部室 の 箱 の 中" in state["answer"]
    assert state["self_answer_citations"][0]["source_id"] == f"ka_{article_id}"
    assert state["knowledge_article_ids"] == [article_id]


def test_a_draft_does_not_answer_but_is_counted(db, fake_embedder, org) -> None:
    draft_id = _article(db, fake_embedder, org["org"], verify=False)
    agent = _agent(db, fake_embedder, org)
    state, config = _ask(agent, "kb-draft", org["org"])

    assert state["self_answer_grounded"] is False
    assert state["knowledge_draft_ids"] == [draft_id]
    assert agent.get_state(config).next == ("send",)  # went on to a person


def test_another_organizations_knowledge_is_not_used(db, fake_embedder, org) -> None:
    _article(db, fake_embedder, org["other"])
    agent = _agent(db, fake_embedder, org)
    state, _ = _ask(agent, "kb-other", org["org"])
    assert state["self_answer_grounded"] is False


def test_a_question_for_a_person_is_not_taken_by_knowledge(db, fake_embedder, org) -> None:
    _article(db, fake_embedder, org["org"])
    agent = _agent(db, fake_embedder, org, seeks=True)
    state, config = _ask(agent, "kb-seek", org["org"])

    assert state["self_answer_grounded"] is False
    assert agent.get_state(config).next == ("send",)
    assert [r["person_id"] for r in state["recommendations"]] == [org["alice"]]


def test_knowledge_from_a_channel_the_asker_cannot_read_is_not_used(db, fake_embedder, org) -> None:
    _article(db, fake_embedder, org["org"], sources=[SourceRef(channel_message_id=org["secret"])])
    agent = _agent(db, fake_embedder, org)
    state, _ = _ask(agent, "kb-private", org["org"])
    assert state["self_answer_grounded"] is False

    reader = SourceScope(channels=frozenset({PRIVATE}))
    state, _ = _ask(agent, "kb-private-reader", org["org"], scope=reader)
    assert state["self_answer_grounded"] is True


def test_a_question_without_an_organization_skips_knowledge(db, fake_embedder, org) -> None:
    _article(db, fake_embedder, org["org"])
    agent = _agent(db, fake_embedder, org)

    class _Retriever:
        def search(self, query, *, query_vector=None):
            from tekijin.agent.state import empty_retrieval

            return empty_retrieval()

    agent = build_agent(
        fake_embedder,
        db,
        intent_model=_Intent(),
        retriever=_Retriever(),
        scorer=type("S", (), {"rank": lambda self, *a, **k: {"recommendations": []}})(),
        knowledge_answer_min_similarity=0.3,
    )
    state, _ = _ask(agent, "kb-company", None)
    assert state["self_answer_grounded"] is False


def test_the_step_is_absent_when_the_flag_is_off(db, fake_embedder, org) -> None:
    agent = _agent(db, fake_embedder, org, floor=None)
    assert "knowledge_answer" not in agent.get_graph().nodes

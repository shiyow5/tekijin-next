"""Answering from what the organization ingested (#81, R3)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pytest
from sqlalchemy import text

from tekijin.agent.graph import build_agent
from tekijin.agent.protocols import IntentResult, SelfAnswerResult
from tekijin.agent.state import empty_retrieval
from tekijin.data.db import get_sessionmaker
from tekijin.data.sources import upsert_source_item
from tekijin.data.visibility import EVERYTHING
from tekijin.models.tables import Employee, Organization
from tekijin.retrieval.fragments import CitedEvidence
from tekijin.retrieval.sources import ChunkSearch, source_citation_id

NOW = dt.datetime(2026, 11, 2, 12, 0, 0)
KEYS = "部室 鍵 守衛室 借りる"


@pytest.fixture
def orgs(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        org = Organization(name="テストサークル", discord_guild_id="777")
        other = Organization(name="ほかの組織", discord_guild_id="888")
        session.add_all([org, other])
        session.flush()
        member = Employee(name="会計係", org_id=org.id)
        session.add(member)
        session.commit()
        ids = (org.id, other.id, member.id)
    yield ids
    with factory() as session:
        session.execute(text("DELETE FROM chunks"))
        session.execute(text("DELETE FROM source_items"))
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


def _add(session, embedder, org_id, key, body, *, title="部室", url=None):
    return upsert_source_item(
        session,
        embedder,
        org_id=org_id,
        provider="manual",
        external_id=f"note:{key}",
        title=title,
        body=body,
        url=url,
        author_id=None,
        now=NOW,
    )


def _search(session, embedder, org_id, query):
    vector = embedder.encode([query])[0]
    return ChunkSearch(session).search(query, org_id, vector, EVERYTHING)


def test_search_finds_the_matching_chunk_with_its_confidence(engine, fake_embedder, orgs):
    org_id, _, _ = orgs
    with get_sessionmaker(engine)() as session:
        added = _add(session, fake_embedder, org_id, "keys", KEYS)
        _add(session, fake_embedder, org_id, "camp", "合宿 宿代 8000円", title="合宿")
        session.commit()
        result = _search(session, fake_embedder, org_id, KEYS)

    assert result.hits[0]["source_item_id"] == added.item_id
    assert result.confidence > 0.28


def test_search_never_crosses_organizations(engine, fake_embedder, orgs):
    org_id, other_id, _ = orgs
    with get_sessionmaker(engine)() as session:
        _add(session, fake_embedder, other_id, "keys", KEYS)
        session.commit()
        result = _search(session, fake_embedder, org_id, KEYS)
        evidence = ChunkSearch(session).evidence(
            _search(session, fake_embedder, other_id, KEYS).hits, org_id, EVERYTHING
        )

    assert result.hits == []
    assert result.confidence == 0.0
    assert evidence == []


def test_keyword_match_is_found_even_when_the_vector_is_far(engine, fake_embedder, orgs):
    org_id, _, _ = orgs
    with get_sessionmaker(engine)() as session:
        target = _add(session, fake_embedder, org_id, "scope", "望遠鏡 EM200 赤道儀 手順")
        session.commit()
        unrelated = fake_embedder.encode(["まったく 別の 話"])[0]
        result = ChunkSearch(session).search("EM200", org_id, unrelated, EVERYTHING)

    assert target.item_id in {h["source_item_id"] for h in result.hits}


def test_evidence_groups_chunks_by_source_and_carries_title_and_url(engine, fake_embedder, orgs):
    org_id, _, _ = orgs
    with get_sessionmaker(engine)() as session:
        long_body = "\n\n".join(f"鍵 手順 {i} " + "守衛室 借りる " * 60 for i in range(4))
        added = _add(
            session, fake_embedder, org_id, "keys", long_body, url="https://example.com/keys"
        )
        session.commit()
        search = ChunkSearch(session)
        result = _search(session, fake_embedder, org_id, "鍵 手順")
        evidence = search.evidence(result.hits, org_id, EVERYTHING)

    assert len(result.hits) > 1
    assert len(evidence) == 1
    item = evidence[0]
    assert item.source_id == source_citation_id(added.item_id)
    assert item.kind == "source"
    assert item.title == "部室"
    assert item.url == "https://example.com/keys"
    assert item.text.startswith("部室(2026-11-02)")


class _Intent:
    def analyze(self, question, asker, *, context=None) -> IntentResult:
        return IntentResult(topics=["鍵"], confidence=0.9)


class _Composer:
    def __init__(self, *, grounded: bool = True) -> None:
        self.grounded = grounded
        self.seen: list[CitedEvidence] = []

    def compose(self, question: str, evidence: Sequence[CitedEvidence]) -> SelfAnswerResult:
        self.seen = list(evidence)
        if not self.grounded:
            return SelfAnswerResult()
        return SelfAnswerResult("守衛室で借ります。", [evidence[0].source_id], True)


class _LegacyMustNotRun:
    def search(self, query, *, query_vector=None):
        raise AssertionError("an organization's question reached the unscoped tables")


class _Scorer:
    def rank(self, *args, **kwargs):
        return {"recommendations": []}


def _run(session, embedder, composer, *, org_id):
    agent = build_agent(
        embedder,
        session,
        intent_model=_Intent(),
        self_answer_model=composer,
        retriever=_LegacyMustNotRun(),
        scorer=_Scorer(),
        source_search=ChunkSearch(session),
    )
    return agent.invoke(
        {"question": KEYS, "asker": {"id": 1}, "org_id": org_id, "now": NOW},
        {"configurable": {"thread_id": f"t{org_id}{composer.grounded}"}},
    )


def test_member_is_answered_from_sources_with_a_titled_citation(engine, fake_embedder, orgs):
    org_id, _, _ = orgs
    with get_sessionmaker(engine)() as session:
        added = _add(session, fake_embedder, org_id, "keys", KEYS, url="https://example.com/k")
        session.commit()
        state = _run(session, fake_embedder, _Composer(), org_id=org_id)

    assert state["answer"] == "守衛室で借ります。"
    assert state["self_answer_citations"] == [
        {
            "source_id": source_citation_id(added.item_id),
            "kind": "source",
            "title": "部室",
            "url": "https://example.com/k",
        }
    ]


def test_sources_that_do_not_answer_end_with_an_honest_reply(engine, fake_embedder, orgs):
    org_id, _, _ = orgs
    with get_sessionmaker(engine)() as session:
        _add(session, fake_embedder, org_id, "keys", KEYS)
        session.commit()
        state = _run(session, fake_embedder, _Composer(grounded=False), org_id=org_id)

    assert state["answer"] == "資料の中に、この質問の答えは見つかりませんでした。"
    assert state.get("document_id") is None


def test_empty_retrieval_has_no_sources() -> None:
    empty = empty_retrieval()
    assert empty["sources"] == []
    assert empty["source_confidence"] == 0.0


class _SureWithoutTopics:
    def analyze(self, question, asker, *, context=None) -> IntentResult:
        return IntentResult(confidence=0.9, situation="鍵の場所")


def test_a_confident_question_without_topics_reaches_the_search(engine, fake_embedder, orgs):
    org_id, _, _ = orgs
    with get_sessionmaker(engine)() as session:
        _add(session, fake_embedder, org_id, "keys", KEYS)
        session.commit()
        agent = build_agent(
            fake_embedder,
            session,
            intent_model=_SureWithoutTopics(),
            self_answer_model=_Composer(),
            scorer=_Scorer(),
            source_search=ChunkSearch(session),
        )
        state = agent.invoke(
            {"question": KEYS, "asker": None, "org_id": org_id, "now": NOW},
            {"configurable": {"thread_id": "no-topics"}},
        )

    assert state["answer"] == "守衛室で借ります。"


def test_nothing_near_the_question_says_so(engine, fake_embedder, orgs):
    org_id, _, _ = orgs
    with get_sessionmaker(engine)() as session:
        agent = build_agent(
            fake_embedder,
            session,
            intent_model=_SureWithoutTopics(),
            self_answer_model=_Composer(),
            scorer=_Scorer(),
            source_search=ChunkSearch(session),
        )
        state = agent.invoke(
            {"question": KEYS, "asker": None, "org_id": org_id, "now": NOW},
            {"configurable": {"thread_id": "nothing"}},
        )

    assert state["answer"] == "資料の中に、この質問の答えは見つかりませんでした。"


def test_older_switches_cannot_reach_other_organizations(
    engine, fake_embedder, orgs, monkeypatch
) -> None:
    def must_not_run(*args, **kwargs):
        raise AssertionError("an organization's question reached the undivided tables")

    monkeypatch.setattr("tekijin.agent.nodes.answer_from_knowledge", must_not_run)
    scorer = _Scorer()
    scorer.rank = must_not_run
    org_id, _, _ = orgs
    with get_sessionmaker(engine)() as session:
        agent = build_agent(
            fake_embedder,
            session,
            intent_model=_Intent(),
            self_answer_model=_Composer(),
            scorer=scorer,
            source_search=ChunkSearch(session),
            knowledge_answer_min_similarity=0.0,
            score_all_employees=True,
        )
        state = agent.invoke(
            {"question": KEYS, "asker": None, "org_id": org_id, "now": NOW},
            {"configurable": {"thread_id": "switches"}},
        )

    assert state["answer"] == "資料の中に、この質問の答えは見つかりませんでした。"

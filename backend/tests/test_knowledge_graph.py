"""Verified knowledge is evidence for the self-answer (#118, R6).

C4 offers the verified knowledge of the asker's organization that the asker may
read next to the documents, and the composer picks what it uses. Only the articles
the answer cites are counted, and an answer citing one also cites what the article
was made from. A question that asks for a person still goes to a person.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from sqlalchemy import text

from tekijin.agent.graph import build_agent
from tekijin.agent.protocols import IntentResult, SelfAnswerResult
from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import SourceRef, create_draft, verify_article
from tekijin.data.visibility import SourceScope, scope_to_state
from tekijin.knowledge.index import embed_articles
from tekijin.models.tables import (
    ChannelMessage,
    DiscordChannel,
    Employee,
    Organization,
    SourceItem,
)
from tekijin.retrieval.experts import SELF_INTRO, Evidence, Expert, ExpertResult, rank_experts
from tekijin.retrieval.fragments import CitedEvidence
from tekijin.retrieval.sources import SourceSearchResult

NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
QUESTION = "部室 の 鍵 は どこ に あり ます か"
PRIVATE = 8802
# The id the fake document search cites (not a stored row).
GUIDE = 424242


class _Intent:
    def __init__(self, seeks: bool = False) -> None:
        self.seeks = seeks

    def analyze(self, question, asker, *, context=None) -> IntentResult:
        return IntentResult(confidence=0.9, seeks_person=self.seeks)


class _Sources:
    """What the documents hold: nothing by default, or one passage of a guide."""

    def __init__(self, *, found: bool = False) -> None:
        self.found = found

    def search(self, query, org_id, query_vector, scope) -> SourceSearchResult:
        if not self.found:
            return SourceSearchResult(hits=[], confidence=0.0)
        return SourceSearchResult(
            hits=[{"chunk_id": 1, "source_item_id": GUIDE, "score": 0.5}], confidence=0.1
        )

    def evidence(self, hits, org_id, scope) -> list[CitedEvidence]:
        if not hits:
            return []
        return [CitedEvidence(f"src_{GUIDE}", "source", "合宿の手引き 鍵は顧問が持つ", "手引き")]


class _Composer:
    """Records what it was offered and cites the ids it is told to (when offered)."""

    def __init__(self, cite: Callable[[list[CitedEvidence]], list[str]] | None = None) -> None:
        self.cite = cite or (lambda evidence: [e.source_id for e in evidence])
        self.seen: list[CitedEvidence] = []

    def compose(self, question: str, evidence: Sequence[CitedEvidence]) -> SelfAnswerResult:
        self.seen = list(evidence)
        cited = self.cite(self.seen)
        if not cited:
            return SelfAnswerResult()
        return SelfAnswerResult("部室の箱の中です。", cited, True)


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
        item = SourceItem(
            org_id=org.id,
            provider="manual",
            external_id="manual:keys",
            title="鍵の決まり",
            url="https://ex.com/keys",
            body="鍵は部室の箱",
            content_hash="h",
            updated_at=NOW,
        )
        session.add_all([secret, item])
        session.commit()
        ids = {
            "org": org.id,
            "other": other.id,
            "alice": alice.id,
            "secret": secret.id,
            "item": item.id,
        }
    yield ids
    with factory() as session:
        for table in (
            "knowledge_sources",
            "knowledge_articles",
            "channel_messages",
            "discord_channels",
            "source_items",
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


def _article(
    db, fake_embedder, org_id, *, verify=True, sources=(), issue=QUESTION, resolution=None
) -> int:
    article = create_draft(
        db,
        org_id=org_id,
        kind="procedure",
        issue=issue,
        resolution=resolution or "部室 の 箱 の 中",
        sources=sources,
        now=NOW,
    )
    if verify:
        verify_article(db, article.id, org_id=org_id, verifier_id=None, now=NOW)
    embed_articles(db, fake_embedder)
    return article.id


def _agent(
    db,
    fake_embedder,
    org,
    *,
    seeks=False,
    floor: float | None = 0.3,
    composer: _Composer | None = None,
    sources: _Sources | None = None,
    additive: bool = False,
):
    return build_agent(
        fake_embedder,
        db,
        intent_model=_Intent(seeks),
        retriever=_MustNotRun(),
        scorer=_MustNotRun(),
        self_answer_model=composer if composer is not None else _Composer(),
        source_search=sources or _Sources(),
        expert_finder=_Finder(org["alice"]),
        knowledge_answer_min_similarity=floor,
        additive_self_answer_enabled=additive,
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


def _ids(evidence: Sequence[CitedEvidence]) -> list[str]:
    return [e.source_id for e in evidence]


# -- (a) knowledge enters the candidates -----------------------------------
def test_verified_knowledge_is_offered_next_to_the_documents(db, fake_embedder, org) -> None:
    article_id = _article(db, fake_embedder, org["org"])
    composer = _Composer()
    agent = _agent(db, fake_embedder, org, composer=composer, sources=_Sources(found=True))
    state, config = _ask(agent, "kb-offer", org["org"])

    assert _ids(composer.seen) == [f"src_{GUIDE}", f"ka_{article_id}"]
    offered = composer.seen[1]
    assert offered.kind == "knowledge" and offered.title == QUESTION
    assert "答え: 部室 の 箱 の 中" in offered.text
    assert state["retrieval"]["knowledge"][0]["article_id"] == article_id
    assert state["route"] == "document"
    assert state["self_answer_grounded"] is True
    assert not agent.get_state(config).next  # answered, no hand-off


def test_knowledge_alone_can_carry_the_question_to_an_answer(db, fake_embedder, org) -> None:
    # Nothing in the documents: the knowledge's similarity is what sends the
    # question to the composer instead of a person.
    article_id = _article(db, fake_embedder, org["org"])
    composer = _Composer()
    state, _ = _ask(_agent(db, fake_embedder, org, composer=composer), "kb-alone", org["org"])

    assert state["retrieval"]["source_confidence"] == 0.0
    assert state["retrieval"]["document_confidence"] == pytest.approx(
        state["retrieval"]["knowledge_confidence"]
    )
    assert _ids(composer.seen) == [f"ka_{article_id}"]
    assert state["self_answer_grounded"] is True


def test_the_composer_may_leave_the_knowledge_unused(db, fake_embedder, org) -> None:
    # Similar but not the answer: the composer declines, and the run goes on as it
    # would without knowledge (the document terminal offers a person).
    _article(db, fake_embedder, org["org"])
    composer = _Composer(cite=lambda evidence: [])
    agent = _agent(db, fake_embedder, org, composer=composer)
    state, config = _ask(agent, "kb-declined", org["org"])

    assert state["self_answer_grounded"] is False
    assert state["knowledge_article_ids"] == []
    assert state["answer"].startswith("資料の中に、この質問の答えは見つかりませんでした。")
    assert not agent.get_state(config).next


# -- (b) only the cited articles are counted -------------------------------
def test_only_the_articles_the_answer_cites_are_counted(db, fake_embedder, org) -> None:
    used = _article(db, fake_embedder, org["org"])
    unused = _article(
        db, fake_embedder, org["org"], issue="部室 の 鍵 の 予備 は どこ", resolution="顧問 の 机"
    )
    composer = _Composer(cite=lambda evidence: [f"ka_{used}"])
    agent = _agent(db, fake_embedder, org, composer=composer, floor=0.0)
    state, _ = _ask(agent, "kb-one", org["org"])

    assert set(_ids(composer.seen)) == {f"ka_{used}", f"ka_{unused}"}
    assert state["knowledge_article_ids"] == [used]


def test_an_answer_from_the_documents_counts_no_knowledge(db, fake_embedder, org) -> None:
    _article(db, fake_embedder, org["org"])
    composer = _Composer(cite=lambda evidence: [f"src_{GUIDE}"])
    agent = _agent(db, fake_embedder, org, composer=composer, sources=_Sources(found=True))
    state, _ = _ask(agent, "kb-docs", org["org"])

    assert state["self_answer_grounded"] is True
    assert state["knowledge_article_ids"] == []
    assert [c["source_id"] for c in state["self_answer_citations"]] == [f"src_{GUIDE}"]


# -- (c) citations name the article and its sources ------------------------
def test_a_cited_article_is_followed_by_its_sources(db, fake_embedder, org) -> None:
    article_id = _article(
        db, fake_embedder, org["org"], sources=[SourceRef(source_item_id=org["item"])]
    )
    state, _ = _ask(_agent(db, fake_embedder, org), "kb-cite", org["org"])

    assert state["self_answer_citations"] == [
        {"source_id": f"ka_{article_id}", "kind": "knowledge", "title": QUESTION},
        {
            "source_id": f"src_{org['item']}",
            "kind": "source",
            "title": "鍵の決まり",
            "url": "https://ex.com/keys",
        },
    ]
    assert state["answer"] == "部室の箱の中です。"


# -- (d) a question for a person still goes to a person --------------------
def test_a_question_for_a_person_is_not_taken_by_knowledge(db, fake_embedder, org) -> None:
    _article(db, fake_embedder, org["org"])
    agent = _agent(db, fake_embedder, org, seeks=True)
    state, config = _ask(agent, "kb-seek", org["org"])

    assert state["route"] == "person"
    assert state["self_answer_grounded"] is False
    assert agent.get_state(config).next == ("send",)
    assert [r["person_id"] for r in state["recommendations"]] == [org["alice"]]


def test_knowledge_shown_next_to_a_hand_off_is_counted(db, fake_embedder, org) -> None:
    # #413: on the person route the additive answer may cite the knowledge; the
    # hand-off still follows, and the cited article is counted.
    article_id = _article(db, fake_embedder, org["org"])
    agent = _agent(db, fake_embedder, org, seeks=True, additive=True)
    state, config = _ask(agent, "kb-additive", org["org"])

    assert agent.get_state(config).next == ("send",)
    assert state["additive_citations"][0]["source_id"] == f"ka_{article_id}"
    assert state["knowledge_article_ids"] == [article_id]


# -- (e) organization and verified-only ------------------------------------
def test_a_draft_is_not_offered_but_is_counted(db, fake_embedder, org) -> None:
    draft_id = _article(db, fake_embedder, org["org"], verify=False)
    composer = _Composer()
    agent = _agent(db, fake_embedder, org, composer=composer)
    state, config = _ask(agent, "kb-draft", org["org"])

    assert composer.seen == []
    assert state["self_answer_grounded"] is False
    assert state["knowledge_draft_ids"] == [draft_id]
    assert agent.get_state(config).next == ("send",)  # went on to a person


def test_another_organizations_knowledge_is_not_offered(db, fake_embedder, org) -> None:
    _article(db, fake_embedder, org["other"])
    composer = _Composer()
    state, _ = _ask(_agent(db, fake_embedder, org, composer=composer), "kb-other", org["org"])
    assert composer.seen == []
    assert state["self_answer_grounded"] is False


def test_knowledge_from_a_channel_the_asker_cannot_read_is_not_offered(
    db, fake_embedder, org
) -> None:
    _article(db, fake_embedder, org["org"], sources=[SourceRef(channel_message_id=org["secret"])])
    composer = _Composer()
    agent = _agent(db, fake_embedder, org, composer=composer)
    state, _ = _ask(agent, "kb-private", org["org"])
    assert composer.seen == []
    assert state["self_answer_grounded"] is False

    reader = SourceScope(channels=frozenset({PRIVATE}))
    state, _ = _ask(agent, "kb-private-reader", org["org"], scope=reader)
    assert state["self_answer_grounded"] is True
    assert [c["kind"] for c in state["self_answer_citations"]] == ["knowledge", "message"]


def test_a_question_without_an_organization_skips_knowledge(db, fake_embedder, org) -> None:
    _article(db, fake_embedder, org["org"])

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
        self_answer_model=_Composer(),
        knowledge_answer_min_similarity=0.3,
    )
    state, _ = _ask(agent, "kb-company", None)
    assert "knowledge" not in state["retrieval"]
    assert state["self_answer_grounded"] is False


# -- the switch -------------------------------------------------------------
def test_nothing_is_offered_when_the_flag_is_off(db, fake_embedder, org) -> None:
    _article(db, fake_embedder, org["org"])
    composer = _Composer()
    agent = _agent(db, fake_embedder, org, floor=None, composer=composer)
    state, _ = _ask(agent, "kb-off", org["org"])

    assert "knowledge" not in state["retrieval"]
    assert composer.seen == []


def test_the_short_circuit_is_gone(db, fake_embedder, org) -> None:
    for floor in (None, 0.3):
        agent = _agent(db, fake_embedder, org, floor=floor)
        assert "knowledge_answer" not in agent.get_graph().nodes


def test_knowledge_needs_the_composer(db, fake_embedder, org) -> None:
    # Without a composer nothing could read the knowledge, so it is not searched
    # and the routes stay what they were.
    _article(db, fake_embedder, org["org"])
    agent = build_agent(
        fake_embedder,
        db,
        intent_model=_Intent(),
        retriever=_MustNotRun(),
        scorer=_MustNotRun(),
        source_search=_Sources(),
        expert_finder=_Finder(org["alice"]),
        knowledge_answer_min_similarity=0.3,
    )
    state, config = _ask(agent, "kb-no-composer", org["org"])
    assert "knowledge" not in state["retrieval"]
    assert agent.get_state(config).next == ("send",)

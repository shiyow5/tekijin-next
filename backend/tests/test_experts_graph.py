"""A member's question reaches a person through the graph (#107), with fakes only."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from typing import Any

import pytest
from langgraph.types import Command

from tekijin.agent.graph import build_agent
from tekijin.agent.nodes import AgentNodes
from tekijin.agent.protocols import IntentResult
from tekijin.agent.state import AgentState, empty_retrieval
from tekijin.api.service import responder_view
from tekijin.retrieval.experts import (
    DOCUMENT,
    SELF_INTRO,
    Evidence,
    Expert,
    ExpertResult,
    rank_experts,
)
from tekijin.retrieval.sources import SourceSearchResult

NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
ORG = 3
ASKER = 1
NAMES = {2: "岩田 颯太", 3: "小野 咲", 4: "高梨 湊", 5: "望月 凛", ASKER: "質問 する人"}
NO_FLOOR = {SELF_INTRO: 0.0, DOCUMENT: 0.0}


class _Embedder:
    def encode(self, texts: Sequence[str], *, kind: str = "passage") -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _Intent:
    def analyze(self, question, asker, *, context=None) -> IntentResult:
        return IntentResult(confidence=0.9)


class _Sources:
    def __init__(self, confidence: float = 0.1) -> None:
        # Below DOCUMENT_SIM the question takes the person route.
        self.confidence = confidence

    def search(self, query, org_id, query_vector, scope) -> SourceSearchResult:
        return SourceSearchResult(
            hits=[{"chunk_id": 10, "source_item_id": 20, "score": 0.03}],
            confidence=self.confidence,
        )

    def evidence(self, hits, org_id, scope):
        return []


class _Finder:
    """Ranks members 2, 3, 4, 5 in that order, as the database finder would."""

    def __init__(self, order: Sequence[int] = (2, 3, 4, 5)) -> None:
        self.order = list(order)
        self.calls: list[dict[str, Any]] = []
        self.gone: set[int] = set()

    def _result(self, members: Iterable[int]) -> ExpertResult:
        allowed = set(members)
        evidence = [
            Evidence(
                kind=SELF_INTRO,
                member_id=m,
                rank=rank,
                cosine=0.3,
                reason={"type": SELF_INTRO, "detail": f"自己紹介: {NAMES[m]}"},
            )
            for rank, m in enumerate(self.order, start=1)
        ]
        names = {m: (NAMES[m], None) for m in allowed}
        return rank_experts(evidence, names, floors=NO_FLOOR)

    def find(self, **kwargs: Any) -> ExpertResult:
        self.calls.append(kwargs)
        excluded = {*kwargs.get("exclude", ()), kwargs.get("asker_id")}
        return self._result(m for m in NAMES if m not in excluded)

    def describe(self, org_id: int, member_ids: Iterable[int]) -> list[Expert]:
        # ``gone``: opted out or left after C4 ran.
        return [
            Expert(
                member_id=m, name=NAMES[m], dept=None, raw_score=0.0, reasons=(), kinds=frozenset()
            )
            for m in dict.fromkeys(member_ids)
            if m in NAMES and m not in self.gone
        ]


class _ScorerMustNotRun:
    def rank(self, *args, **kwargs):
        raise AssertionError("a member's question reached the company scorer")


class _LegacyMustNotRun:
    def search(self, query, *, query_vector=None):
        raise AssertionError("a member's question reached the undivided tables")


def _agent(finder, sources=None):
    return build_agent(
        _Embedder(),
        None,
        intent_model=_Intent(),
        retriever=_LegacyMustNotRun(),
        scorer=_ScorerMustNotRun(),
        source_search=sources or _Sources(),
        expert_finder=finder,
    )


def _ask(agent, thread: str):
    config = {"configurable": {"thread_id": thread}}
    state = agent.invoke(
        {"question": "赤道儀を見てくれる人", "asker": {"id": ASKER}, "org_id": ORG, "now": NOW},
        config,
    )
    return state, config


def test_a_members_question_reaches_the_draft_with_the_best_person() -> None:
    finder = _Finder()
    agent = _agent(finder)
    state, config = _ask(agent, "org-person")

    assert agent.get_state(config).next == ("send",)
    assert state["route"] == "person"
    assert [r["person_id"] for r in state["recommendations"]] == [2, 3, 4]
    assert state["recommendations"][0]["reasons"][0]["type"] == SELF_INTRO
    assert state["draft"]
    retrieval = state["retrieval"]
    assert retrieval["candidate_people"] == [2, 3, 4, 5]
    assert retrieval["people_confidence"] == pytest.approx(0.3)
    assert finder.calls[0]["asker_id"] == ASKER
    assert finder.calls[0]["source_hits"] == [{"chunk_id": 10, "source_item_id": 20, "score": 0.03}]


def test_a_decline_moves_to_the_next_person_without_searching_again() -> None:
    finder = _Finder()
    agent = _agent(finder)
    _, config = _ask(agent, "org-decline")

    state = agent.invoke(Command(resume="declined"), config)

    assert [r["person_id"] for r in state["recommendations"]] == [3, 4, 5]
    assert state["declined_ids"] == [2]
    assert len(finder.calls) == 1


def test_without_a_finder_a_members_question_still_finds_nobody() -> None:
    agent = _agent(None)
    state, config = _ask(agent, "org-none")

    assert state["recommendations"] == []
    assert state["answer"] == "資料の中に、この質問の答えは見つかりませんでした。"
    assert not agent.get_state(config).next


# --------------------------------------------------------------------------- #
# Non-members keep the company path
# --------------------------------------------------------------------------- #
class _Retriever:
    def search(self, query, *, query_vector=None):
        return {**empty_retrieval(), "candidate_people": [7], "people_confidence": 0.5}


class _Scorer:
    def __init__(self) -> None:
        self.calls = 0

    def rank(self, topics, candidates, asker_id, now, *, top_k=3, **kwargs):
        self.calls += 1
        return {
            "recommendations": [
                {
                    "person_id": 7,
                    "name": "社員",
                    "dept": "営業",
                    "score": 0.5,
                    "confidence": "高",
                    "reasons": [],
                }
            ]
        }


class _FinderMustNotRun:
    def find(self, **kwargs):
        raise AssertionError("a question without an organization reached the finder")

    def describe(self, org_id, member_ids):
        raise AssertionError("a question without an organization reached the finder")


class _TopicIntent:
    def analyze(self, question, asker, *, context=None) -> IntentResult:
        return IntentResult(topics=["請求"], confidence=0.9)


def test_a_question_without_an_organization_is_scored_as_before() -> None:
    scorer = _Scorer()
    agent = build_agent(
        _Embedder(),
        None,
        intent_model=_TopicIntent(),
        retriever=_Retriever(),
        scorer=scorer,
        source_search=_Sources(),
        expert_finder=_FinderMustNotRun(),
    )
    state = agent.invoke(
        {"question": "請求書の再発行", "asker": {"id": 1}, "now": NOW},
        {"configurable": {"thread_id": "no-org"}},
    )

    assert scorer.calls == 1
    assert [r["person_id"] for r in state["recommendations"]] == [7]
    assert "experts" not in state["retrieval"]


# --------------------------------------------------------------------------- #
# C6 for a member's question, node by node
# --------------------------------------------------------------------------- #
def _nodes(finder) -> AgentNodes:
    return AgentNodes(
        intent_model=_Intent(),
        sufficiency_model=None,  # type: ignore[arg-type]
        draft_model=None,  # type: ignore[arg-type]
        embedder=_Embedder(),
        retriever=_LegacyMustNotRun(),  # type: ignore[arg-type]
        scorer=_ScorerMustNotRun(),  # type: ignore[arg-type]
        source_search=_Sources(),
        expert_finder=finder,
    )


def _state(finder, **extra: Any) -> AgentState:
    rows = [e.as_recommendation() for e in finder.find(asker_id=ASKER).experts]
    state: AgentState = {
        "org_id": ORG,
        "asker": {"id": ASKER},
        "now": NOW,
        "route": "person",
        "retrieval": {**empty_retrieval(), "experts": rows},
    }
    state.update(extra)  # type: ignore[typeddict-item]
    return state


def test_survivors_keep_their_place_and_the_freed_slot_is_filled() -> None:
    finder = _Finder()
    existing = [{"person_id": 4, "name": NAMES[4], "reasons": []}]
    out = _nodes(finder).c6_score(_state(finder, recommendations=existing, declined_ids=[2]))

    assert [r["person_id"] for r in out["recommendations"]] == [4, 3, 5]


def test_a_referral_is_seated_first_even_without_evidence() -> None:
    finder = _Finder(order=(2, 3, 4))
    out = _nodes(finder).c6_score(_state(finder, referred_responder_id=5))

    recs = out["recommendations"]
    assert [r["person_id"] for r in recs] == [5, 2, 3]
    assert recs[0]["reasons"][0] == {"type": "referral", "detail": "担当者からの推薦"}
    assert out["referred_responder_id"] is None


def test_a_referral_with_evidence_keeps_its_reasons_after_the_referral() -> None:
    finder = _Finder()
    out = _nodes(finder).c6_score(_state(finder, referred_responder_id=4))

    top = out["recommendations"][0]
    assert top["person_id"] == 4
    assert [r["type"] for r in top["reasons"]] == ["referral", SELF_INTRO]


def test_the_asker_is_never_recommended_even_if_a_row_names_them() -> None:
    finder = _Finder(order=(ASKER, 2))
    rows = [e.as_recommendation() for e in finder._result([ASKER, 2]).experts]
    state = _state(finder)
    state["retrieval"] = {**empty_retrieval(), "experts": rows}

    out = _nodes(finder).c6_score(state)

    assert [r["person_id"] for r in out["recommendations"]] == [2]


def test_c6_does_not_change_the_rows_c4_kept() -> None:
    finder = _Finder()
    state = _state(finder, referred_responder_id=2)
    before = [dict(r) for r in state["retrieval"]["experts"]]

    _nodes(finder).c6_score(state)

    assert state["retrieval"]["experts"] == before


def test_the_person_asked_does_not_see_the_evidence_about_them() -> None:
    """ADR-0012: reasons are shown to the asker only; the hand-off view drops them
    for a member's question and keeps them for the company path."""

    finder = _Finder()
    primary = finder.find(asker_id=ASKER).experts[0].as_recommendation()

    member_view = responder_view(primary, org_id=ORG)
    company_view = responder_view(primary, org_id=None)

    assert member_view.reasons == []
    assert member_view.person_id == "E002"
    assert member_view.name == NAMES[2]
    assert [r.type for r in company_view.reasons] == [SELF_INTRO]
    assert primary["reasons"]  # the asker's copy is left as it was


def test_someone_who_opted_out_after_c4_is_not_brought_back() -> None:
    finder = _Finder()
    state = _state(finder, declined_ids=[2])
    finder.gone = {3}

    out = _nodes(finder).c6_score(state)

    assert [r["person_id"] for r in out["recommendations"]] == [4, 5]


def test_a_referral_to_someone_who_left_after_c4_is_dropped() -> None:
    finder = _Finder()
    state = _state(finder, referred_responder_id=4)
    finder.gone = {4}

    out = _nodes(finder).c6_score(state)

    assert [r["person_id"] for r in out["recommendations"]] == [2, 3, 5]
    assert all(r["reasons"][0]["type"] != "referral" for r in out["recommendations"])


def test_sources_without_the_answer_offer_the_best_person() -> None:
    """The document route: the sources were close but held no answer (no composer
    wired here), so the person the evidence points to is offered as the fallback."""

    agent = _agent(_Finder(), sources=_Sources(confidence=0.5))
    state, config = _ask(agent, "org-document")

    assert state["route"] == "document"
    assert not agent.get_state(config).next
    assert state["fallback_responder"]["person_id"] == 2
    assert state["answer"] == (
        "資料の中に、この質問の答えは見つかりませんでした。岩田 颯太さんに相談できます。"
    )

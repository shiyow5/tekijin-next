"""A question that asks for a person reaches the hand-off (#108), with fakes only."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Any

import pytest

from tekijin.agent.graph import build_agent
from tekijin.agent.protocols import IntentResult, SelfAnswerResult
from tekijin.agent.route import DOCUMENT, PERSON, RouteDecision, decide_route, seek_person
from tekijin.agent.state import empty_retrieval
from tekijin.agent.stubs import KeywordIntentModel
from tekijin.llm.schemas import IntentSchema
from tekijin.llm.vllm import VllmIntentModel
from tekijin.retrieval.experts import SELF_INTRO, Evidence, Expert, ExpertResult, rank_experts
from tekijin.retrieval.fragments import CitedEvidence
from tekijin.retrieval.sources import SourceSearchResult

NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
ORG = 3
ASKER = 1
NAMES = {2: "岩田 颯太", 3: "小野 咲", 4: "高梨 湊"}


# --------------------------------------------------------------------------- #
# C1: the schema, the vLLM adapter and the keyword stub
# --------------------------------------------------------------------------- #
class _FakeStructured:
    def __init__(self, result) -> None:
        self._result = result

    def invoke(self, _prompt):
        return self._result


def test_the_schema_defaults_to_not_seeking_a_person() -> None:
    assert IntentSchema().seeks_person is False
    assert IntentSchema.model_validate({"seeks_person": True}).seeks_person is True
    assert "seeks_person" in IntentSchema.model_json_schema()["properties"]


def test_the_vllm_adapter_passes_seeks_person_through() -> None:
    model = _FakeStructured(
        IntentSchema(situation="赤道儀の不調", confidence=0.8, seeks_person=True)
    )
    result = VllmIntentModel(model=model).analyze("赤道儀を見てくれる人を探しています", None)
    assert result.seeks_person is True


def test_a_refused_question_never_seeks_a_person() -> None:
    model = _FakeStructured(IntentSchema(out_of_scope=True, confidence=0.9, seeks_person=True))
    result = VllmIntentModel(model=model).analyze("q", None)
    assert result.out_of_scope is True and result.seeks_person is False


def test_the_prompt_mentions_seeks_person() -> None:
    system = VllmIntentModel.prompt("q", None)[0][1]
    assert "seeks_person" in system


@pytest.mark.parametrize(
    "question",
    [
        "赤道儀の調子が悪いので見てくれる人を探しています",
        "天の川をきれいに撮るコツを教えてくれる人はいますか",
        "補助金の申請書の書き方を相談したいです",
        "プラネタリウムの解説の台本について相談できる人はいますか",
        "星座の神話にくわしい人を探しています",
        "GitHub Pagesの設定で困っています。誰に聞けばいいですか",
        "合宿に望遠鏡を車で運んでくれそうな人はいますか",
    ],
)
def test_the_stub_sees_a_request_for_a_person(question: str) -> None:
    assert KeywordIntentModel().analyze(question, None).seeks_person is True


@pytest.mark.parametrize(
    "question",
    ["部室の鍵はどこで借りられますか", "部員の中で一番成績が悪いのは誰ですか", "今日の天気"],
)
def test_the_stub_leaves_other_questions_alone(question: str) -> None:
    assert KeywordIntentModel().analyze(question, None).seeks_person is False


# --------------------------------------------------------------------------- #
# C5: seek_person on top of decide_route
# --------------------------------------------------------------------------- #
def _document_retrieval(candidates: list[int]) -> Any:
    return {
        **empty_retrieval(),
        "document_confidence": 0.45,
        "source_confidence": 0.45,
        "candidate_people": candidates,
        "people_confidence": 0.25,
    }


def test_a_person_seeker_with_candidates_goes_to_the_person_route() -> None:
    retrieval = _document_retrieval([2, 3])
    before = decide_route(retrieval)
    assert before.route == DOCUMENT

    after = seek_person(before, retrieval, seeks_person=True)

    assert after.route == PERSON
    assert after.confidence == pytest.approx(0.25)


def test_without_a_candidate_the_route_is_unchanged() -> None:
    retrieval = _document_retrieval([])
    before = decide_route(retrieval)
    assert seek_person(before, retrieval, seeks_person=True) is before


def test_a_question_that_does_not_seek_a_person_is_unchanged() -> None:
    retrieval = _document_retrieval([2])
    before = decide_route(retrieval)
    assert seek_person(before, retrieval, seeks_person=False) is before


def test_a_person_decision_is_kept_as_it_was() -> None:
    decision = RouteDecision(PERSON, "既定", 0.4)
    assert seek_person(decision, _document_retrieval([2]), seeks_person=True) is decision


# --------------------------------------------------------------------------- #
# The graph, with fakes
# --------------------------------------------------------------------------- #
class _Embedder:
    def encode(self, texts: Sequence[str], *, kind: str = "passage") -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _Intent:
    def __init__(self, seeks: bool) -> None:
        self.seeks = seeks

    def analyze(self, question, asker, *, context=None) -> IntentResult:
        return IntentResult(confidence=0.9, seeks_person=self.seeks)


class _Sources:
    """Sources close enough for the document route, holding one citable chunk."""

    def search(self, query, org_id, query_vector, scope) -> SourceSearchResult:
        return SourceSearchResult(
            hits=[{"chunk_id": 10, "source_item_id": 20, "score": 0.03}], confidence=0.45
        )

    def evidence(self, hits, org_id, scope) -> list[CitedEvidence]:
        return [CitedEvidence(source_id="src_20", kind="source", text="赤道儀の整備は岩田さん")]


class _Finder:
    def __init__(self, members: Sequence[int] = (2, 3, 4)) -> None:
        self.members = list(members)

    def find(self, **kwargs: Any) -> ExpertResult:
        evidence = [
            Evidence(
                kind=SELF_INTRO,
                member_id=m,
                rank=rank,
                cosine=0.3,
                reason={"type": SELF_INTRO, "detail": f"自己紹介: {NAMES[m]}"},
            )
            for rank, m in enumerate(self.members, start=1)
        ]
        names = {m: (NAMES[m], None) for m in self.members}
        return rank_experts(evidence, names, floors={SELF_INTRO: 0.0})

    def describe(self, org_id: int, member_ids) -> list[Expert]:
        return [
            Expert(
                member_id=m, name=NAMES[m], dept=None, raw_score=0.0, reasons=(), kinds=frozenset()
            )
            for m in dict.fromkeys(member_ids)
            if m in NAMES
        ]


class _Composer:
    def __init__(self) -> None:
        self.calls = 0

    def compose(self, question, evidence) -> SelfAnswerResult:
        self.calls += 1
        return SelfAnswerResult(
            answer="赤道儀の整備は岩田さんが担当しています。",
            cited_source_ids=[e.source_id for e in evidence],
            grounded=True,
        )


class _LegacyMustNotRun:
    def search(self, query, *, query_vector=None):
        raise AssertionError("a member's question reached the undivided tables")


class _ScorerMustNotRun:
    def rank(self, *args, **kwargs):
        raise AssertionError("a member's question reached the company scorer")


def _agent(*, seeks: bool, finder=None, composer=None, additive: bool = True):
    return build_agent(
        _Embedder(),
        None,
        intent_model=_Intent(seeks),
        retriever=_LegacyMustNotRun(),
        scorer=_ScorerMustNotRun(),
        self_answer_model=composer or _Composer(),
        additive_self_answer_enabled=additive,
        source_search=_Sources(),
        expert_finder=finder if finder is not None else _Finder(),
    )


def _ask(agent, thread: str, org_id: int | None = ORG):
    config = {"configurable": {"thread_id": thread}}
    state = agent.invoke(
        {
            "question": "赤道儀を見てくれる人を探しています",
            "asker": {"id": ASKER},
            "org_id": org_id,
            "now": NOW,
        },
        config,
    )
    return state, config


def test_a_person_seeker_reaches_the_hand_off_with_the_cited_answer_alongside() -> None:
    agent = _agent(seeks=True)
    state, config = _ask(agent, "seek")

    assert agent.get_state(config).next == ("send",)
    assert state["route"] == PERSON
    assert [r["person_id"] for r in state["recommendations"]] == [2, 3, 4]
    assert state["draft"]
    # #413 on a member's question: the organization's own sources are cited.
    assert state["additive_answer_text"] == "赤道儀の整備は岩田さんが担当しています。"
    assert [c["source_id"] for c in state["additive_citations"]] == ["src_20"]
    assert not state.get("self_answer_grounded")


def test_without_seeking_a_person_the_sources_answer_as_before() -> None:
    agent = _agent(seeks=False)
    state, config = _ask(agent, "answer")

    assert not agent.get_state(config).next
    assert state["route"] == DOCUMENT
    assert state["self_answer_grounded"] is True
    assert state["answer"] == "赤道儀の整備は岩田さんが担当しています。"


def test_a_person_seeker_with_nobody_found_gets_the_answer_as_before() -> None:
    agent = _agent(seeks=True, finder=_Finder(members=()))
    state, config = _ask(agent, "nobody")

    assert not agent.get_state(config).next
    assert state["route"] == DOCUMENT
    assert state["self_answer_grounded"] is True


def test_the_additive_answer_stays_off_when_disabled() -> None:
    composer = _Composer()
    agent = _agent(seeks=True, composer=composer, additive=False)
    state, config = _ask(agent, "no-additive")

    assert agent.get_state(config).next == ("send",)
    assert not state.get("additive_answer_text")
    assert composer.calls == 0


class _Retriever:
    def search(self, query, *, query_vector=None):
        return {
            **empty_retrieval(),
            "documents": [{"doc_id": "doc_1", "score": 0.1}],
            "document_confidence": 0.45,
            "candidate_people": [7],
            "people_confidence": 0.1,
        }


class _Scorer:
    def rank(self, topics, candidates, asker_id, now, *, top_k=3, **kwargs):
        return {"recommendations": []}


def test_a_question_without_an_organization_keeps_its_route() -> None:
    """seeks_person is read for a member's question only."""

    agent = build_agent(
        _Embedder(),
        None,
        intent_model=_Intent(True),
        retriever=_Retriever(),
        scorer=_Scorer(),
        source_search=_Sources(),
        expert_finder=_Finder(),
    )
    state, _ = _ask(agent, "company", org_id=None)

    assert state["seeks_person"] is True
    assert state["route"] == DOCUMENT

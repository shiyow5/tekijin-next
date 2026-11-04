"""Database-free unit tests for the agent's pure logic.

Covers the C5 route decision (all branches + tunable thresholds + determinism),
the conditional-edge routers, and the deterministic LLM stubs. No DB, no model,
no network.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tekijin.agent import graph as graph_mod
from tekijin.agent.nodes import AgentNodes, _top_by_score
from tekijin.agent.protocols import IntentResult
from tekijin.agent.route import (
    DOCUMENT,
    DOCUMENT_SIM,
    PERSON,
    PERSON_WEAK_SIM,
    PRIOR_ANSWER,
    PRIOR_ANSWER_SIM,
    decide_route,
)
from tekijin.agent.state import AgentState, PastAnswer, RetrievalResult
from tekijin.agent.stubs import (
    MAX_FOLLOWUPS,
    KeywordIntentModel,
    RuleAnswerabilityModel,
    RuleSufficiencyModel,
    TemplateDraftModel,
    collect_known_values,
)


def _retrieval(
    *,
    answer: float = 0.0,
    document: float = 0.0,
    people_sim: float = 0.0,
    people: Sequence[int] = (),
    past_answers: list[PastAnswer] | None = None,
    reuse: int = 0,
) -> RetrievalResult:
    # C5 routes on the absolute cosine similarities (*_confidence), not RRF scores.
    # ``past_answers`` defaults to one entry whenever answer confidence is set, so
    # the prior_answer gate (needs actual past answers) is satisfied by default.
    if past_answers is None:
        answer_hit: PastAnswer = {
            "qa_id": "a",
            "score": 0.01,
            "responder_id": 7,
            "reuse_count": reuse,
        }
        past_answers = [answer_hit] if answer else []
    return {
        "past_answers": list(past_answers),
        "documents": [],
        "candidate_people": list(people),
        "answer_confidence": answer,
        "document_confidence": document,
        "people_confidence": people_sim,
        "person_question_similarity": {},
    }


# --------------------------------------------------------------------------- #
# C5 route decision (on absolute cosine similarity)
# --------------------------------------------------------------------------- #
def test_route_prior_answer_when_answer_is_near_duplicate() -> None:
    r = _retrieval(answer=0.90, people_sim=0.95, people=[1, 2])
    decision = decide_route(r)
    assert decision.route == PRIOR_ANSWER
    assert decision.confidence == pytest.approx(0.90)
    assert "過去QA" in decision.reason


def test_route_prior_answer_at_exact_threshold() -> None:
    # Boundary: exactly at the bar counts (>=).
    assert decide_route(_retrieval(answer=PRIOR_ANSWER_SIM, people=[1])).route == PRIOR_ANSWER
    just_below = _retrieval(answer=PRIOR_ANSWER_SIM - 0.001, people=[1])
    assert decide_route(just_below).route == PERSON


def test_route_no_prior_answer_without_past_answers() -> None:
    # Fix 2: a near-duplicate *question* with NO past answers must NOT route to
    # prior_answer (nobody to hand off to) — falls back to person.
    r = _retrieval(answer=0.95, people=[1], past_answers=[])
    assert decide_route(r).route == PERSON


# --------------------------------------------------------------------------- #
# #327: corpus-count routing for prior_answer (reuse_count, not cosine)
# --------------------------------------------------------------------------- #
def test_corpus_count_routing_off_by_default() -> None:
    # A heavily-reused answer with LOW cosine (Nemotron's real regime) stays on
    # person when the feature is off (prior_answer_reuse_min=None) — unchanged C5.
    r = _retrieval(answer=0.30, people=[1], reuse=8)
    assert decide_route(r).route == PERSON


def test_corpus_count_routing_fires_on_reused_answer() -> None:
    # With the reuse floor on, a low-cosine but heavily-reused top answer routes
    # prior_answer — the route cosine could never separate (0.30 < PRIOR_ANSWER_SIM).
    r = _retrieval(answer=0.30, people=[1], reuse=5)
    decision = decide_route(r, prior_answer_reuse_min=3)
    assert decision.route == PRIOR_ANSWER
    assert "再利用" in decision.reason


def test_corpus_count_routing_respects_reuse_min() -> None:
    # Below the reuse floor -> not canonical enough, stays person.
    r = _retrieval(answer=0.30, people=[1], reuse=2)
    assert decide_route(r, prior_answer_reuse_min=3).route == PERSON


def test_corpus_count_routing_respects_relevance_floor() -> None:
    # A reused answer that is essentially off-topic (cosine below the noise floor)
    # does NOT fire — reuse_count discriminates, the floor screens pure noise.
    r = _retrieval(answer=0.05, people=[1], reuse=9)
    decision = decide_route(r, prior_answer_reuse_min=3, prior_answer_relevance_floor=0.15)
    assert decision.route == PERSON


def test_corpus_count_routing_needs_past_answers() -> None:
    # No past answers to hand off to -> never fires even with high reuse configured.
    r = _retrieval(answer=0.0, people=[1], past_answers=[])
    assert decide_route(r, prior_answer_reuse_min=3).route == PERSON


def test_route_document_when_person_signal_weak() -> None:
    # No near-duplicate answer, weak profile match, strong document -> document,
    # even though candidate people are present (Fix A: was dead before).
    r = _retrieval(answer=0.10, document=0.75, people_sim=0.30, people=[1, 2, 3])
    decision = decide_route(r)
    assert decision.route == DOCUMENT
    assert decision.confidence == pytest.approx(0.75)
    assert "文書" in decision.reason


def test_route_document_at_exact_thresholds() -> None:
    # document_confidence == DOCUMENT_SIM and people_confidence just below the weak
    # bar -> document; people_confidence == PERSON_WEAK_SIM (not weak) -> person.
    at_bar = _retrieval(document=DOCUMENT_SIM, people_sim=PERSON_WEAK_SIM - 0.001, people=[1])
    assert decide_route(at_bar).route == DOCUMENT
    people_ok = _retrieval(document=DOCUMENT_SIM, people_sim=PERSON_WEAK_SIM, people=[1])
    assert decide_route(people_ok).route == PERSON


def test_route_high_answer_without_past_answers_does_not_demote_to_document() -> None:
    # Guard (route.py document branch): answer_conf == PRIOR_ANSWER_SIM but
    # past_answers is empty, so prior_answer is skipped by its gate. A strong
    # document + weak people would otherwise demote — but the ``answer_conf <
    # prior_answer_sim`` term keeps this strong-answer query on the PERSON line.
    r = _retrieval(
        answer=PRIOR_ANSWER_SIM,  # == bar, so NOT < prior_answer_sim
        document=DOCUMENT_SIM,  # == bar
        people_sim=PERSON_WEAK_SIM - 0.001,  # weak profile match
        people=[1],
        past_answers=[],  # no answers -> prior_answer gate fails
    )
    assert decide_route(r).route == PERSON  # not DOCUMENT


def test_route_person_when_profile_match_is_strong() -> None:
    # A strong profile match keeps the person route even with a strong document.
    r = _retrieval(answer=0.10, document=0.85, people_sim=0.80, people=[1, 2, 3])
    decision = decide_route(r)
    assert decision.route == PERSON
    assert decision.confidence == pytest.approx(0.80)  # max(people, answer)
    assert "主線" in decision.reason


def test_route_person_fallback_when_nothing() -> None:
    decision = decide_route(_retrieval())
    assert decision.route == PERSON
    assert decision.confidence == 0.0


def test_route_thresholds_are_tunable() -> None:
    r = _retrieval(answer=0.82, people=[1])
    # Raise the prior_answer bar above the confidence -> falls back to person.
    assert decide_route(r, prior_answer_sim=0.95).route == PERSON


def test_route_missing_confidence_keys_default_to_zero() -> None:
    # A retriever that OMITS the confidence fields -> the .get() defaults kick in
    # -> all-zero -> person (cast: intentionally a partial payload).
    partial = cast(RetrievalResult, {"candidate_people": [1]})
    assert decide_route(partial).route == PERSON


# --------------------------------------------------------------------------- #
# conditional-edge routers
# --------------------------------------------------------------------------- #
def test_after_c1_routes_off_topic() -> None:
    assert graph_mod._after_c1({"out_of_scope": True}) == "off_topic"
    assert graph_mod._after_c1({"out_of_scope": False}) == "c2_sufficiency"


def test_after_c2_routes_on_sufficiency() -> None:
    assert graph_mod._after_c2({"sufficient": True}) == "c3_embed"
    assert graph_mod._after_c2({"sufficient": False}) == "ask"


def test_after_c5_returns_route() -> None:
    assert graph_mod._after_c5({"route": DOCUMENT}) == DOCUMENT
    assert graph_mod._after_c5({}) == PERSON  # default


def test_after_c6_and_send_routers() -> None:
    assert graph_mod._after_c6({"recommendations": [{"person_id": 1}]}) == "c7_draft"
    assert graph_mod._after_c6({"recommendations": []}) == "no_candidate"
    assert graph_mod._after_send({"outcome": "declined"}) == "reroute"
    assert graph_mod._after_send({"outcome": "accepted"}) == "c8_update"
    # redraft loops back through C7 to regenerate the draft (#260).
    assert graph_mod._after_send(cast("AgentState", {"outcome": "redraft"})) == "c7_draft"


def test_after_send_reconfirms_unexpected_outcome() -> None:
    # Fix B: anything but accepted/declined loops back to send (never c8_update).
    assert graph_mod._after_send({"outcome": None}) == "send"
    # "garbage" is not a valid Outcome literal -> cast the intentionally-bad input.
    assert graph_mod._after_send(cast("AgentState", {"outcome": "garbage"})) == "send"
    assert graph_mod._after_send({}) == "send"


# --------------------------------------------------------------------------- #
# node helpers + reset validation (Fix I, K)
# --------------------------------------------------------------------------- #
def _nodes() -> AgentNodes:
    class _Stub:
        def analyze(self, *a, **k):  # pragma: no cover - not called here
            raise AssertionError

    # These tests only exercise reset()/helpers, which never call the models, so a
    # single Any-typed stub satisfies every dependency slot.
    stub: Any = _Stub()
    return AgentNodes(
        intent_model=stub,
        sufficiency_model=stub,
        draft_model=stub,
        embedder=stub,
        retriever=stub,
        scorer=stub,
    )


def _nodes_with_intent(intent_model: Any) -> AgentNodes:
    stub: Any = object()
    return AgentNodes(
        intent_model=intent_model,
        sufficiency_model=stub,
        draft_model=stub,
        embedder=stub,
        retriever=stub,
        scorer=stub,
    )


def test_c1_deterministic_filter_overrides_a_permissive_model() -> None:
    # #155: even if the intent MODEL waves a disallowed question through as a benign
    # in-scope result, the deterministic net forces out_of_scope and blanks topics —
    # so a model swap cannot regress the rejection.
    class _PermissiveIntent:
        def analyze(self, *_a, **_k):
            return IntentResult(topics=["ネットワーク"], out_of_scope=False, confidence=0.9)

    nodes = _nodes_with_intent(_PermissiveIntent())
    out = nodes.c1_intent({"question": "社員全員の自宅住所の一覧が欲しいです。"})
    assert out["out_of_scope"] is True
    assert out["topics"] == [] and out["products"] == []
    assert out["intent_confidence"] == 0.0


def test_c1_lets_a_clean_question_through_to_the_model() -> None:
    # The net only ADDS rejections: a clean question still uses the model's verdict.
    class _CleanIntent:
        def analyze(self, *_a, **_k):
            return IntentResult(topics=["ネットワーク・VPN"], out_of_scope=False, confidence=0.7)

    nodes = _nodes_with_intent(_CleanIntent())
    out = nodes.c1_intent({"question": "VPNの設定手順を教えてください。"})
    assert out["out_of_scope"] is False
    assert out["topics"] == ["ネットワーク・VPN"]
    assert out["intent_confidence"] == 0.7


# --------------------------------------------------------------------------- #
# C4 query expansion (#371): fold C1 topics into the retrieval query (feat-gate)
# --------------------------------------------------------------------------- #
class _RecordingRetriever:
    """Records each ``search`` call as ``(query, query_vector)``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def search(self, query: str, *, query_vector: Any = None) -> dict[str, Any]:
        self.calls.append((query, query_vector))
        return {"candidate_people": []}


def _nodes_for_retrieve(retriever: Any, *, query_expansion_enabled: bool = False) -> AgentNodes:
    stub: Any = object()
    return AgentNodes(
        intent_model=stub,
        sufficiency_model=stub,
        draft_model=stub,
        embedder=stub,
        retriever=retriever,
        scorer=stub,
        query_expansion_enabled=query_expansion_enabled,
    )


def test_c4_retrieve_default_uses_raw_query_and_reuses_c3_vector() -> None:
    # OFF (default): byte-for-byte the pre-#371 behaviour — raw question, reused C3
    # embedding (no re-embed), topics ignored by retrieval.
    rec = _RecordingRetriever()
    nodes = _nodes_for_retrieve(rec)
    nodes.c4_retrieve({"question": "Q", "topics": ["A", "B"], "query_vector": [0.1]})
    assert rec.calls == [("Q", [0.1])]


def test_c4_retrieve_expansion_folds_topics_and_reembeds() -> None:
    # ON + topics present: the retrieval query is the question plus the C1 topics,
    # and the reused C3 vector (which embeds only the raw question) is dropped so the
    # dense channel re-embeds the expanded string.
    rec = _RecordingRetriever()
    nodes = _nodes_for_retrieve(rec, query_expansion_enabled=True)
    nodes.c4_retrieve({"question": "Q", "topics": ["A", "B"], "query_vector": [0.1]})
    assert rec.calls == [("Q A B", None)]


# --------------------------------------------------------------------------- #
# #405: question-fit wiring — C6 forwards C4's qsim map to the scorer iff enabled
# --------------------------------------------------------------------------- #
class _RecordingScorer:
    """Records the ``question_similarity`` kwarg each ``rank`` call receives."""

    def __init__(self) -> None:
        self.qsim_seen: list[Any] = []
        self.kwarg_present: list[bool] = []
        self.candidates_seen: list[list[int]] = []
        self.top_k_seen: list[int] = []

    def rank(self, topics, candidates, asker_id, now, *, top_k=3, **kwargs) -> dict:
        self.kwarg_present.append("question_similarity" in kwargs)
        self.qsim_seen.append(kwargs.get("question_similarity"))
        self.candidates_seen.append(list(candidates))
        self.top_k_seen.append(top_k)
        return {"recommendations": []}


class _ReturningScorer:
    """A scorer double that returns real rows, so tests can assert on ``reasons``."""

    def rank(self, topics, candidates, asker_id, now, *, top_k=3, **kwargs) -> dict:
        return {
            "recommendations": [
                {"person_id": c, "name": f"E{c}", "score": 0.5, "reasons": []}
                for c in list(candidates)[:top_k]
            ]
        }


class _FakeRoster:
    """An :class:`EmployeeSource` double — only ``.id`` is read (#87)."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def list_employees(self) -> list[Any]:
        return [SimpleNamespace(id=i) for i in self._ids]


def _nodes_for_score(
    scorer: Any,
    *,
    question_fit_enabled: bool = False,
    employee_source: Any = None,
) -> AgentNodes:
    stub: Any = object()
    return AgentNodes(
        intent_model=stub,
        sufficiency_model=stub,
        draft_model=stub,
        embedder=stub,
        retriever=stub,
        scorer=scorer,
        question_fit_enabled=question_fit_enabled,
        employee_source=employee_source,
    )


def _c6_state(qsim: dict[int, float]) -> dict[str, Any]:
    import datetime as dt

    retrieval = _retrieval(people=[1, 2])
    retrieval["person_question_similarity"] = qsim
    return {
        "topics": ["ネットワーク・VPN"],
        "retrieval": retrieval,
        "now": dt.datetime(2026, 8, 22, 0, 0, 0),
        "asker": {"id": 0},
    }


def test_c6_forwards_question_similarity_when_enabled() -> None:
    # ON: the scorer receives C4's per-person qsim map so it can add a question-fit
    # term. (Routing is unaffected — C5 does not read the scorer.)
    scorer = _RecordingScorer()
    nodes = _nodes_for_score(scorer, question_fit_enabled=True)
    nodes.c6_score(_c6_state({1: 0.8, 2: 0.2}))
    assert scorer.kwarg_present == [True]
    assert scorer.qsim_seen == [{1: 0.8, 2: 0.2}]


def test_c6_omits_question_similarity_when_disabled() -> None:
    # OFF (default): rank() is called exactly as pre-#405 — the kwarg is not even
    # passed, so a scorer double that predates #405 keeps working byte-identically.
    scorer = _RecordingScorer()
    nodes = _nodes_for_score(scorer, question_fit_enabled=False)
    nodes.c6_score(_c6_state({1: 0.8, 2: 0.2}))
    assert scorer.kwarg_present == [False]


def _c6_referral_state(referred: int | None, **over: Any) -> dict[str, Any]:
    state = _c6_state({})
    state["referred_responder_id"] = referred
    state.update(over)
    return state


def test_c6_seats_a_referred_person_at_rank_1() -> None:
    # #518: a responder referral seats the named person (99, not in the C4 pool [1,2])
    # at rank 1 and annotates why, while the pool fills the remaining slots.
    nodes = _nodes_for_score(_ReturningScorer())
    recs = nodes.c6_score(_c6_referral_state(99))["recommendations"]
    assert recs[0]["person_id"] == 99
    assert recs[0]["reasons"][0] == {"type": "referral", "detail": "担当者からの推薦"}
    assert [r["person_id"] for r in recs[1:]] == [1, 2]  # pool fills the rest


def test_c6_referral_pin_is_dormant_when_unset() -> None:
    # No referral -> byte-identical: just the scored pool, no injected rank-1, and
    # the result never carries the clear key (return shape unchanged from develop).
    nodes = _nodes_for_score(_ReturningScorer())
    out = nodes.c6_score(_c6_referral_state(None))
    assert [r["person_id"] for r in out["recommendations"]] == [1, 2]
    assert "referred_responder_id" not in out


def test_c6_consumes_the_referral_after_seating() -> None:
    # #518 review (HIGH): the referral is cleared once scored, so a later unrelated
    # decline reverts to the normal keep-survivors reroute instead of wiping again.
    nodes = _nodes_for_score(_ReturningScorer())
    out = nodes.c6_score(_c6_referral_state(99))
    assert out["recommendations"][0]["person_id"] == 99
    assert out["referred_responder_id"] is None


def test_c6_referral_pin_never_seats_the_asker() -> None:
    # Referring to the asker (id 0 here) is dropped — they can't answer their own Q.
    nodes = _nodes_for_score(_ReturningScorer())
    recs = nodes.c6_score(_c6_referral_state(0))["recommendations"]
    assert [r["person_id"] for r in recs] == [1, 2]


def test_c6_referral_pin_takes_precedence_over_prior_answer_pin() -> None:
    # A responder's explicit referral (9) wins over a prior_answer pin (5).
    nodes = _nodes_for_score(_ReturningScorer())
    recs = nodes.c6_score(_c6_referral_state(9, pinned_responder_id=5, route=PRIOR_ANSWER))[
        "recommendations"
    ]
    assert recs[0]["person_id"] == 9


def test_c6_scores_the_whole_roster_when_wired() -> None:
    # #87: C4 narrows the candidate set to "people appearing in the top chunks",
    # which drops people who HOLD the evidence but whose chunks did not rank.
    # Wired, C6 scores everyone — here 3 and 4 are on the roster but not in C4's
    # set, and they still reach the scorer.
    scorer = _RecordingScorer()
    nodes = _nodes_for_score(scorer, employee_source=_FakeRoster([1, 2, 3, 4]))
    nodes.c6_score(_c6_state({}))
    assert scorer.candidates_seen == [[1, 2, 3, 4]]


def test_c6_keeps_c4_candidates_when_not_wired() -> None:
    # OFF (default): byte-identical to pre-#87 — only C4's set is scored.
    scorer = _RecordingScorer()
    nodes = _nodes_for_score(scorer)
    nodes.c6_score(_c6_state({}))
    assert scorer.candidates_seen == [[1, 2]]


def test_c6_roster_still_drops_declined() -> None:
    # Widening the pool must not resurrect a candidate the responder already
    # declined. (The asker is dropped inside the scorer, not here, so this test
    # deliberately does not cover it — that is unchanged by #87.)
    scorer = _RecordingScorer()
    nodes = _nodes_for_score(scorer, employee_source=_FakeRoster([1, 2, 3, 4]))
    state = _c6_state({})
    state["declined_ids"] = [2, 3]
    nodes.c6_score(state)
    assert scorer.candidates_seen == [[1, 4]]


def test_c6_roster_backfill_skips_already_shown_candidates() -> None:
    # The reroute path is where a widened pool is most likely to misbehave: the
    # survivors of a decline keep their slots, so C6 must top up only the freed
    # ones and must never re-score someone already on screen. Here candidate 1
    # survived and 3 was declined, so only 2 and 4 may reach the scorer — and only
    # for the two freed slots.
    scorer = _RecordingScorer()
    nodes = _nodes_for_score(scorer, employee_source=_FakeRoster([1, 2, 3, 4]))
    state = _c6_state({})
    state["recommendations"] = [{"person_id": 1, "name": "既存", "score": 0.9, "reasons": []}]
    state["declined_ids"] = [3]
    nodes.c6_score(state)
    assert scorer.candidates_seen == [[2, 4]]
    assert scorer.top_k_seen == [2]


def test_c4_retrieve_expansion_without_topics_falls_back_to_raw() -> None:
    # ON but no topics (C1 found none): nothing to expand, so stay on the raw-query
    # path and keep reusing the C3 vector — never degrade a topic-less run.
    rec = _RecordingRetriever()
    nodes = _nodes_for_retrieve(rec, query_expansion_enabled=True)
    nodes.c4_retrieve({"question": "Q", "topics": [], "query_vector": [0.1]})
    assert rec.calls == [("Q", [0.1])]


def test_top_by_score_picks_max_and_handles_empty() -> None:
    assert _top_by_score([]) is None
    items = [{"doc_id": "a", "score": 0.01}, {"doc_id": "b", "score": 0.03}]
    top = _top_by_score(items)
    assert top is not None and top["doc_id"] == "b"
    reversed_top = _top_by_score(list(reversed(items)))
    assert reversed_top is not None and reversed_top["doc_id"] == "b"  # order-independent


def test_reset_validates_question_and_now() -> None:
    import datetime as dt

    nodes = _nodes()
    good_now = dt.datetime(2026, 8, 22, 12, 0, 0)
    with pytest.raises(ValueError, match="question is required"):
        nodes.reset({"now": good_now})
    with pytest.raises(ValueError, match="question is required"):
        nodes.reset({"question": "   ", "now": good_now})
    with pytest.raises(ValueError, match="now is required"):
        nodes.reset({"question": "q"})
    aware = dt.datetime(2026, 8, 22, tzinfo=dt.timezone(dt.timedelta(hours=9)))
    with pytest.raises(ValueError, match="naive"):
        nodes.reset({"question": "q", "now": aware})


def test_reset_clears_per_question_control_fields() -> None:
    import datetime as dt

    fresh = _nodes().reset(
        {"question": "q", "now": dt.datetime(2026, 8, 22), "followup_count": 5, "declined_ids": [9]}
    )
    assert fresh["followup_count"] == 0
    assert fresh["declined_ids"] == []
    assert fresh["answer"] is None
    assert fresh["recommendations"] == []
    assert fresh["retrieval"]["candidate_people"] == []


# --------------------------------------------------------------------------- #
# C1 intent stub
# --------------------------------------------------------------------------- #
def test_intent_extracts_topics_products_and_type() -> None:
    result = KeywordIntentModel().analyze("VPNの拠点間接続の技術相談です", {"dept": "営業"})
    assert "ネットワーク・VPN" in result.topics
    assert "VPN" in result.products
    assert result.question_type == "技術相談"
    assert result.out_of_scope is False
    assert 0.0 < result.confidence <= 1.0


def test_intent_covers_all_22_canonical_topics() -> None:
    from tekijin.agent.stubs import TOPIC_KEYWORDS

    # Fix 2: the stub can extract every one of the 22 canonical topics — and the
    # keys match the scorer's exact-topic join (skills.topic) verbatim.
    assert len(TOPIC_KEYWORDS) == 22
    model = KeywordIntentModel()
    # Spot-check topics that were missing before (EC / SNS / mobile / 基幹 …).
    assert "ECサイト構築" in model.analyze("ECサイトの構築を相談したい", None).topics
    assert "SNS運用" in model.analyze("SNSの運用代行について", None).topics
    assert "モバイルアプリ開発" in model.analyze("スマホアプリの開発の件", None).topics
    assert "基幹システム" in model.analyze("基幹システムのERP刷新", None).topics
    assert "購買・仕入れ" in model.analyze("調達と仕入れの相談です", None).topics
    assert "総務・法務" in model.analyze("契約審査など法務の相談", None).topics


def test_intent_flags_out_of_scope() -> None:
    result = KeywordIntentModel().analyze("今日の天気を教えて", None)
    assert result.out_of_scope is True
    assert result.question_type == "業務外"
    assert result.confidence == 0.9


def test_chitchat_is_clarified_not_deflected() -> None:
    # Design decision (Fix 7): a bare greeting is low-signal but in-scope for a
    # work helpdesk, so C1 does NOT mark it out_of_scope, and C2 asks to clarify
    # (rather than deflecting like genuine off-topic input).
    intent = KeywordIntentModel().analyze("こんにちは", None)
    assert intent.out_of_scope is False
    result = RuleSufficiencyModel().check("こんにちは", intent, 0)
    assert result.sufficient is False
    assert "具体的" in (result.followup_question or "")


def test_intent_classifies_quote_admin_chitchat() -> None:
    model = KeywordIntentModel()
    assert model.analyze("この製品の見積をお願いします", None).question_type == "見積"
    assert model.analyze("経費精算の手続きについて", None).question_type == "事務手続き"
    assert model.analyze("こんにちは、よろしくお願いします", None).question_type == "雑談"
    # No signal at all -> plain product QA.
    assert model.analyze("これについて教えて", None).question_type == "製品QA"


def test_intent_short_abbrev_matches_on_word_boundary() -> None:
    # Fix 1: short ASCII abbreviations must not fire inside English words.
    model = KeywordIntentModel()
    for false_positive in (
        "security",  # would wrongly match "ec"
        "please review this project",  # "pr" / "ec"
        "how do I connect the cable",  # "ec"
        "improve the process",  # "pr"
        "ability check",  # "bi"
    ):
        result = model.analyze(false_positive, None)
        assert result.topics == []
        assert result.products == []
    # But a genuine EC question still hits (boundary via the Japanese char).
    assert "ECサイト構築" in model.analyze("ECサイトを構築したい", None).topics
    assert "EC" in model.analyze("現行のECで相談", None).products


def test_greeting_with_topic_is_technical_consult() -> None:
    # Fix 4: a concrete topic overrides the greeting classification, so the
    # substantive request is not skipped as chitchat.
    result = KeywordIntentModel().analyze("こんにちは、ネットワークの技術相談です", None)
    assert result.question_type == "技術相談"
    assert "ネットワーク・VPN" in result.topics


# --------------------------------------------------------------------------- #
# C1 intent topic mediation via retrieved context (#69)
# --------------------------------------------------------------------------- #
def test_intent_accepts_context_and_is_backward_compatible() -> None:
    # The new keyword-only `context` defaults to None -> identical to today.
    model = KeywordIntentModel()
    without = model.analyze("VPNの拠点間接続の技術相談です", None)
    with_none = model.analyze("VPNの拠点間接続の技術相談です", None, context=None)
    assert without.topics == with_none.topics
    assert without.confidence == with_none.confidence


def test_context_fragments_surface_a_topic_the_question_missed() -> None:
    # The question words the need without a canonical topic keyword, but a
    # retrieved fragment names one -> mediation adds it (#69: vocabulary bridge).
    model = KeywordIntentModel()
    question = "取引先とのやり取りの履歴をまとめて残したい"
    assert model.analyze(question, None).topics == []  # no keyword in the question
    context = ["過去のQ&A: SFAで顧客管理の履歴を残す方法 / CRMに商談履歴を蓄積します"]
    result = model.analyze(question, None, context=context)
    assert "CRM・営業支援" in result.topics


def test_context_does_not_move_question_type_or_confidence() -> None:
    # #275 review (HIGH): context ONLY adds to the emitted topics that feed C6.
    # question_type and confidence stay driven by the user's actual question, so a
    # merely topic-adjacent retrieval hit cannot force a needless C2 follow-up or
    # lift confidence past the clarify threshold for the wrong reason.
    model = KeywordIntentModel()
    question = "取引先とのやり取りの履歴をまとめて残したい"  # no question-side topic/product
    base = model.analyze(question, None)
    mediated = model.analyze(question, None, context=["CRMで商談履歴を蓄積します"])
    assert "CRM・営業支援" in mediated.topics  # topics ARE mediated
    assert mediated.question_type == base.question_type  # but type is not
    assert mediated.confidence == base.confidence  # and confidence is not


def test_context_never_pulls_an_out_of_scope_question_back_in() -> None:
    # Off-topic input stays out_of_scope even if a fragment mentions a topic —
    # context is reference evidence, not the user's ask.
    model = KeywordIntentModel()
    result = model.analyze("今日の天気を教えて", None, context=["CRMの営業支援について"])
    assert result.out_of_scope is True
    assert result.topics == []


def test_context_derived_topics_keep_canonical_order() -> None:
    # Topics stay in the canonical vocabulary order regardless of whether they
    # came from the question or the context (deterministic output).
    model = KeywordIntentModel()
    # Question yields セキュリティ; context yields CRM・営業支援 (earlier in the table).
    result = model.analyze("UTMのセキュリティ相談", None, context=["CRMの顧客管理について"])
    assert "CRM・営業支援" in result.topics and "セキュリティ" in result.topics
    from tekijin.agent.stubs import TOPIC_KEYWORDS

    order = list(TOPIC_KEYWORDS)
    assert result.topics == sorted(result.topics, key=order.index)


# --------------------------------------------------------------------------- #
# C2 sufficiency stub
# --------------------------------------------------------------------------- #
def test_sufficiency_asks_once_for_missing_slots() -> None:
    # Topic-only question (no product, no 拠点) -> both slots missing.
    q = "ネットワークの技術相談です"
    intent = KeywordIntentModel().analyze(q, None)
    result = RuleSufficiencyModel().check(q, intent, followup_count=0)
    assert result.sufficient is False
    assert set(result.missing) == {"現行製品", "対象拠点数"}
    assert result.followup_question and "教えてください" in result.followup_question


def test_sufficiency_satisfied_when_slots_present() -> None:
    q = "現行のUTMから移行、対象は3拠点です"
    intent = KeywordIntentModel().analyze(q, None)
    result = RuleSufficiencyModel().check(q, intent, followup_count=0)
    assert result.sufficient is True
    assert result.missing == []


def test_sufficiency_proceeds_after_max_followups() -> None:
    intent = KeywordIntentModel().analyze("UTMの技術相談です", None)
    result = RuleSufficiencyModel().check("UTMの技術相談です", intent, followup_count=MAX_FOLLOWUPS)
    assert result.sufficient is True  # never ask more than once


def test_sufficiency_low_intent_asks_to_clarify() -> None:
    # Fix H: no topic extracted / low confidence -> treat as unclear intent and
    # ask to clarify, instead of proceeding to a dead-end no_candidate.
    q = "これについて教えて"
    intent = KeywordIntentModel().analyze(q, None)  # topics empty, low confidence
    result = RuleSufficiencyModel().check(q, intent, 0)
    assert result.sufficient is False
    assert result.missing == ["相談内容"]
    assert result.followup_question and "具体的" in result.followup_question


def test_sufficiency_keeps_unresolved_slots_after_cap() -> None:
    # Fix F: at the follow-up cap, proceed (sufficient) but KEEP unresolved slots
    # so C7 can flag them — do not silently clear missing.
    q = "ネットワークの技術相談です"  # both required slots missing
    intent = KeywordIntentModel().analyze(q, None)
    result = RuleSufficiencyModel().check(q, intent, MAX_FOLLOWUPS)
    assert result.sufficient is True
    assert set(result.missing) == {"現行製品", "対象拠点数"}


def test_sufficiency_requires_slot_values_not_labels() -> None:
    # Fix 6: bare labels ("現行環境", "拠点間") without a product value or a site
    # COUNT do not satisfy the slots -> still asks for clarification.
    q = "現行環境と拠点間ネットワークの技術相談です"
    intent = KeywordIntentModel().analyze(q, None)
    assert intent.products == []  # no known product value mentioned
    result = RuleSufficiencyModel().check(q, intent, followup_count=0)
    assert result.sufficient is False
    assert set(result.missing) == {"現行製品", "対象拠点数"}


def test_sufficiency_site_count_needs_a_number() -> None:
    # "拠点" without a number is not a count; "3拠点" is. (The intent is unused for
    # this slot, but pass a real IntentResult to keep the call well-typed.)
    intent = IntentResult()
    assert RuleSufficiencyModel._slot_present("対象拠点数", "拠点間の相談", intent) is False
    assert RuleSufficiencyModel._slot_present("対象拠点数", "対象は5拠点です", intent) is True


# --------------------------------------------------------------------------- #
# C2 speed (#376): skip the sufficiency LLM call when C1 is confident+on-topic
# --------------------------------------------------------------------------- #
def _sufficiency_state(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "question": "VPNの設定手順を教えてください。",
        "topics": ["ネットワーク・VPN"],
        "products": [],
        "situation": None,
        "intent_confidence": 0.9,
        "followup_count": 0,
    }
    base.update(over)
    return base


class _ExplodingSufficiency:
    def check(self, *_a: Any, **_k: Any):
        raise AssertionError("sufficiency model must not be called when can_route")


def _c2_nodes(sufficiency: Any) -> AgentNodes:
    stub: Any = object()
    return AgentNodes(
        intent_model=stub,
        sufficiency_model=sufficiency,
        draft_model=stub,
        embedder=stub,
        retriever=stub,
        scorer=stub,
    )


def test_c2_skips_sufficiency_llm_when_confident_and_on_topic() -> None:
    # A confident, on-topic C1 result is already routable (the #113 valve), so C2
    # must NOT invoke the (LLM) sufficiency model — it decides from C1 alone,
    # removing one of the three serial generations on the critical path (#376).
    # question_type "製品QA" (the default) carries no required slots -> missing [].
    out = _c2_nodes(_ExplodingSufficiency()).c2_sufficiency(_sufficiency_state())
    assert out["sufficient"] is True
    assert out["missing"] == [] and out["followup_question"] is None
    assert out["intent_unresolved"] is False


def test_c2_fast_path_still_surfaces_missing_slots_for_the_draft() -> None:
    # #376 regression guard: skipping the LLM must NOT drop the estimate slots the
    # hand-off draft flags. A confident 技術相談 with no product / no site count still
    # yields missing=[現行製品, 対象拠点数] (computed deterministically, no LLM call),
    # so C7's 「補足いただきたい点」hint is preserved exactly as before #376.
    state = _sufficiency_state(question_type="技術相談", products=[])
    out = _c2_nodes(_ExplodingSufficiency()).c2_sufficiency(state)
    assert out["sufficient"] is True
    assert out["missing"] == ["現行製品", "対象拠点数"]
    assert out["followup_question"] is None


def test_c2_calls_sufficiency_llm_when_not_routable() -> None:
    # Below the confidence threshold: C2 still consults the model (unchanged path).
    from tekijin.agent.protocols import SufficiencyResult

    seen: list[float] = []

    class _RecordingSufficiency:
        def check(self, question: Any, intent: Any, followup_count: Any):
            seen.append(intent.confidence)
            return SufficiencyResult(
                sufficient=False, missing=["拠点数"], followup_question="拠点数は？"
            )

    out = _c2_nodes(_RecordingSufficiency()).c2_sufficiency(
        _sufficiency_state(intent_confidence=0.3)
    )
    assert seen == [0.3]  # the model WAS consulted on the low-confidence path
    assert out["sufficient"] is False and out["followup_question"] == "拠点数は？"


# --------------------------------------------------------------------------- #
# C7 draft stub
# --------------------------------------------------------------------------- #
def test_draft_is_polite_and_names_the_responder() -> None:
    draft = TemplateDraftModel().draft(
        "VPNの相談です", {"name": "高梨", "dept": "技術部"}, {"dept": "営業"}, ["現行製品"]
    )
    assert "高梨さん（技術部）" in draft
    assert "VPNの相談です" in draft
    assert "現行製品" in draft
    assert "お世話になっております" in draft


def test_draft_handles_missing_name_and_dept() -> None:
    draft = TemplateDraftModel().draft("質問です", {}, None, [])
    assert "ご担当者さん" in draft
    assert "補足" not in draft  # no missing slots -> no supplement section


def test_collect_known_values_from_products_and_site_count() -> None:
    # #175: a slot's concrete value is surfaced iff it is actually filled.
    values = collect_known_values("VPNの技術相談です。5拠点あります", "技術相談", ["VPN"])
    assert values == {"現行製品": "VPN", "対象拠点数": "5拠点"}


def test_collect_known_values_rescans_question_when_products_empty() -> None:
    # Defensive fallback: if `products` is left empty but a known product name is
    # in the text, the value is still recovered from the question.
    values = collect_known_values("見積もりです CRM を使っています", "見積", [])
    assert values["現行製品"] == "CRM"
    assert "対象拠点数" not in values  # not mentioned -> stays unfilled


def test_collect_known_values_prefers_earliest_mentioned_product() -> None:
    # The value is shown to the responder as a confirmed premise, so pick the
    # product mentioned first in the text, not the keyword-table order (VPN<CRM).
    values = collect_known_values("技術相談です。CRMとVPNを併用しています", "技術相談", [])
    assert values["現行製品"] == "CRM"


def test_collect_known_values_empty_for_types_without_slots() -> None:
    assert collect_known_values("こんにちは", "雑談", []) == {}


def test_draft_injects_situation_and_known_values() -> None:
    draft = TemplateDraftModel().draft(
        "VPNの相談です",
        {"name": "高梨", "dept": "技術部"},
        None,
        [],
        situation="拠点間接続が不安定",
        topics=["ネットワーク・VPN"],
        known_values={"現行製品": "VPN", "対象拠点数": "5拠点"},
    )
    assert "【背景】" in draft and "拠点間接続が不安定" in draft
    assert "現行製品：VPN" in draft
    assert "対象拠点数：5拠点" in draft
    assert "ネットワーク・VPN" in draft


def _nodes_with_draft(draft_model: Any) -> AgentNodes:
    stub: Any = object()
    return AgentNodes(
        intent_model=stub,
        sufficiency_model=stub,
        draft_model=draft_model,
        embedder=stub,
        retriever=stub,
        scorer=stub,
    )


def test_c7_draft_injects_structured_context_into_the_draft() -> None:
    # #175 (the real fix): C7 passes situation/topics + the filled slot values to
    # the draft, so the hand-off reflects the system's structured understanding
    # (question and premises), not a thin echo of the raw question.
    nodes = _nodes_with_draft(TemplateDraftModel())
    state: AgentState = {
        "question": "技術相談です。CRM を使っていて 3拠点 あります",
        "question_type": "技術相談",
        "products": ["CRM"],
        "situation": "移行を検討中",
        "topics": ["CRM・営業支援"],
        "missing": [],
        "recommendations": [{"person_id": 7, "name": "田中", "dept": "営業"}],
        "asker": None,
    }
    draft = nodes.c7_draft(state)["draft"]
    assert "現行製品：CRM" in draft
    assert "対象拠点数：3拠点" in draft
    assert "移行を検討中" in draft
    assert "CRM・営業支援" in draft


def test_c7_draft_never_double_lists_a_filled_slot() -> None:
    # Defensive dedup: even if `missing` still names a slot we now surface as a
    # known value, the draft must not show it as both 確認済み and 補足 (they
    # normally agree, since C2 recomputes `missing` on the re-understood question).
    nodes = _nodes_with_draft(TemplateDraftModel())
    state: AgentState = {
        "question": "技術相談です。CRM を使っていて 3拠点 あります",
        "question_type": "技術相談",
        "products": ["CRM"],
        "missing": ["現行製品", "対象拠点数"],  # stale/contradictory on purpose
        "recommendations": [{"person_id": 7, "name": "田中", "dept": "営業"}],
        "asker": None,
    }
    draft = nodes.c7_draft(state)["draft"]
    assert "現行製品：CRM" in draft
    # both slots are surfaced as known values -> no 補足いただきたい点 section
    assert "補足いただきたい点" not in draft


# --------------------------------------------------------------------------- #
# evidence-sufficiency critic stub (#70)
# --------------------------------------------------------------------------- #
def test_answerability_stub_rejects_when_no_candidate_evidence() -> None:
    result = RuleAnswerabilityModel().assess("海外の知財登録の相談", [])
    assert result.confidence == 0  # nobody to answer -> reject signal
    assert result.reason


def test_answerability_stub_scales_with_evidence_count() -> None:
    model = RuleAnswerabilityModel()
    one = model.assess("VPNの相談", ["社員1: ネットワーク案件3件"])
    three = model.assess(
        "VPNの相談",
        ["社員1: ネットワーク案件3件", "社員2: VPN構築2件", "社員3: 回線移行1件"],
    )
    assert 0 < one.confidence < three.confidence <= 100
    # Blank lines do not count as evidence.
    assert model.assess("q", ["  ", ""]).confidence == 0


# --------------------------------------------------------------------------- #
# answerability evidence builder + router (#70 part2)
# --------------------------------------------------------------------------- #
def test_self_answer_stub_returns_ungrounded_without_evidence() -> None:
    from tekijin.agent.stubs import TemplateSelfAnswerModel

    result = TemplateSelfAnswerModel().compose("VPNの相談", [])
    assert result.grounded is False and result.answer == "" and result.cited_source_ids == []


def test_self_answer_stub_composes_and_cites_from_evidence() -> None:
    from tekijin.agent.stubs import TemplateSelfAnswerModel
    from tekijin.retrieval.fragments import CitedEvidence

    evidence = [
        CitedEvidence("qa_1", "qa", "VPNは保守時間内に更新します"),
        CitedEvidence("doc_3", "document", "ネットワーク運用手順"),
        CitedEvidence("blank", "qa", "   "),  # blank text -> not cited
    ]
    result = TemplateSelfAnswerModel().compose("VPNの相談", evidence)
    assert result.grounded is True
    assert "VPNは保守時間内に更新します" in result.answer
    assert result.cited_source_ids == ["qa_1", "doc_3"]  # blank dropped, links preserved


def test_answerability_evidence_empty_when_no_recommendations() -> None:
    from tekijin.agent.nodes import answerability_evidence

    assert answerability_evidence([]) == []


def test_answerability_evidence_one_line_per_candidate_with_reasons() -> None:
    from tekijin.agent.nodes import answerability_evidence

    recs = [
        {
            "person_id": 1,
            "name": "山田",
            "dept": "技術部",
            "reasons": [
                {"type": "skill", "detail": "VPN構築の実績"},
                {"type": "cert", "detail": "ネットワーク資格"},
            ],
        },
        {"person_id": 2, "name": "佐藤", "dept": None, "reasons": []},
    ]
    lines = answerability_evidence(recs)
    assert len(lines) == 2
    # name/dept + the reason details are all present for the critic to weigh.
    assert "山田（技術部）" in lines[0]
    assert "VPN構築の実績" in lines[0] and "ネットワーク資格" in lines[0]
    # A candidate with no reasons still contributes a line (name only, no dept).
    assert lines[1] == "佐藤"


def test_after_answerability_routes_on_the_boolean() -> None:
    # The node writes ``answerable``; the router only reads it (stays pure).
    assert graph_mod._after_answerability(cast("AgentState", {"answerable": True})) == "c7_draft"
    assert graph_mod._after_answerability(cast("AgentState", {"answerable": False})) == "no_expert"
    # Missing key (critic never ran) is treated as reject, never a silent hand-off.
    assert graph_mod._after_answerability(cast("AgentState", {})) == "no_expert"

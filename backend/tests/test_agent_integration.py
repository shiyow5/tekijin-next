"""Integration tests for the LangGraph agent against live PostgreSQL + pgvector.

Uses the shared ``session`` / ``seed_counts`` / ``fake_embedder`` fixtures. The
LLM nodes are the deterministic stubs and the C4 retriever is usually an injected
fake returning a controlled retrieval dict, so every branch is exercised without
a model or retrieval nondeterminism — while C6 still scores against the real
seeded DB. ``now`` is injected, so runs are reproducible.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from langgraph.types import Command

from tekijin.agent import build_agent
from tekijin.agent.route import DOCUMENT, PERSON, PRIOR_ANSWER
from tekijin.agent.state import RetrievalResult
from tekijin.models.tables import Answer, Question, Skill
from tekijin.retrieval.indexing import embed_corpus

NOW = dt.datetime(2026, 9, 15, 12, 0, 0)
TOPIC = "ネットワーク・VPN"
# A well-formed question: yields topic ネットワーク・VPN and satisfies C2 (mentions a
# current product and 拠点), so no clarification interrupt on the happy path.
GOOD_Q = "現行のVPN機器で3拠点の拠点間接続について相談したいです"
# A genuinely vague question: NO topic keyword extractable, so C1 confidence stays
# below the routing threshold and C2 still asks a clarification. Since the #113
# safety valve only proceeds when C1 already has a confident topic, this is what
# exercises the ask/interrupt path now (a topic-only question routes straight
# through — finding the person is the product's job, not the asker's).
VAGUE_Q = "相談したいことがあります"


class _FixedIntent:
    """C1 stand-in with a topic but sub-threshold confidence.

    Used to exercise the C2 follow-up machinery under the #113 safety valve: a
    topic is present (so C6 can still score and reach the hand-off), but the
    confidence is below the routing threshold, so the valve does NOT auto-proceed
    and the injected/insufficient C2 path is what drives the ask -> cap flow.
    """

    def analyze(self, question, asker):  # noqa: ARG002
        from tekijin.agent.protocols import IntentResult

        return IntentResult(topics=[TOPIC], products=[], question_type="技術相談", confidence=0.4)


class _FixedAnswerability:
    """#70 evidence-sufficiency critic stand-in returning a fixed confidence."""

    def __init__(self, confidence: int) -> None:
        self._confidence = confidence
        self.calls: list[tuple[str, list[str]]] = []

    def assess(self, question, candidate_evidence):
        from tekijin.agent.protocols import AnswerabilityResult

        self.calls.append((question, list(candidate_evidence)))
        return AnswerabilityResult(confidence=self._confidence, reason="判定理由")


class _FixedSelfAnswer:
    """#291 self-answer composer stand-in with a fixed grounded/ungrounded verdict."""

    def __init__(
        self, *, grounded: bool, answer: str = "社内記録による回答です。", cites=()
    ) -> None:
        self._grounded = grounded
        self._answer = answer
        self._cites = list(cites)
        self.calls: list[tuple[str, list]] = []

    def compose(self, question, evidence):
        from tekijin.agent.protocols import SelfAnswerResult

        self.calls.append((question, list(evidence)))
        if not self._grounded:
            return SelfAnswerResult(answer="", cited_source_ids=[], grounded=False)
        return SelfAnswerResult(answer=self._answer, cited_source_ids=self._cites, grounded=True)


class _FakeRetriever:
    """C4 stand-in returning a fixed retrieval dict (controls the C5 route)."""

    def __init__(
        self,
        *,
        answers=(),
        documents=(),
        people=(),
        answer_confidence=0.0,
        document_confidence=0.0,
        people_confidence=0.0,
    ) -> None:
        self._payload: RetrievalResult = {
            "past_answers": list(answers),
            "documents": list(documents),
            "candidate_people": list(people),
            "answer_confidence": answer_confidence,
            "document_confidence": document_confidence,
            "people_confidence": people_confidence,
        }
        self.calls: list[tuple[str, bool]] = []  # (query, got_query_vector)

    def search(self, query: str, *, query_vector: Sequence[float] | None = None) -> RetrievalResult:
        self.calls.append((query, query_vector is not None))
        return self._payload


def _seed_skill(session, sid: str, employee_id: int, topic: str = TOPIC) -> None:
    session.add(Skill(id=sid, employee_id=employee_id, topic=topic, level="中級", source="self"))
    session.flush()


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _init(question: str = GOOD_Q, asker=None) -> dict:
    return {
        "question": question,
        "asker": asker,
        "now": NOW,
        "followup_count": 0,
        "declined_ids": [],
    }


def _is_paused(agent, cfg) -> bool:
    return bool(agent.get_state(cfg).next)


# --------------------------------------------------------------------------- #
# happy path: person route -> recommendations -> hand-off
# --------------------------------------------------------------------------- #
def test_happy_path_person_route(seed_counts, session, fake_embedder) -> None:
    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_happy_{emp}", emp)
    retriever = _FakeRetriever(
        answers=[{"qa_id": "a", "score": 0.005, "responder_id": 1}],
        documents=[{"doc_id": "d", "score": 0.001}],
        people=[1, 2, 3],
    )
    agent = build_agent(fake_embedder, session, retriever=retriever)
    cfg = _cfg("happy")

    state = agent.invoke(_init(), cfg)
    assert state["route"] == PERSON
    assert state["recommendations"]  # C6 scored the candidates
    assert _is_paused(agent, cfg)  # paused at send for the outcome

    final = agent.invoke(Command(resume="accepted"), cfg)
    assert "取り次ぎました" in final["answer"]
    assert final["draft"] and final["recommendations"][0]["name"] in final["answer"]


# --------------------------------------------------------------------------- #
# out_of_scope terminal
# --------------------------------------------------------------------------- #
def test_out_of_scope_is_deflected(seed_counts, session, fake_embedder) -> None:
    agent = build_agent(fake_embedder, session, retriever=_FakeRetriever(people=[1]))
    state = agent.invoke(_init("今日の天気とランチのおすすめを教えて"), _cfg("oos"))
    assert state["out_of_scope"] is True
    assert "お答えできません" in state["answer"]
    assert "recommendations" not in state or not state.get("recommendations")


# --------------------------------------------------------------------------- #
# C2 insufficient -> interrupt -> resume -> proceeds
# --------------------------------------------------------------------------- #
def test_insufficient_triggers_interrupt_then_resumes(seed_counts, session, fake_embedder) -> None:
    for emp in (1, 2):
        _seed_skill(session, f"sk_insuf_{emp}", emp)
    agent = build_agent(fake_embedder, session, retriever=_FakeRetriever(people=[1, 2]))
    cfg = _cfg("insuf")

    # Vague question (no extractable topic) -> C2 asks a clarification interrupt.
    agent.invoke(_init(VAGUE_Q), cfg)
    assert _is_paused(agent, cfg)
    state_at_ask = agent.get_state(cfg)
    assert state_at_ask.next == ("ask",)

    # Reply introduces the topic (VPN); the followup cap makes C2 sufficient next pass.
    agent.invoke(Command(resume="現行はVPN機器で、対象は3拠点です"), cfg)
    after = agent.get_state(cfg).values
    assert after["followup_count"] == 1
    assert after["sufficient"] is True
    assert _is_paused(agent, cfg)  # now paused at send

    final = agent.invoke(Command(resume="accepted"), cfg)
    assert "取り次ぎました" in final["answer"]


def test_confident_topic_skips_clarification(seed_counts, session, fake_embedder) -> None:
    # #113 safety valve: a confident, on-topic consultation must NOT be pushed back
    # for estimate-style slots (現行製品/対象拠点数). "ネットワークの技術相談です" yields the
    # topic ネットワーク・VPN at confidence ≥ threshold, so C2 proceeds straight to the
    # hand-off instead of asking — finding the responder is the product's job.
    for emp in (1, 2):
        _seed_skill(session, f"sk_conf_{emp}", emp)
    agent = build_agent(fake_embedder, session, retriever=_FakeRetriever(people=[1, 2]))
    cfg = _cfg("conf")
    state = agent.invoke(_init("ネットワークの技術相談です"), cfg)
    assert state["sufficient"] is True
    assert agent.get_state(cfg).next == ("send",)  # reached hand-off, no ask interrupt


# --------------------------------------------------------------------------- #
# C5 three routes
# --------------------------------------------------------------------------- #
def test_prior_answer_route(seed_counts, session, fake_embedder) -> None:
    _seed_skill(session, "sk_pa_1", 1)
    retriever = _FakeRetriever(
        answers=[{"qa_id": "a", "score": 0.05, "responder_id": 1}],
        answer_confidence=0.9,  # near-duplicate past QA (absolute cosine)
        people=[1],
    )
    agent = build_agent(fake_embedder, session, retriever=retriever)
    cfg = _cfg("pa")
    state = agent.invoke(_init(), cfg)
    assert state["route"] == PRIOR_ANSWER
    assert state["prior_answer_note"] and "回答" in state["prior_answer_note"]
    # prior_answer still flows to C6 (hand off to the responder).
    assert _is_paused(agent, cfg)  # reached send after C6/C7


def test_document_route_is_terminal(seed_counts, session, fake_embedder) -> None:
    retriever = _FakeRetriever(
        documents=[{"doc_id": "doc_0007", "score": 0.05}],
        document_confidence=0.8,  # strongly on-topic document
        people_confidence=0.2,  # weak person signal
        people=[1, 2, 3],  # candidates present, but weak -> document still wins
    )
    agent = build_agent(fake_embedder, session, retriever=retriever)
    cfg = _cfg("doc")
    state = agent.invoke(_init(), cfg)
    assert state["route"] == DOCUMENT
    assert "doc_0007" in state["answer"]
    assert state["document_id"] == "doc_0007"  # surfaced for the viewer (#143)
    assert not _is_paused(agent, cfg)  # terminal, no interrupt


def test_document_route_offers_a_person_fallback(seed_counts, session, fake_embedder) -> None:
    # #279: a document-routed question that DOES have an expert behind the weak
    # profile no longer dead-ends at zero person-recall. C6 runs on the document
    # route too, so the DOCUMENT terminal cites the doc AND names a fallback person
    # — while staying terminal (self-resolution first, no hand-off interrupt).
    for emp in (1, 2):
        _seed_skill(session, f"sk_docfb_{emp}", emp)  # real topic evidence
    retriever = _FakeRetriever(
        documents=[{"doc_id": "doc_0007", "score": 0.05}],
        document_confidence=0.8,  # strongly on-topic document
        people_confidence=0.2,  # weak profile match -> document route
        people=[1, 2],
    )
    agent = build_agent(fake_embedder, session, retriever=retriever)
    cfg = _cfg("docfb")
    state = agent.invoke(_init(), cfg)

    assert state["route"] == DOCUMENT
    assert state["document_id"] == "doc_0007"
    assert state["recommendations"]  # C6 ranked the fallback experts
    assert state["recommendations"][0]["name"] in state["answer"]  # named as a backstop
    assert "doc_0007" in state["answer"]  # document is still the main line
    assert not _is_paused(agent, cfg)  # terminal — no send interrupt


# --------------------------------------------------------------------------- #
# answerability critic (#70): reject weak in-house evidence, accept strong
# --------------------------------------------------------------------------- #
def test_answerability_accepts_and_proceeds_to_handoff(seed_counts, session, fake_embedder) -> None:
    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_ans_ok_{emp}", emp)
    critic = _FixedAnswerability(confidence=85)  # strong in-house track record
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_FakeRetriever(people=[1, 2, 3]),
        answerability_model=critic,
        answerability_threshold=40,
    )
    cfg = _cfg("ans_ok")
    state = agent.invoke(_init(), cfg)
    assert state["route"] == PERSON
    assert state["answerability_confidence"] == 85 and state["answerable"] is True
    assert state["recommendations"]  # candidates survive the critic
    assert _is_paused(agent, cfg)  # reached the send hand-off, as before
    # The critic saw the ranked candidates' evidence, not an empty list.
    assert critic.calls and critic.calls[0][1]

    final = agent.invoke(Command(resume="accepted"), cfg)
    assert "取り次ぎました" in final["answer"]


def test_answerability_rejects_to_no_expert_terminal(seed_counts, session, fake_embedder) -> None:
    # C6 DID rank candidates, but the critic judges the in-house evidence
    # insufficient (海外法務/知財… "痕跡が無い領域") -> graceful no_expert terminal,
    # NOT a hand-off to a weak match.
    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_ans_ng_{emp}", emp)
    critic = _FixedAnswerability(confidence=15)  # below threshold
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_FakeRetriever(people=[1, 2, 3]),
        answerability_model=critic,
        answerability_threshold=40,
    )
    cfg = _cfg("ans_ng")
    state = agent.invoke(_init(), cfg)
    assert state["route"] == PERSON
    assert state["answerability_confidence"] == 15 and state["answerable"] is False
    assert not _is_paused(agent, cfg)  # terminal — no send interrupt, no hand-off
    assert "詳しいメンバーの記録が、組織の中に見つかりませんでした" in state["answer"]
    assert state.get("draft") is None  # never drafted a hand-off


def test_answerability_bypassed_on_document_route(seed_counts, session, fake_embedder) -> None:
    # The document route is self-resolution, not a hand-off — the critic must not
    # sit on that path (a low score must not turn a document answer into no_expert).
    for emp in (1, 2):
        _seed_skill(session, f"sk_ans_doc_{emp}", emp)
    critic = _FixedAnswerability(confidence=5)  # would reject if it were consulted
    retriever = _FakeRetriever(
        documents=[{"doc_id": "doc_0007", "score": 0.05}],
        document_confidence=0.8,
        people_confidence=0.2,
        people=[1, 2],
    )
    agent = build_agent(
        fake_embedder,
        session,
        retriever=retriever,
        answerability_model=critic,
        answerability_threshold=40,
    )
    state = agent.invoke(_init(), _cfg("ans_doc"))
    assert state["route"] == DOCUMENT
    assert "doc_0007" in state["answer"]  # document terminal, unaffected by the critic
    assert critic.calls == []  # the critic was never consulted on the document route


def test_answerability_unwired_is_passthrough(seed_counts, session, fake_embedder) -> None:
    # Default (no critic): the graph is the pre-#70 flow — no critique node, no
    # answerability keys in state, straight to the send hand-off.
    for emp in (1, 2):
        _seed_skill(session, f"sk_ans_off_{emp}", emp)
    agent = build_agent(fake_embedder, session, retriever=_FakeRetriever(people=[1, 2]))
    cfg = _cfg("ans_off")
    state = agent.invoke(_init(), cfg)
    assert _is_paused(agent, cfg)  # reached send, unchanged
    # The critique/terminal nodes are not even added to the compiled graph.
    assert "answerability" not in agent.get_graph().nodes
    assert "no_expert" not in agent.get_graph().nodes
    # reset() seeds the keys to inert defaults, but no critic ever ran to set them.
    assert state["answerable"] is False


# --------------------------------------------------------------------------- #
# self-answer (#291): answer from data on the document route, or fall back
# --------------------------------------------------------------------------- #
def _doc_retriever() -> _FakeRetriever:
    return _FakeRetriever(
        documents=[{"doc_id": "doc_001", "score": 0.05}],
        document_confidence=0.8,  # strong document -> document route
        people_confidence=0.2,
        people=[1, 2],
    )


def test_self_answer_grounded_terminates_with_citations(
    seed_counts, session, fake_embedder
) -> None:
    for emp in (1, 2):
        _seed_skill(session, f"sk_sa_ok_{emp}", emp)
    composer = _FixedSelfAnswer(grounded=True, answer="保守時間内に更新します。", cites=["doc_001"])
    agent = build_agent(
        fake_embedder, session, retriever=_doc_retriever(), self_answer_model=composer
    )
    state = agent.invoke(_init(), _cfg("sa_ok"))

    assert state["route"] == DOCUMENT  # C5 still routes on data strength
    assert state["self_answer_grounded"] is True
    assert state["answer"] == "保守時間内に更新します。"  # answered from data, not a hand-off
    assert state["self_answer_citations"] == [{"source_id": "doc_001", "kind": "document"}]
    assert not _is_paused(agent, _cfg("sa_ok"))  # terminal, no send interrupt
    assert composer.calls  # the composer was consulted


def test_self_answer_ungrounded_falls_back_to_document(seed_counts, session, fake_embedder) -> None:
    for emp in (1, 2):
        _seed_skill(session, f"sk_sa_ng_{emp}", emp)
    composer = _FixedSelfAnswer(grounded=False)
    agent = build_agent(
        fake_embedder, session, retriever=_doc_retriever(), self_answer_model=composer
    )
    state = agent.invoke(_init(), _cfg("sa_ng"))

    assert state["route"] == DOCUMENT
    assert state["self_answer_grounded"] is False
    assert "doc_001" in state["answer"]  # fell back to the document terminal (#279 fallback)
    assert not _is_paused(agent, _cfg("sa_ng"))


def test_self_answer_not_consulted_on_person_route(seed_counts, session, fake_embedder) -> None:
    # A person-routed question (weak data) must NOT run the self-answer composer —
    # it goes straight to the hand-off, unchanged.
    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_sa_pr_{emp}", emp)
    composer = _FixedSelfAnswer(grounded=True, cites=["x"])
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_FakeRetriever(people=[1, 2, 3]),  # person route
        self_answer_model=composer,
    )
    cfg = _cfg("sa_pr")
    state = agent.invoke(_init(), cfg)
    assert state["route"] == PERSON
    assert composer.calls == []  # composer never consulted on the person route
    assert _is_paused(agent, cfg)  # reached the send hand-off


# --------------------------------------------------------------------------- #
# additive self-answer (#413): cite past knowledge ALONGSIDE the person hand-off
# --------------------------------------------------------------------------- #
def _person_retriever_with_data(doc_conf: float) -> _FakeRetriever:
    # Strong person signal (people_conf >= PERSON_WEAK_SIM) + a document below the
    # DOCUMENT_SIM demotion bar -> PERSON route, with ``doc_conf`` controlling the
    # additive floor gate.
    return _FakeRetriever(
        documents=[{"doc_id": "doc_001", "score": 0.05}],
        document_confidence=doc_conf,
        people_confidence=0.5,
        people=[1, 2, 3],
    )


def test_additive_answer_cites_on_person_route_and_still_hands_off(
    seed_counts, session, fake_embedder
) -> None:
    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_add_ok_{emp}", emp)
    composer = _FixedSelfAnswer(grounded=True, answer="過去の類似回答です。", cites=["doc_001"])
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_person_retriever_with_data(0.25),  # >= floor 0.20, < DOCUMENT_SIM
        self_answer_model=composer,
        additive_self_answer_enabled=True,
    )
    cfg = _cfg("add_ok")
    state = agent.invoke(_init(), cfg)

    assert state["route"] == PERSON  # still a person route
    assert composer.calls  # additive compose ran (floor cleared)
    # The cited answer is attached ADDITIVELY, without terminating or self-resolving.
    assert state["additive_answer_text"] == "過去の類似回答です。"
    assert state["additive_citations"] == [{"source_id": "doc_001", "kind": "document"}]
    assert state.get("self_answer_grounded") is False  # NOT marked self-resolved
    assert _is_paused(agent, cfg)  # the hand-off still happens (person recall intact)


def test_additive_answer_gated_below_floor_skips_compose(
    seed_counts, session, fake_embedder
) -> None:
    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_add_lo_{emp}", emp)
    composer = _FixedSelfAnswer(grounded=True, cites=["doc_001"])
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_person_retriever_with_data(0.10),  # below floor 0.20
        self_answer_model=composer,
        additive_self_answer_enabled=True,
    )
    cfg = _cfg("add_lo")
    state = agent.invoke(_init(), cfg)

    assert state["route"] == PERSON
    assert composer.calls == []  # gated: no compose LLM call, no latency
    assert not state.get("additive_answer_text")
    assert _is_paused(agent, cfg)  # plain hand-off


def test_additive_answer_ungrounded_leaves_no_reference(
    seed_counts, session, fake_embedder
) -> None:
    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_add_ng_{emp}", emp)
    composer = _FixedSelfAnswer(grounded=False)
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_person_retriever_with_data(0.25),
        self_answer_model=composer,
        additive_self_answer_enabled=True,
    )
    cfg = _cfg("add_ng")
    state = agent.invoke(_init(), cfg)

    assert state["route"] == PERSON
    assert composer.calls  # consulted, but not grounded
    assert not state.get("additive_answer_text")  # no citation surfaced
    assert _is_paused(agent, cfg)


def test_additive_answer_compose_failure_still_hands_off(
    seed_counts, session, fake_embedder
) -> None:
    # #413 safety premise: a composer error on the additive path must degrade to a
    # plain hand-off, NEVER crash the person run (person recall must not regress).
    class _RaisingComposer:
        calls: list = []

        def compose(self, question, evidence):
            self.calls.append((question, list(evidence)))
            raise ValueError("unparseable structured output")

    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_add_err_{emp}", emp)
    composer = _RaisingComposer()
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_person_retriever_with_data(0.25),
        self_answer_model=composer,
        additive_self_answer_enabled=True,
    )
    cfg = _cfg("add_err")
    state = agent.invoke(_init(), cfg)

    assert state["route"] == PERSON
    assert composer.calls  # it was consulted and raised...
    assert not state.get("additive_answer_text")  # ...but the run absorbed it
    assert _is_paused(agent, cfg)  # and the hand-off still happens


def test_additive_answer_off_by_default_person_route_unchanged(
    seed_counts, session, fake_embedder
) -> None:
    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_add_off_{emp}", emp)
    composer = _FixedSelfAnswer(grounded=True, cites=["doc_001"])
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_person_retriever_with_data(0.25),
        self_answer_model=composer,
        # additive_self_answer_enabled defaults False
    )
    cfg = _cfg("add_off")
    state = agent.invoke(_init(), cfg)

    assert state["route"] == PERSON
    assert composer.calls == []  # additive_answer node not even wired
    assert "additive_answer" not in agent.get_graph().nodes
    assert _is_paused(agent, cfg)


def test_self_answer_unwired_leaves_document_route_unchanged(
    seed_counts, session, fake_embedder
) -> None:
    for emp in (1, 2):
        _seed_skill(session, f"sk_sa_off_{emp}", emp)
    agent = build_agent(fake_embedder, session, retriever=_doc_retriever())  # no composer
    state = agent.invoke(_init(), _cfg("sa_off"))
    assert state["route"] == DOCUMENT and "doc_001" in state["answer"]
    assert "self_answer" not in agent.get_graph().nodes  # node not added
    assert state.get("self_answer_grounded") is False  # reset default, never set


def _seed_qa(session, qa_id: str, responder_id: int, body: str = "過去の回答本文") -> None:
    session.add(Question(id=f"q_{qa_id}", asker_id=2, body="過去の質問", topics=[TOPIC]))
    session.flush()
    session.add(Answer(id=qa_id, question_id=f"q_{qa_id}", responder_id=responder_id, body=body))
    session.flush()


def _prior_answer_retriever(qa_id: str, responder_id: int) -> _FakeRetriever:
    return _FakeRetriever(
        answers=[{"qa_id": qa_id, "score": 0.05, "responder_id": responder_id}],
        answer_confidence=0.9,  # near-duplicate past QA -> prior_answer route
        people=[responder_id],
    )


def test_self_answer_grounded_on_prior_answer_terminates(
    seed_counts, session, fake_embedder
) -> None:
    # #291 review: the prior_answer route also runs self-answer. A grounded answer
    # terminates at self_answered (no pinned hand-off), citing the past Q&A.
    _seed_skill(session, "sk_sapa_ok", 1)
    _seed_qa(session, "a_sapa", responder_id=1, body="VPNは保守時間内に更新します")
    composer = _FixedSelfAnswer(
        grounded=True, answer="過去回答より、保守時間内に更新します。", cites=["a_sapa"]
    )
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_prior_answer_retriever("a_sapa", 1),
        self_answer_model=composer,
    )
    cfg = _cfg("sa_pa_ok")
    state = agent.invoke(_init(), cfg)
    assert state["route"] == PRIOR_ANSWER
    assert state["self_answer_grounded"] is True
    assert state["self_answer_citations"] == [{"source_id": "a_sapa", "kind": "qa"}]
    assert not _is_paused(agent, cfg)  # terminal, no hand-off interrupt


def test_self_answer_ungrounded_falls_back_to_prior_answer_handoff(
    seed_counts, session, fake_embedder
) -> None:
    # Not grounded on the prior_answer route -> fall back to the pinned hand-off.
    _seed_skill(session, "sk_sapa_ng", 1)
    _seed_qa(session, "a_sapang", responder_id=1)
    composer = _FixedSelfAnswer(grounded=False)
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_prior_answer_retriever("a_sapang", 1),
        self_answer_model=composer,
    )
    cfg = _cfg("sa_pa_ng")
    state = agent.invoke(_init(), cfg)
    assert state["route"] == PRIOR_ANSWER
    assert state["self_answer_grounded"] is False
    assert _is_paused(agent, cfg)  # fell back to the prior_answer -> send hand-off
    assert agent.get_state(cfg).next == ("send",)


# --------------------------------------------------------------------------- #
# decline -> reroute to the next candidate
# --------------------------------------------------------------------------- #
def test_decline_reroutes_to_next_candidate(seed_counts, session, fake_embedder) -> None:
    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_dec_{emp}", emp)
    retriever = _FakeRetriever(people=[1, 2, 3])
    agent = build_agent(fake_embedder, session, retriever=retriever)
    cfg = _cfg("decline")

    first = agent.invoke(_init(), cfg)
    first_recs = first["recommendations"]
    first_pick = first_recs[0]["person_id"]
    surviving = first_recs[1:]  # ranks 2/3 as shown before the decline

    # The first pick declines -> reroute keeps ranks 2/3 exactly as shown,
    # rather than rescoring everyone (#D5/#206). With only 3 people in the pool
    # total, there is nobody left to backfill the freed slot with, so the kept
    # 2 survivors are the whole result (the 4-candidate test below covers the
    # backfill case).
    agent.invoke(Command(resume="declined"), cfg)
    rerouted = agent.get_state(cfg).values
    assert first_pick in rerouted["declined_ids"]
    new_recs = rerouted["recommendations"]
    assert new_recs == surviving  # byte-identical survivors, nothing else
    second_pick = new_recs[0]["person_id"]
    assert second_pick != first_pick

    final = agent.invoke(Command(resume="accepted"), cfg)
    assert "取り次ぎました" in final["answer"]


def test_decline_reroute_backfills_only_freed_slot(seed_counts, session, fake_embedder) -> None:
    # 4 eligible candidates so a genuinely fresh 4th person is available to
    # backfill the slot freed by the decline, proving c6_score does not
    # rescore ranks 2/3 on reroute (#D5/#206).
    for emp in (1, 2, 3, 4):
        _seed_skill(session, f"sk_dec4_{emp}", emp)
    retriever = _FakeRetriever(people=[1, 2, 3, 4])
    agent = build_agent(fake_embedder, session, retriever=retriever)
    cfg = _cfg("decline4")

    first = agent.invoke(_init(), cfg)
    first_recs = first["recommendations"]
    assert len(first_recs) == 3
    former_second, former_third = first_recs[1], first_recs[2]

    agent.invoke(Command(resume="declined"), cfg)
    rerouted = agent.get_state(cfg).values["recommendations"]
    assert len(rerouted) == 3
    assert rerouted[0] == former_second
    assert rerouted[1] == former_third
    shown_before = {r["person_id"] for r in first_recs}
    assert rerouted[2]["person_id"] not in shown_before  # fresh backfill


def test_no_candidate_terminal(seed_counts, session, fake_embedder) -> None:
    # Nothing retrieved at all -> person fallback with no candidates -> the
    # no_candidate terminal (C6 has nobody to score).
    agent = build_agent(fake_embedder, session, retriever=_FakeRetriever())
    state = agent.invoke(_init(), _cfg("nocand"))
    assert state["route"] == PERSON
    assert state["recommendations"] == []
    assert "適任者が見つかりません" in state["answer"]


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_run_is_deterministic(seed_counts, session, fake_embedder) -> None:
    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_det_{emp}", emp)
    retriever = _FakeRetriever(people=[1, 2, 3])

    def run(thread: str) -> dict:
        agent = build_agent(fake_embedder, session, retriever=retriever)
        cfg = _cfg(thread)
        agent.invoke(_init(), cfg)
        return agent.invoke(Command(resume="accepted"), cfg)

    first = run("det_a")
    second = run("det_b")
    assert first["answer"] == second["answer"]
    assert [r["person_id"] for r in first["recommendations"]] == [
        r["person_id"] for r in second["recommendations"]
    ]
    assert first["route"] == second["route"]


# --------------------------------------------------------------------------- #
# stream updates (the basis for #32 SSE)
# --------------------------------------------------------------------------- #
def test_stream_yields_node_updates(seed_counts, session, fake_embedder) -> None:
    for emp in (1, 2):
        _seed_skill(session, f"sk_stream_{emp}", emp)
    retriever = _FakeRetriever(people=[1, 2])
    agent = build_agent(fake_embedder, session, retriever=retriever)

    nodes = []
    for update in agent.stream(_init(), _cfg("stream"), stream_mode="updates"):
        nodes.extend(update.keys())
    # The deterministic node sequence up to the send interrupt (reset runs first).
    assert nodes[:7] == [
        "reset",
        "c1_intent",
        "c2_sufficiency",
        "c3_embed",
        "c4_retrieve",
        "c5_route",
        "c6_score",
    ]
    assert "c7_draft" in nodes


# --------------------------------------------------------------------------- #
# real HybridRetriever wiring (smoke: no assertion on which people)
# --------------------------------------------------------------------------- #
def test_real_retriever_end_to_end(seed_counts, session, fake_embedder) -> None:
    embed_corpus(session, fake_embedder)  # populate dense vectors for real C4
    agent = build_agent(fake_embedder, session)  # default HybridRetriever
    cfg = _cfg("real")
    state = agent.invoke(_init(), cfg)
    # The real retrieval drove a valid route and the run did not error.
    assert state["route"] in {PERSON, PRIOR_ANSWER, DOCUMENT}
    retrieval = state["retrieval"]
    assert isinstance(retrieval["candidate_people"], list)
    # C4 emits the absolute cosine confidences C5 routes on, in [0, 1].
    for key in ("answer_confidence", "document_confidence", "people_confidence"):
        assert 0.0 <= retrieval[key] <= 1.0


def test_newly_covered_topic_reaches_recommendation(seed_counts, session, fake_embedder) -> None:
    # Fix 2: an EC question (previously uncoverable by the stub) now extracts the
    # ECサイト構築 topic, so a candidate with that skill can be recommended.
    _seed_skill(session, "sk_ec_1", 1, topic="ECサイト構築")
    agent = build_agent(fake_embedder, session, retriever=_FakeRetriever(people=[1]))
    q = "ECサイト構築について、現行のECサイトで3拠点向けに相談したいです"
    state = agent.invoke(_init(q), _cfg("ec"))
    assert "ECサイト構築" in state["topics"]
    assert [r["person_id"] for r in state["recommendations"]] == [1]


def test_c4_reuses_c3_query_vector(seed_counts, session, fake_embedder) -> None:
    # Fix 4: the C3 embedding is passed to C4.search so the dense channels do not
    # re-embed (one encode per query, not two).
    retriever = _FakeRetriever(people=[1])
    _seed_skill(session, "sk_reuse", 1)
    agent = build_agent(fake_embedder, session, retriever=retriever)
    agent.invoke(_init(), _cfg("reuse"))
    assert retriever.calls  # C4 was called
    assert all(got_vector for _, got_vector in retriever.calls)  # always with a vector


# --------------------------------------------------------------------------- #
# fix B: an unexpected outcome never reaches the success terminal
# --------------------------------------------------------------------------- #
def test_unexpected_outcome_does_not_reach_c8(seed_counts, session, fake_embedder) -> None:
    for emp in (1, 2):
        _seed_skill(session, f"sk_bad_{emp}", emp)
    agent = build_agent(fake_embedder, session, retriever=_FakeRetriever(people=[1, 2]))
    cfg = _cfg("badout")

    agent.invoke(_init(), cfg)
    assert agent.get_state(cfg).next == ("send",)
    # A garbage outcome loops back to send (re-confirm) — never c8_update.
    agent.invoke(Command(resume="maybe?"), cfg)
    assert agent.get_state(cfg).next == ("send",)
    assert agent.get_state(cfg).values.get("answer") is None
    # A valid outcome then completes.
    final = agent.invoke(Command(resume="accepted"), cfg)
    assert "取り次ぎました" in final["answer"]


# --------------------------------------------------------------------------- #
# fix D: a 2nd question on the same thread resets per-question control
# --------------------------------------------------------------------------- #
def test_second_question_resets_control_fields(seed_counts, session, fake_embedder) -> None:
    for emp in (1, 2):
        _seed_skill(session, f"sk_reset_{emp}", emp)
    agent = build_agent(fake_embedder, session, retriever=_FakeRetriever(people=[1, 2]))
    cfg = _cfg("twoq")  # SAME thread for both questions

    # Q1: decline the first pick so declined_ids gets populated.
    q1 = agent.invoke(_init(), cfg)
    first_pick = q1["recommendations"][0]["person_id"]
    agent.invoke(Command(resume="declined"), cfg)
    agent.invoke(Command(resume="accepted"), cfg)  # Q1 ends
    assert first_pick in agent.get_state(cfg).values["declined_ids"]

    # Q2: a brand-new question on the SAME thread -> reset clears the carry-over.
    agent.invoke(_init(), cfg)
    q2_state = agent.get_state(cfg).values
    assert q2_state["declined_ids"] == []  # not inherited from Q1
    assert q2_state["followup_count"] == 0
    # The person declined in Q1 is eligible again in Q2.
    assert first_pick in [r["person_id"] for r in q2_state["recommendations"]]


def test_resume_does_not_reset_followup_loop(seed_counts, session, fake_embedder) -> None:
    for emp in (1, 2):
        _seed_skill(session, f"sk_noreset_{emp}", emp)
    agent = build_agent(fake_embedder, session, retriever=_FakeRetriever(people=[1, 2]))
    cfg = _cfg("noreset")
    agent.invoke(_init(VAGUE_Q), cfg)  # vague -> C2 asks
    assert agent.get_state(cfg).next == ("ask",)
    # Resume (NOT a new question) must keep the followup_count it accrued.
    agent.invoke(Command(resume="現行はVPN、3拠点です"), cfg)
    assert agent.get_state(cfg).values["followup_count"] == 1


# --------------------------------------------------------------------------- #
# fix E: the follow-up cap is enforced by the node, not just the stub
# --------------------------------------------------------------------------- #
def test_node_enforces_followup_cap(seed_counts, session, fake_embedder) -> None:
    from tekijin.agent.protocols import SufficiencyResult

    class _NeverSufficient:
        def check(self, question, intent, followup_count):  # ignores the cap
            return SufficiencyResult(sufficient=False, missing=["x"], followup_question="もっと？")

    for emp in (1, 2):
        _seed_skill(session, f"sk_cap_{emp}", emp)
    agent = build_agent(
        fake_embedder,
        session,
        intent_model=_FixedIntent(),  # topic present but sub-threshold -> valve off
        sufficiency_model=_NeverSufficient(),
        retriever=_FakeRetriever(people=[1, 2]),
    )
    cfg = _cfg("cap")
    agent.invoke(_init(), cfg)  # model says insufficient -> ask
    assert agent.get_state(cfg).next == ("ask",)
    # After one clarification the NODE forces sufficiency -> proceeds (no loop).
    agent.invoke(Command(resume="追加情報です"), cfg)
    assert agent.get_state(cfg).next == ("send",)  # reached the hand-off, not ask


def test_c2_fast_path_keeps_missing_in_handoff_draft(seed_counts, session, fake_embedder) -> None:
    # #376 end-to-end: a confident 技術相談 with no product / no site count skips the C2
    # LLM call (the exploding model would raise if consulted), yet the hand-off draft
    # still surfaces the open estimate slots — the deterministic ``missing`` on the
    # fast path flows through C7 exactly as the pre-#376 model's ``missing`` did.
    from tekijin.agent.protocols import IntentResult

    class _ConfidentTechIntent:
        def analyze(self, question, asker):  # noqa: ARG002
            return IntentResult(
                topics=[TOPIC], products=[], question_type="技術相談", confidence=0.9
            )

    class _ExplodingSufficiency:
        def check(self, *_a, **_k):
            raise AssertionError("C2 sufficiency LLM must be skipped when can_route")

    for emp in (1, 2):
        _seed_skill(session, f"sk_miss_{emp}", emp)
    agent = build_agent(
        fake_embedder,
        session,
        intent_model=_ConfidentTechIntent(),
        sufficiency_model=_ExplodingSufficiency(),
        retriever=_FakeRetriever(people=[1, 2]),
    )
    cfg = _cfg("fastmiss")
    state = agent.invoke(_init(question="ネットワークの技術相談です。"), cfg)
    assert agent.get_state(cfg).next == ("send",)  # no ask pause: valve proceeded
    draft = state["draft"]
    assert "補足いただきたい点" in draft
    assert "現行製品" in draft and "対象拠点数" in draft


# --------------------------------------------------------------------------- #
# fix G: prior_answer hands off to the past responder, not a higher scorer
# #307: prior_answer still backfills up to 3 candidates from the general pool
# instead of dead-ending on the one pinned person.
# --------------------------------------------------------------------------- #
def test_prior_answer_pins_the_responder(seed_counts, session, fake_embedder) -> None:
    # Retrieval lists candidate_people [1, 2, 3] but the strong past answer is by
    # employee 5; prior_answer must hand off to 5 first (not lose them to a
    # higher-scoring 1/2/3), then fill the remaining 2 slots from 1/2/3.
    for emp in (1, 2, 3):
        _seed_skill(session, f"sk_pin_{emp}", emp)  # strong candidates
    retriever = _FakeRetriever(
        answers=[{"qa_id": "a", "score": 0.05, "responder_id": 5}],
        answer_confidence=0.9,  # near-duplicate past QA
        people=[1, 2, 3],
    )
    agent = build_agent(fake_embedder, session, retriever=retriever)
    cfg = _cfg("pin")
    state = agent.invoke(_init(), cfg)
    assert state["route"] == PRIOR_ANSWER
    assert state["pinned_responder_id"] == 5
    person_ids = [r["person_id"] for r in state["recommendations"]]
    assert person_ids[0] == 5  # pinned, guaranteed first
    assert len(person_ids) == 3  # #307: backfilled from the general pool
    assert set(person_ids[1:]) <= {1, 2, 3}


def test_prior_answer_falls_back_when_pinned_declines(seed_counts, session, fake_embedder) -> None:
    # Fix 3: the pinned past responder declines -> drop the pin and fall back to
    # the candidate_people pool (never dead-end on a single decline).
    for emp in (1, 2):
        _seed_skill(session, f"sk_pinfb_{emp}", emp)
    retriever = _FakeRetriever(
        answers=[{"qa_id": "a", "score": 0.05, "responder_id": 1}],
        answer_confidence=0.9,
        people=[1, 2],  # fallback pool
    )
    agent = build_agent(fake_embedder, session, retriever=retriever)
    cfg = _cfg("pinfb")
    state = agent.invoke(_init(), cfg)
    assert state["route"] == PRIOR_ANSWER
    # #307: pinned (1) first, then the general pool backfills the 2nd slot.
    assert [r["person_id"] for r in state["recommendations"]] == [1, 2]

    # Pinned responder 1 declines -> un-pin; candidate 2 (already recommended)
    # survives untouched and no further backfill is available (pool exhausted).
    agent.invoke(Command(resume="declined"), cfg)
    rerouted = agent.get_state(cfg).values
    assert 1 in rerouted["declined_ids"]
    assert [r["person_id"] for r in rerouted["recommendations"]] == [2]
    final = agent.invoke(Command(resume="accepted"), cfg)
    assert "取り次ぎました" in final["answer"]


def test_prior_answer_pin_that_is_the_asker_falls_back(seed_counts, session, fake_embedder) -> None:
    # Fix 3: the top past answer's responder IS the asker -> the pin is invalid
    # (they cannot answer their own question); fall back to candidate_people.
    for emp in (1, 2):
        _seed_skill(session, f"sk_pinself_{emp}", emp)
    retriever = _FakeRetriever(
        answers=[{"qa_id": "a", "score": 0.05, "responder_id": 5}],  # responder == asker
        answer_confidence=0.9,
        people=[1, 2],  # fallback pool
    )
    agent = build_agent(fake_embedder, session, retriever=retriever)
    state = agent.invoke(_init(asker={"id": 5}), _cfg("pinself"))
    assert state["route"] == PRIOR_ANSWER
    assert state["pinned_responder_id"] == 5
    person_ids = [r["person_id"] for r in state["recommendations"]]
    assert 5 not in person_ids  # asker never recommended to themselves
    assert set(person_ids) <= {1, 2}


# --------------------------------------------------------------------------- #
# fix 5: unresolved intent (still no topic after one clarification) terminal
# --------------------------------------------------------------------------- #
def test_unresolved_intent_terminal(seed_counts, session, fake_embedder) -> None:
    agent = build_agent(fake_embedder, session, retriever=_FakeRetriever(people=[1]))
    cfg = _cfg("unresolved")
    # No topic extractable -> C2 asks to clarify (interrupt at ask).
    agent.invoke(_init("これについて教えて"), cfg)
    assert agent.get_state(cfg).next == ("ask",)
    # The reply STILL carries no topic -> capped + unresolved intent -> graceful
    # terminal, NOT a silent no_candidate.
    final = agent.invoke(Command(resume="よくわからないのですが"), cfg)
    assert not agent.get_state(cfg).next  # terminated
    assert final["intent_unresolved"] is True
    assert "特定できませんでした" in final["answer"]
    assert not final.get("recommendations")


# --------------------------------------------------------------------------- #
# fix I: a non-string resume is not interpolated into the question
# --------------------------------------------------------------------------- #
def test_ask_ignores_non_string_resume(seed_counts, session, fake_embedder) -> None:
    for emp in (1, 2):
        _seed_skill(session, f"sk_badreply_{emp}", emp)
    agent = build_agent(
        fake_embedder,
        session,
        intent_model=_FixedIntent(),  # topic present but sub-threshold -> valve off, C2 asks
        retriever=_FakeRetriever(people=[1, 2]),
    )
    cfg = _cfg("badreply")
    original = "ネットワークの技術相談です"
    agent.invoke(_init(original), cfg)
    assert agent.get_state(cfg).next == ("ask",)
    # A non-string resume payload is safely ignored (not appended); the cap then
    # lets the run proceed rather than corrupting the question.
    agent.invoke(Command(resume={"unexpected": "payload"}), cfg)
    values = agent.get_state(cfg).values
    assert values["question"] == original  # not corrupted with the dict
    assert agent.get_state(cfg).next == ("send",)


# --------------------------------------------------------------------------- #
# knowledge answer (#357 slice 4c): answer from structured knowledge before C5
# --------------------------------------------------------------------------- #
def _seed_approved_knowledge(session, fake_embedder, source_id, problem, action) -> None:
    from tekijin.data.knowledge import (
        get_knowledge_unit_by_source,
        set_review_status,
        upsert_knowledge_unit,
    )
    from tekijin.knowledge.index import embed_knowledge_units

    upsert_knowledge_unit(
        session,
        kind="case",
        problem=problem,
        action=action,
        result="受注",
        topics=["CRM・営業支援"],
        source_type="daily_report",
        source_id=source_id,
        confidence=0.9,
    )
    session.flush()
    dto = get_knowledge_unit_by_source(session, "daily_report", source_id)
    set_review_status(session, dto.id, "approved")
    session.flush()
    embed_knowledge_units(session, fake_embedder)


def test_knowledge_answer_terminates_before_routing(seed_counts, session, fake_embedder) -> None:
    # A query matching an approved knowledge unit is answered from structure and
    # terminates at self_answered — before C5 routing (bypassing #327/ADR-0007).
    # Use the canonical GOOD_Q so C2 passes; seed a unit whose problem IS that query
    # so the (bag-of-tokens) fake embedding is a strong match.
    _seed_approved_knowledge(session, fake_embedder, "kb_c4c_1", GOOD_Q, "SFA/CRM を提案")
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_FakeRetriever(people=[1, 2]),
        knowledge_answer_min_similarity=0.1,
    )
    state = agent.invoke(_init(), _cfg("kb_ok"))

    assert state["self_answer_grounded"] is True
    assert GOOD_Q in state["answer"]  # the case's problem is surfaced in the answer
    # Cited as the knowledge unit (ku_{id}), kind "knowledge".
    assert len(state["self_answer_citations"]) == 1
    cite = state["self_answer_citations"][0]
    assert cite["kind"] == "knowledge" and cite["source_id"].startswith("ku_")
    # It never reached a person hand-off: terminal, no send interrupt (C5/C6 never ran;
    # any ``route`` value is only the reset() default, not a real routing decision).
    assert not _is_paused(agent, _cfg("kb_ok"))


def test_knowledge_answer_falls_through_when_irrelevant(
    seed_counts, session, fake_embedder
) -> None:
    # No approved knowledge matches (floor above any similarity) -> the run proceeds
    # to normal retrieval/routing exactly as pre-#357 (a person hand-off here).
    _seed_approved_knowledge(session, fake_embedder, "kb_c4c_2", "全く無関係な話題", "何もしない")
    for emp in (1, 2):
        _seed_skill(session, f"sk_kb_{emp}", emp)
    agent = build_agent(
        fake_embedder,
        session,
        retriever=_FakeRetriever(people=[1, 2]),
        knowledge_answer_min_similarity=0.99,  # unreachable floor -> never fires
    )
    state = agent.invoke(_init(), _cfg("kb_fall"))

    assert state["self_answer_grounded"] is False
    assert _is_paused(agent, _cfg("kb_fall"))  # fell through to the person hand-off


def test_knowledge_answer_dormant_when_unwired(seed_counts, session, fake_embedder) -> None:
    # No floor -> no knowledge_answer node is added; graph is the pre-#357 flow.
    agent = build_agent(fake_embedder, session, retriever=_FakeRetriever(people=[1, 2]))
    assert "knowledge_answer" not in agent.get_graph().nodes
    assert "self_answered" not in agent.get_graph().nodes

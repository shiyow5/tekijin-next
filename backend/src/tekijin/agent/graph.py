"""Assemble the C1-C8 flow as a LangGraph ``StateGraph``.

The wiring mirrors the model-definition §1 flowchart one-to-one: rounded boxes
become ``add_node`` and diamonds become ``add_conditional_edges``. LLM nodes
(C1/C2/C7) default to the deterministic stubs; the retriever (C4) and scorer (C6)
default to the real modules but are injectable, so tests can force any branch
without a model, a network call, or retrieval nondeterminism.

Persistence uses an injected checkpointer (default ``MemorySaver``); #32 swaps in
``PostgresSaver`` without touching this graph. ``thread_id`` (per session) drives
``interrupt``/``resume``.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from tekijin.agent.nodes import AgentNodes
from tekijin.agent.protocols import (
    AnswerabilityModel,
    DraftModel,
    ExpertFinder,
    IntentModel,
    KnowledgeSearch,
    Retriever,
    SelfAnswerModel,
    SourceSearch,
    SufficiencyModel,
)
from tekijin.agent.route import DOCUMENT, PERSON, PRIOR_ANSWER
from tekijin.agent.state import AgentState
from tekijin.agent.stubs import KeywordIntentModel, RuleSufficiencyModel, TemplateDraftModel
from tekijin.data.repository import Repository
from tekijin.knowledge.answer import KnowledgeCandidates
from tekijin.retrieval.embedding import Embedder
from tekijin.retrieval.retriever import HybridRetriever
from tekijin.scorer.scorer import ExpertiseScorer
from tekijin.scorer.weights import DEFAULT_WEIGHTS, Weights


def _knowledge_search(
    session: Any,
    floor: float | None,
    given: KnowledgeSearch | None,
    self_answer_model: SelfAnswerModel | None,
) -> KnowledgeSearch | None:
    """What C4 offers verified knowledge from (#118), or ``None`` to offer none."""

    if floor is None or self_answer_model is None:
        return None
    return given or KnowledgeCandidates(session, min_similarity=floor)


# -- conditional-edge routers (pure) -------------------------------------
def _after_c1(state: AgentState) -> str:
    # Node id differs from the "out_of_scope" state key (LangGraph forbids reuse).
    return "off_topic" if state.get("out_of_scope") else "c2_sufficiency"


def _after_c2(state: AgentState) -> str:
    # Capped but still no topic -> graceful "couldn't identify the request"
    # terminal (never silently search on an empty intent and hit no_candidate).
    if state.get("intent_unresolved"):
        return "unresolved_intent"
    return "c3_embed" if state.get("sufficient") else "ask"


def _after_c5(state: AgentState) -> str:
    return state.get("route", PERSON)


def _after_c6(state: AgentState) -> str:
    # #279: a document-routed question runs C6 too, so a real expert behind a
    # weak-profile document is offered as a fallback instead of dead-ending at zero
    # person-recall — but it stays on the DOCUMENT terminal (self-resolution first,
    # no hand-off interrupt), never c7_draft/send.
    if state.get("route") == DOCUMENT:
        return "document"
    return "c7_draft" if state.get("recommendations") else "no_candidate"


def _after_self_answer(state: AgentState) -> str:
    # #291: grounded -> the assistant answered from data; terminate at
    # ``self_answered``. Not grounded (evidence insufficient) -> fall back to the
    # route we arrived on: a document stays on its self-resolution terminal (via
    # C6's #279 person fallback), a prior_answer hands off to the pinned responder.
    if state.get("self_answer_grounded"):
        return "self_answered"
    return "c6_score" if state.get("route") == DOCUMENT else "prior_answer"


def _after_answerability(state: AgentState) -> str:
    # #70: the critic node already compared its 0–100 score to the injected
    # threshold and wrote ``answerable``; the router stays pure. Below threshold
    # -> graceful ``no_expert`` terminal instead of handing off a weak match.
    return "c7_draft" if state.get("answerable") else "no_expert"


def _after_send(state: AgentState) -> str:
    # The outcome is external input (Command(resume=...)); validate it. Only the
    # known values proceed — anything else (None, a typo, a payload) loops back to
    # ``send`` to re-confirm, so a bad value never silently reaches the success
    # terminal (c8_update).
    outcome = state.get("outcome")
    if outcome == "declined":
        return "reroute"
    if outcome == "accepted":
        return "c8_update"
    if outcome == "redraft":
        # Asker asked to regenerate the hand-off text ("下書きの作り直し", #260): go
        # back to C7 for the SAME top candidate, then ``c7_draft`` pauses at ``send``
        # again. The regeneration rides the SSE thought-process like any C7 run.
        return "c7_draft"
    return "send"


def build_agent(
    embedder: Embedder,
    session: Any,
    *,
    intent_model: IntentModel | None = None,
    sufficiency_model: SufficiencyModel | None = None,
    draft_model: DraftModel | None = None,
    answerability_model: AnswerabilityModel | None = None,
    answerability_threshold: int = 40,
    self_answer_model: SelfAnswerModel | None = None,
    retriever: Retriever | None = None,
    scorer: ExpertiseScorer | None = None,
    weights: Weights = DEFAULT_WEIGHTS,
    checkpointer: Any | None = None,
    retriever_top_k: int = 10,
    rrf_k: int = 60,
    bm25_weight: float | None = None,
    prior_answer_reuse_min: int | None = None,
    prior_answer_relevance_floor: float = 0.15,
    knowledge_answer_min_similarity: float | None = None,
    query_expansion_enabled: bool = False,
    question_fit_enabled: bool = False,
    additive_self_answer_enabled: bool = False,
    additive_self_answer_floor: float = 0.20,
    score_all_employees: bool = False,
    similar_askers_enabled: bool = False,
    source_search: SourceSearch | None = None,
    expert_finder: ExpertFinder | None = None,
    knowledge_search: KnowledgeSearch | None = None,
):
    """Compile and return the C1-C8 agent graph.

    Args:
        embedder: Query embedder (C3); FakeEmbedder in tests.
        session: SQLAlchemy session for the default retriever/scorer.
        intent_model / sufficiency_model / draft_model: LLM-node stubs; default to
            the deterministic keyword/rule/template implementations.
        retriever: C4 component with ``.search(query) -> dict``; default
            ``HybridRetriever``.
        scorer: C6 scorer; default ``ExpertiseScorer`` over ``session``.
        answerability_model: optional #70 evidence-sufficiency critic. ``None``
            (default) compiles the pre-#70 graph unchanged — no critique node is
            added. When supplied, a critique node runs between C6 and C7 and
            rejects below ``answerability_threshold`` to a ``no_expert`` terminal.
        self_answer_model: optional #291 self-answer composer. ``None`` (default)
            leaves the data-derived routes unchanged. When supplied, the document /
            prior_answer routes first try a cited answer from the retrieved
            evidence; a grounded answer terminates at ``self_answered``, otherwise
            the run falls back to the original route (document terminal / hand-off).
        score_all_employees: #87, MEASURED AND REJECTED (ADR-0009) — kept only so the
            measurement can be reproduced. ``False`` (default) scores only the people
            C4's top chunks surfaced. ``True`` hands C6 the whole roster instead; the
            C5 route signal keeps reading C4's set either way, but note that with the
            whole roster C6 essentially always finds someone, so the ``no_candidate``
            terminal (「適任者が見つかりませんでした」) becomes unreachable in practice.
        source_search: #81, what a question carrying an ``org_id`` searches instead
            of ``retriever``.
        expert_finder: #107, who a member's question is handed to, found from the
            evidence C4 searched. ``None`` leaves a member's question with nobody.
        knowledge_answer_min_similarity: #118, the similarity a verified knowledge
            article must reach to be offered to the self-answer composer next to the
            sources. ``None`` (default) offers no knowledge. Knowledge is searched
            only when ``self_answer_model`` is wired, since nothing else reads it.
        knowledge_search: what finds that knowledge; defaults to
            ``KnowledgeCandidates`` over ``session`` with the floor above.
        checkpointer: LangGraph checkpointer; default ``MemorySaver``.
    """

    nodes = AgentNodes(
        intent_model=intent_model or KeywordIntentModel(),
        sufficiency_model=sufficiency_model or RuleSufficiencyModel(),
        draft_model=draft_model or TemplateDraftModel(),
        embedder=embedder,
        retriever=retriever
        or HybridRetriever(
            embedder,
            session,
            top_k=retriever_top_k,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
        ),
        scorer=scorer or ExpertiseScorer(Repository(session), weights=weights),
        answerability_model=answerability_model,
        answerability_threshold=answerability_threshold,
        self_answer_model=self_answer_model,
        # The composer re-hydrates evidence text from ids via the repository's batch
        # lookups (#69); only needed when self-answer is wired.
        fragment_source=Repository(session) if self_answer_model is not None else None,
        prior_answer_reuse_min=prior_answer_reuse_min,
        prior_answer_relevance_floor=prior_answer_relevance_floor,
        knowledge_search=_knowledge_search(
            session, knowledge_answer_min_similarity, knowledge_search, self_answer_model
        ),
        # #475 Screen 01: c1_intent counts other prior askers in the same topic area
        # only when this session is wired (None = OFF -> understood byte-identical).
        similar_asker_session=session if similar_askers_enabled else None,
        # #371: fold C1 topics into the C4 retrieval query (False = OFF, dormant).
        query_expansion_enabled=query_expansion_enabled,
        # #405: add the question↔past-answer term to C6 (False = OFF, dormant).
        question_fit_enabled=question_fit_enabled,
        # #413: additive cited answer on the person route (False = OFF, dormant).
        additive_self_answer_enabled=additive_self_answer_enabled,
        additive_self_answer_floor=additive_self_answer_floor,
        # #87: score the whole roster in C6 (False = OFF -> C4's candidate set).
        employee_source=Repository(session) if score_all_employees else None,
        source_search=source_search,
        expert_finder=expert_finder,
    )
    # #70: the critic is wired only when a model is supplied. Off (the default) the
    # graph is byte-for-byte the pre-#70 flow — C6 -> C7 directly.
    critique_wired = answerability_model is not None
    # #291: self-answer is wired only when a composer is supplied. Off (default) the
    # data-derived routes keep their pre-#291 behaviour (document terminal / pinned
    # hand-off) untouched.
    self_answer_wired = self_answer_model is not None

    graph = StateGraph(AgentState)
    graph.add_node("reset", nodes.reset)
    graph.add_node("c1_intent", nodes.c1_intent)
    graph.add_node("c2_sufficiency", nodes.c2_sufficiency)
    graph.add_node("ask", nodes.ask)
    graph.add_node("c3_embed", nodes.c3_embed)
    graph.add_node("c4_retrieve", nodes.c4_retrieve)
    graph.add_node("c5_route", nodes.c5_route)
    graph.add_node("prior_answer", nodes.prior_answer)
    graph.add_node("c6_score", nodes.c6_score)
    graph.add_node("c7_draft", nodes.c7_draft)
    graph.add_node("send", nodes.send)
    graph.add_node("reroute", nodes.reroute)
    graph.add_node("c8_update", nodes.c8_update)
    graph.add_node("off_topic", nodes.out_of_scope)
    graph.add_node("document", nodes.document)
    graph.add_node("no_candidate", nodes.no_candidate)
    graph.add_node("unresolved_intent", nodes.unresolved_intent)
    if critique_wired:
        graph.add_node("answerability", nodes.answerability)
        graph.add_node("no_expert", nodes.no_expert)
    if self_answer_wired:
        graph.add_node("self_answer", nodes.self_answer)
    if self_answer_wired:
        graph.add_node("self_answered", nodes.self_answered)

    # START -> reset -> C1. ``reset`` clears per-question control fields on a fresh
    # invoke; ``resume`` bypasses START, so mid-flow interrupts keep their state.
    graph.add_edge(START, "reset")
    graph.add_edge("reset", "c1_intent")
    graph.add_conditional_edges(
        "c1_intent",
        _after_c1,
        {"off_topic": "off_topic", "c2_sufficiency": "c2_sufficiency"},
    )
    graph.add_conditional_edges(
        "c2_sufficiency",
        _after_c2,
        {"c3_embed": "c3_embed", "ask": "ask", "unresolved_intent": "unresolved_intent"},
    )
    graph.add_edge("ask", "c1_intent")  # re-understand the enriched question
    # #118: verified knowledge no longer answers before C4; C4 offers it to the
    # self-answer composer next to the sources.
    graph.add_edge("c3_embed", "c4_retrieve")
    graph.add_edge("c4_retrieve", "c5_route")
    # #291: when self-answer is wired, the DATA-DERIVED routes (document /
    # prior_answer) try a cited self-answer first; the PERSON route (weak data)
    # goes straight to the hand-off as before. Off, the mapping is the pre-#291 one.
    data_route_target = "self_answer" if self_answer_wired else None
    # #413: additive cited answer on the person route. Only when self_answer is
    # wired (shares the composer) AND enabled. PERSON then flows through the
    # ``additive_answer`` node, which composes a grounded citation to show ALONGSIDE
    # the hand-off and ALWAYS continues to c6_score (never terminates, never steals
    # the route). Off, PERSON goes straight to c6_score exactly as before.
    additive_wired = self_answer_wired and additive_self_answer_enabled
    person_target = "additive_answer" if additive_wired else "c6_score"
    if additive_wired:
        graph.add_node("additive_answer", nodes.additive_answer)
        graph.add_edge("additive_answer", "c6_score")
    graph.add_conditional_edges(
        "c5_route",
        _after_c5,
        # #279: the document route now also runs C6 to rank a person fallback; the
        # DOCUMENT terminal presents the document AND those candidates.
        {
            PERSON: person_target,
            PRIOR_ANSWER: data_route_target or "prior_answer",
            DOCUMENT: data_route_target or "c6_score",
        },
    )
    if self_answer_wired:
        graph.add_conditional_edges(
            "self_answer",
            _after_self_answer,
            # grounded -> answer; else fall back to the route we came from.
            {
                "self_answered": "self_answered",
                "c6_score": "c6_score",
                "prior_answer": "prior_answer",
            },
        )
    if self_answer_wired:
        graph.add_edge("self_answered", END)
    graph.add_edge("prior_answer", "c6_score")
    # #70: when the critic is wired, the hand-off branch first passes through the
    # answerability node; _after_c6 is unchanged (still returns "c7_draft"), only
    # its target is redirected. The document/no_candidate terminals are untouched.
    handoff_target = "answerability" if critique_wired else "c7_draft"
    graph.add_conditional_edges(
        "c6_score",
        _after_c6,
        {"c7_draft": handoff_target, "no_candidate": "no_candidate", "document": "document"},
    )
    if critique_wired:
        graph.add_conditional_edges(
            "answerability",
            _after_answerability,
            {"c7_draft": "c7_draft", "no_expert": "no_expert"},
        )
        graph.add_edge("no_expert", END)
    graph.add_edge("c7_draft", "send")
    graph.add_conditional_edges(
        "send",
        _after_send,
        {
            "reroute": "reroute",
            "c8_update": "c8_update",
            "c7_draft": "c7_draft",  # redraft loop (#260): regenerate then re-pause at send
            "send": "send",
        },
    )
    graph.add_edge("reroute", "c6_score")
    graph.add_edge("c8_update", END)
    graph.add_edge("off_topic", END)
    graph.add_edge("document", END)
    graph.add_edge("no_candidate", END)
    graph.add_edge("unresolved_intent", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())

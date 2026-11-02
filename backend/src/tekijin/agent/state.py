"""Shared graph state for the C1-C8 agent flow.

``AgentState`` is the single ``TypedDict`` threaded through the LangGraph
``StateGraph``. Every node returns a *partial* update (a dict with only the keys
it changed); LangGraph merges those into the running state. Keys are grouped by
the component that writes them (C1 intent, C2 sufficiency, … C8 update) plus a
few control keys (``now``, ``answer``, decline loop).

Nothing here reads the clock or the network — ``now`` is injected by the caller
so the whole run is reproducible.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal, NotRequired, TypedDict

# C5 route labels and responder outcomes, as closed sets so a typo is a type
# error and the graph routers can validate against them.
Route = Literal["person", "prior_answer", "document"]
Outcome = Literal["accepted", "declined"]


class PastAnswer(TypedDict):
    """One past-answer hit from C4."""

    qa_id: str
    score: float
    responder_id: int | None  # None if the answer's responder is unknown
    # #327: how often this answer was reused — the corpus-count signal C5 uses to
    # route prior_answer (cosine cannot separate it under Nemotron). 0 when unknown.
    reuse_count: int


class DocumentHit(TypedDict):
    """One document hit from C4."""

    doc_id: str
    score: float


class DailyHit(TypedDict):
    """One daily-report hit from C4 (#433). A daily report as a knowledge source."""

    daily_id: int
    score: float


class SourceHit(TypedDict):
    """One chunk of something the organization ingested (#81)."""

    chunk_id: int
    source_item_id: int
    score: float


class RetrievalResult(TypedDict):
    """The C4 (HybridRetriever) output shape shared by C5/C6.

    The ``*_score`` values inside ``past_answers`` / ``documents`` are RRF fusion
    scores — good for *ranking* within a channel, but they carry no absolute
    meaning across queries. So each channel also reports its top absolute dense
    cosine similarity (``*_confidence``, in ``[0, 1]``, ``0.0`` when the channel is
    empty); C5 routes on those, not on the RRF scores.
    """

    past_answers: list[PastAnswer]
    documents: list[DocumentHit]
    candidate_people: list[int]
    # Responders of `past_answers` who have left (#506). Separate from
    # `candidate_people` because the prior_answer pin deliberately reaches
    # OUTSIDE that pool — its whole point is to hand off to whoever answered
    # before, even when they did not rank. So the pin needs its own signal for
    # "this person is gone", not pool membership.
    departed_people: NotRequired[list[int]]
    answer_confidence: float
    document_confidence: float
    people_confidence: float
    # #433: daily-report hits — a searchable knowledge source for System 1. Always
    # present but EMPTY unless ``daily_knowledge_enabled`` (develop byte-identical
    # when off). self-answer / #413 additive cite these alongside answers/documents.
    daily_reports: list[DailyHit]
    # #405: per-person max cosine of the QUESTION against that person's past
    # answers (from the answer dense channel), keyed by responder_id. C6 adds it as
    # a question-fit term when ``question_fit_enabled``; absent people default to
    # 0.0. Always populated by C4 (cheap — reuses the answer dense hits), harmless
    # when the feature is off.
    person_question_similarity: dict[int, float]
    # #81: the asker's organization's own sources. A member's question fills only
    # these; ``document_confidence`` then carries ``source_confidence`` so C5 routes
    # as it does for documents.
    sources: NotRequired[list[SourceHit]]
    source_confidence: NotRequired[float]


def empty_retrieval() -> RetrievalResult:
    """A fresh, empty :class:`RetrievalResult` (the default before C4 runs)."""

    return {
        "past_answers": [],
        "documents": [],
        "candidate_people": [],
        "departed_people": [],
        "answer_confidence": 0.0,
        "document_confidence": 0.0,
        "people_confidence": 0.0,
        "person_question_similarity": {},
        "daily_reports": [],
        "sources": [],
        "source_confidence": 0.0,
    }


class AgentState(TypedDict, total=False):
    """The full state of one question's journey through the graph."""

    # -- input -------------------------------------------------------------
    question: str
    asker: dict[str, Any] | None
    now: dt.datetime  # naive; the scorer requires it
    # The asker's organization (#81). ``None`` for an account without one, which
    # still reads the older, undivided tables.
    org_id: int | None

    # -- durable persistence identity (set by the API layer) ---------------
    # These live in the checkpoint so that outcome recording survives a
    # process restart / eviction and does not depend on the volatile
    # in-memory session registry.
    question_id: str | None  # uuid4 assigned at POST /ask
    recommendation_ids: list[int]  # DB ids of every *shown* recommendation
    primary_recommendation_id: int | None  # the one actually handed off
    # The last terminal SSE event ({"event", "data"}) of a completed run, kept so
    # a client that disconnected before receiving ``done`` / terminal ``message``
    # can re-fetch it on reconnect (EventSource retry) — read-only replay.
    last_event: dict[str, Any] | None

    # -- C1 intent ---------------------------------------------------------
    topics: list[str]
    products: list[str]
    situation: str | None
    question_type: str
    out_of_scope: bool
    intent_confidence: float
    # #83: the branch the asker explicitly asked the responder to be at (None = no
    # constraint). C6 treats it as a condition to satisfy, not a scoring term.
    constraint_branch: str | None
    # #475 Screen 01: how many OTHER askers have previously asked a question in the
    # same topic area (reassurance — "you are not the only one"). Set by c1_intent
    # only when a read session is wired (``similar_askers_enabled``); absent/0
    # otherwise, so the understood event is byte-identical when the feature is off.
    similar_asker_count: int

    # -- C2 sufficiency ----------------------------------------------------
    sufficient: bool
    missing: list[str]
    followup_question: str | None
    followup_count: int  # how many times we have asked back (cap = 1)
    intent_unresolved: bool  # capped but still no topic -> graceful terminal

    # -- C3 embedding ------------------------------------------------------
    query_vector: list[float]

    # -- C4 retrieval ------------------------------------------------------
    retrieval: RetrievalResult

    # -- C5 route ----------------------------------------------------------
    route: Route
    route_reason: str
    route_confidence: float
    prior_answer_note: str | None  # set on the prior_answer (補助) route
    pinned_responder_id: int | None  # prior_answer: hand off to THIS person
    # #518: the responder said "他に適任者がいる" and named THIS person. Set by the
    # refer endpoint just before a reroute; c6_score seats them at rank 1 (over the
    # general pool), so the hand-off moves to the named person instead of the
    # scorer's next pick. None (the default) leaves c6 byte-identical.
    referred_responder_id: int | None

    # -- self-answer (#291) ------------------------------------------------
    # Set by the self_answer node (only when a composer is wired) on the
    # data-derived routes. ``grounded`` True -> the assistant answered directly
    # from retrieved sources and the graph terminates at ``self_answered``; False
    # -> fall back to the original route (document / person hand-off). ``text`` is
    # the composed answer, ``citations`` the sources it used ({source_id, kind})
    # so the chat can render links back to each.
    self_answer_grounded: bool
    self_answer_text: str | None
    self_answer_citations: list[dict[str, str]]

    # -- additive self-answer (#413) --------------------------------------
    # Set by the additive_answer node on the PERSON route (only when
    # ``additive_self_answer_enabled``). Unlike ``self_answer_*`` above, this NEVER
    # terminates the run or marks it self-resolved — the human hand-off still
    # happens. When a grounded cited answer exists it is surfaced ALONGSIDE the
    # recommendation ("参考: 過去の類似回答") so System 1 fires on person-routed
    # knowledge questions too, without stealing the route (person recall unchanged).
    additive_answer_text: str | None
    additive_citations: list[dict[str, str]]

    # -- C6 scoring --------------------------------------------------------
    recommendations: list[dict[str, Any]]

    # -- answerability critic (#70) ---------------------------------------
    # Evidence-sufficiency verdict on the ranked candidates, run BETWEEN C6 and
    # C7 (only when a critic is wired). ``answerable`` is the node's own
    # threshold decision (kept in state so the graph router stays pure); on
    # False the graph rejects to the ``no_expert`` terminal instead of handing
    # off a weak match. Absent (all three) when no critic is wired.
    answerability_confidence: int  # 0–100
    answerability_reason: str | None
    answerable: bool

    # -- C7 draft ----------------------------------------------------------
    draft: str | None

    # -- decline loop / C8 -------------------------------------------------
    outcome: Outcome | None  # from the responder (validated at the send router)
    declined_ids: list[int]

    # -- terminal output ---------------------------------------------------
    answer: str | None
    document_id: str | None  # document route: the cited doc, surfaced to the client
    # Document route only: the already-ranked person offered when the document
    # does not solve the question. Structured separately from the prose so the
    # client can render an actionable hand-off button (#351).
    fallback_responder: dict[str, Any] | None
    document_fallback_requested: bool

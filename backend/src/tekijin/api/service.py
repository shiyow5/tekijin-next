"""Application service that drives the C1-C8 agent behind the HTTP API.

One :class:`AgentService` is created per process and shared across requests. It
holds the long-lived pieces (checkpointer, embedder, LLM nodes, sessionmaker) and
builds a *fresh* graph per stream (bound to a fresh DB session), so no SQLAlchemy
session is shared across requests. Interrupt/resume is carried by the shared
checkpointer keyed on ``thread_id == session_id`` — a graph rebuilt for a later
request resumes the earlier paused state.

Durability & concurrency (the core of this design):

* **Persistence identity lives in the durable checkpoint, not memory.** The
  ``question_id`` (a ``uuid4``) rides in the ``AgentState`` from ``/ask``; the DB
  ids of the shown recommendations — and the ``primary_recommendation_id`` whose
  outcome is later recorded — are written back into the state with
  ``graph.update_state`` after C6. So ``/answer`` records the outcome by reading
  ``graph.get_state(config).values`` and survives an eviction / restart. As a
  belt-and-suspenders fallback (e.g. a disconnect before ``update_state`` ran),
  the outcome target is re-derivable from the DB by ``question_id``.
* **A per-session lock makes accept, resume and stream mutually exclusive.** The
  in-memory registry shrinks to just "the next input to run" plus a TTL stamp;
  every state-transition (accept / resume / consume-and-stream) runs under the
  session's lock, so concurrent ``/ask`` cannot double-insert, concurrent
  ``/answer`` cannot drop an outcome, and concurrent ``/events`` cannot start the
  graph twice (double recommendation rows).
* **Eviction never drops a paused session.** ``_sweep`` skips any session that is
  mid-interrupt (a human is being waited on), so long HITL pauses are safe.

This still requires the API to run SINGLE worker (the lock + registry are
in-process); that is documented on the Makefile ``serve-prod`` target.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from langgraph.types import Command
from sqlalchemy.orm import Session, sessionmaker
from sse_starlette import ServerSentEvent

from tekijin.agent.graph import build_agent
from tekijin.agent.nodes import draft_context
from tekijin.agent.protocols import (
    AnswerabilityModel,
    DraftModel,
    IntentModel,
    QuestionStructurer,
    SelfAnswerModel,
    SufficiencyModel,
)
from tekijin.api import schemas
from tekijin.api.events import (
    TERMINAL_EVENTS,
    TERMINAL_NODES,
    interrupt_event,
    node_event,
    progress_events,
    reconnect_events,
    replay_terminal,
)
from tekijin.channels.notifier import get_notifier
from tekijin.data.db import session_scope
from tekijin.data.feedback import record_feedback
from tekijin.data.handoff import employee_brief, question_consult_method, responder_reuse_stats
from tekijin.data.messages import create_message, thread_parties
from tekijin.data.writes import (
    create_answer,
    detach_question_session,
    employee_exists,
    insert_shown_recommendations,
    latest_primary_recommendation,
    latest_recommendation_for_person,
    mark_question_resolved,
    mark_self_resolved,
    persist_question,
    recommendation_employee_id,
    recommendation_outcome,
    record_events,
    reopen_question_for_handoff,
    reorder_recommendation_ranks,
    set_recommendation_outcome,
    shown_recommendation_ids,
    update_question_body,
    update_question_consult_method,
    update_question_route,
    update_question_topics,
)
from tekijin.knowledge.index import embed_knowledge_units
from tekijin.retrieval.embedding import PASSAGE, Embedder

logger = logging.getLogger(__name__)

# In-memory session entries older than this (no activity) are evicted so an /ask
# that is never followed by /events cannot leak. Paused (mid-interrupt) sessions
# are exempt (see ``_sweep``).
SESSION_TTL_SECONDS = 3600.0

# The nodes the graph pauses *before* when it interrupts (HITL waits).
_INTERRUPT_NODES: frozenset[str] = frozenset({"ask", "send"})


class SessionConflict(Exception):
    """The request conflicts with the session's state (maps to HTTP 409)."""


class SessionInvalid(Exception):
    """The resume kind does not match the pending interrupt (maps to HTTP 422)."""


class AskerNotFound(Exception):
    """The ``asker_id`` is not a known employee (maps to HTTP 404)."""


class HandoffNotFound(Exception):
    """No responder handoff is available for this session (maps to HTTP 404)."""


class ServiceBusy(Exception):
    """Too many runs are executing on the (single-GPU) LLM (maps to HTTP 503).

    Backpressure (#180): the app sheds NEW questions when at capacity rather than
    piling more concurrent graph runs onto vLLM and risking OOM / a host stall. The
    hard per-GPU bound is still vLLM's own ``max-num-seqs``; this is the app-level
    admission gate so callers get a fast, graceful "混雑中" instead of a stuck run.
    """


def _default_now() -> dt.datetime:
    # Naive (no tzinfo) to match stored ``created_at`` and the scorer's contract.
    return dt.datetime.now()  # noqa: DTZ005 - naive is intentional


def _interrupt_payload(value: Any) -> dict[str, Any]:
    # stream yields {"__interrupt__": (Interrupt(...),)}; take the first payload.
    if value and isinstance(value, tuple):
        return getattr(value[0], "value", {}) or {}
    return {}


def _segment_latency_ms(
    event_rows: list[tuple[str, dt.datetime, dt.datetime, dict[str, Any] | None]],
) -> int:
    """Total processing time (ms) of the recorded stages in this run segment (#177)."""

    total = sum((ended - started).total_seconds() for _stage, started, ended, _meta in event_rows)
    return round(total * 1000)


def _error_event() -> ServerSentEvent:
    # Generic client-facing error — never leaks exception / SQL / model details.
    return ServerSentEvent(
        event="error",
        data=schemas.ErrorData(error="内部エラーが発生しました").model_dump_json(),
    )


@dataclass
class _SessionCtx:
    """Per-session in-memory bookkeeping: just the input queue + a TTL stamp.

    Persistence identity (question_id, recommendation ids) is NOT here — it lives
    in the durable checkpoint state, so it survives eviction of this object.
    """

    pending: Any = None  # next input to stream (AgentState or Command), or None
    touched_at: float = field(default_factory=time.monotonic)


class AgentService:
    """Builds graphs, tracks per-session dispatch state, and streams SSE events."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        checkpointer: Any,
        embedder: Embedder,
        intent_model: IntentModel,
        sufficiency_model: SufficiencyModel,
        draft_model: DraftModel,
        question_structurer: QuestionStructurer | None = None,
        answerability_model: AnswerabilityModel | None = None,
        answerability_threshold: int = 40,
        self_answer_model: SelfAnswerModel | None = None,
        retriever: Any | None = None,
        scorer: Any | None = None,
        bm25_weight: float | None = None,
        prior_answer_reuse_min: int | None = None,
        prior_answer_relevance_floor: float = 0.15,
        daily_evidence: bool = False,
        daily_knowledge_enabled: bool = False,
        knowledge_answer_min_similarity: float | None = None,
        query_expansion_enabled: bool = False,
        question_fit_enabled: bool = False,
        branch_constraint_enabled: bool = False,
        additive_self_answer_enabled: bool = False,
        additive_self_answer_floor: float = 0.20,
        score_all_employees: bool = False,
        similar_askers_enabled: bool = False,
        max_concurrent_runs: int = 0,
        now_factory: Any = _default_now,
        clock: Any = time.monotonic,
    ) -> None:
        self._session_factory = session_factory
        self._checkpointer = checkpointer
        self._embedder = embedder
        self._intent = intent_model
        self._sufficiency = sufficiency_model
        self._draft = draft_model
        # #475 Screen 01: on-demand question re-drafter. Runs OUTSIDE the graph via
        # ``structure_question`` (result-screen endpoint), never on the C1 critical
        # path. None -> the feature is unavailable (503); the factory always supplies
        # one (stub or vLLM), so it is only None in tests that don't exercise it.
        self._structurer = question_structurer
        # #70: evidence-sufficiency critic. None (default) compiles the pre-#70
        # graph (no critique node); when set, the SSE/persist layer defers the
        # recommend event + rows until the critic accepts (see ``_run``), so a
        # rejected set never becomes a phantom pending row (mirrors #281).
        self._answerability = answerability_model
        self._answerability_threshold = answerability_threshold
        # #291: self-answer composer; None (default) compiles the pre-#291 graph.
        self._self_answer = self_answer_model
        # #357 slice 4c: knowledge-answer similarity floor; None (default) compiles
        # the pre-#357 graph (no knowledge_answer node). Passed to build_agent.
        self._knowledge_answer_min_similarity = knowledge_answer_min_similarity
        # Optional C4/C6 overrides — default (None) uses the real HybridRetriever
        # / ExpertiseScorer over the request session; tests inject deterministic
        # fakes so the SSE flow does not depend on retrieval scores.
        self._retriever = retriever
        self._scorer = scorer
        # C4 BM25 fusion weight from the SUPPLIED settings (via the factory), so a
        # custom Settings is honored rather than the cached global (#68). None →
        # the default HybridRetriever reads settings itself.
        self._bm25_weight = bm25_weight
        # #327: corpus-count routing for prior_answer (None = OFF). Passed to
        # build_agent so C5 can revive the route on reuse_count when calibrated.
        self._prior_answer_reuse_min = prior_answer_reuse_min
        self._prior_answer_relevance_floor = prior_answer_relevance_floor
        # #355: include daily reports as C6 evidence. Passed to build_agent's default
        # scorer (None = dormant). Ignored when a scorer is injected (tests).
        self._daily_evidence = daily_evidence
        # #433: daily reports as a System 1 knowledge source (False = OFF, dormant).
        self._daily_knowledge_enabled = daily_knowledge_enabled
        # #371: fold C1 topics into the C4 retrieval query (False = OFF, dormant).
        self._query_expansion_enabled = query_expansion_enabled
        # #405: add the question↔past-answer term to C6 (False = OFF, dormant).
        self._question_fit_enabled = question_fit_enabled
        self._branch_constraint_enabled = branch_constraint_enabled
        # #413: additive cited answer on the person route (False = OFF, dormant).
        self._additive_self_answer_enabled = additive_self_answer_enabled
        self._additive_self_answer_floor = additive_self_answer_floor
        self._score_all_employees = score_all_employees
        # #475 Screen 01: attach the "N other askers in this area" reassurance count
        # to understood (False = OFF -> understood byte-identical). Read-only display.
        self._similar_askers_enabled = similar_askers_enabled
        # Backpressure (#180): max graph runs executing at once before /ask sheds new
        # questions with 503. 0 (default) disables it — set from settings via the
        # factory in production. Guarded by its own lock; independent of the per-
        # session locks so admission is a cheap global check.
        self._max_concurrent_runs = max_concurrent_runs
        self._active_runs = 0
        self._runs_lock = threading.Lock()
        self._now_factory = now_factory
        # Monotonic clock for TTL bookkeeping — injectable so ``_sweep`` is
        # deterministic in tests (the process ``time.monotonic`` epoch is the boot
        # time, which is arbitrary and, on a fresh CI runner, small; see #0).
        self._clock = clock
        self._registry: dict[str, _SessionCtx] = {}
        self._registry_guard = threading.Lock()  # guards _registry membership only
        # One lock per session, handed out under a guard lock. Serialises accept /
        # resume / stream for the same session (see the module docstring).
        self._locks_guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    # -- session access for the dashboard --------------------------------- #
    @property
    def session_factory(self) -> sessionmaker[Session]:
        return self._session_factory

    def now(self) -> dt.datetime:
        """The service's clock (injected in tests) — for route-level timestamps."""

        return self._now_factory()

    def close(self) -> None:
        """Release long-lived resources at shutdown (checkpointer pool, engine)."""

        pool = getattr(self._checkpointer, "conn", None)  # PostgresSaver holds a pool
        closer = getattr(pool, "close", None)
        if callable(closer):
            closer()
        engine = self._session_factory.kw.get("bind")
        if engine is not None:
            engine.dispose()

    # -- backpressure (#180) ---------------------------------------------- #
    def _reject_if_saturated(self) -> None:
        """Raise :class:`ServiceBusy` when the LLM run pool is full (admission gate)."""

        if self._max_concurrent_runs <= 0:
            return
        with self._runs_lock:
            if self._active_runs >= self._max_concurrent_runs:
                raise ServiceBusy(
                    f"{self._active_runs} runs already executing "
                    f"(max {self._max_concurrent_runs}); try again shortly"
                )

    @contextlib.contextmanager
    def _run_slot(self) -> Iterator[None]:
        """Count one in-flight graph run for the duration of its execution.

        Held only while ``graph.stream`` actually runs (LLM/GPU work), so a run that
        pauses at an interrupt or finishes releases its slot. The ``finally`` runs on
        normal completion, on an exception, and when the generator is closed
        (``GeneratorExit``). On a client DISCONNECT the release therefore depends on
        the ASGI layer finalizing this generator — the same generator finalization the
        per-session lock and ``session.close()`` already rely on; CPython does it
        promptly via refcounting, but it is not a hard contract. Only NEW questions are
        shed (see ``start_question``); resumes / continuations occupy a slot but are
        never rejected.
        """

        with self._runs_lock:
            self._active_runs += 1
        try:
            yield
        finally:
            with self._runs_lock:
                self._active_runs -= 1

    # -- locking ---------------------------------------------------------- #
    def _lock(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, threading.Lock())

    # -- registry (membership changes go through these, under the guard) --- #
    def _reg_get(self, session_id: str) -> _SessionCtx | None:
        with self._registry_guard:
            return self._registry.get(session_id)

    def _reg_ensure(self, session_id: str) -> _SessionCtx:
        with self._registry_guard:
            return self._registry.setdefault(session_id, _SessionCtx())

    # -- TTL sweep -------------------------------------------------------- #
    def _sweep(self) -> None:
        """Evict idle sessions (and their locks); never a mid-interrupt one.

        Runs opportunistically on ``/ask``. Membership changes are made under the
        registry guard; a candidate is only evicted while its per-session lock is
        held (non-blocking), so it cannot race a live dispatch on that session.
        """

        cutoff = self._clock() - SESSION_TTL_SECONDS
        with self._registry_guard:
            candidates = [sid for sid, ctx in self._registry.items() if ctx.touched_at < cutoff]
        for sid in candidates:
            # A human is being waited on (clarification / outcome): keep it.
            if self._next_nodes(sid):
                continue
            lock = self._lock(sid)
            if not lock.acquire(blocking=False):
                continue  # a dispatch is touching this session now — next sweep
            try:
                with self._registry_guard:
                    ctx = self._registry.get(sid)
                    if ctx is None or ctx.touched_at >= cutoff:
                        continue  # revived since the scan above
                    del self._registry[sid]
                with self._locks_guard:  # GC the lock so ids don't leak (codex#2)
                    self._locks.pop(sid, None)
            finally:
                lock.release()

    def _config(self, session_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": session_id}}

    def _snapshot(self, session_id: str) -> Any:
        """Durable state snapshot (``.next`` + ``.values``) for this session."""

        session = self._session_factory()
        try:
            return self._graph(session).get_state(self._config(session_id))
        finally:
            session.close()

    def _next_nodes(self, session_id: str) -> tuple[str, ...]:
        """Durable interrupt state: the nodes the graph is paused before (if any)."""

        return tuple(self._snapshot(session_id).next)

    def session_participants(self, session_id: str) -> tuple[int | None, int | None]:
        """``(asker_id, responder_id)`` for a session, from the durable state (#241).

        Object-level authorization for the session-scoped endpoints (``/handoff``,
        ``/answer``, ``/events``): a non-admin may only touch a session they are a
        participant of. The asker is set at ``/ask``; the responder is the current
        primary recommendation (it changes on a decline/reroute, so a rerouted-away
        responder is no longer a participant). Both are ``None`` for an unknown
        session — the caller then falls through to the handler's normal 404 rather
        than turning a missing session into a 403.
        """

        snapshot = self._snapshot(session_id)
        values = getattr(snapshot, "values", None) or {}
        asker_id = (values.get("asker") or {}).get("id")
        recs = values.get("recommendations") or []
        responder_id = recs[0]["person_id"] if recs else None
        return (asker_id, responder_id)

    def pending_handoff_metadata(self, session_id: str) -> dict[str, int | str | None]:
        """Return durable identifiers needed to notify an external hand-off client."""
        snapshot = self._snapshot(session_id)
        values = getattr(snapshot, "values", None) or {}
        asker_id = (values.get("asker") or {}).get("id")
        recs = values.get("recommendations") or []
        primary = values.get("primary_recommendation_id")
        if primary is None and recs:
            primary = recs[0].get("recommendation_id")
        return {
            "question_id": values.get("question_id"),
            "asker_id": asker_id,
            "responder_id": recs[0].get("person_id") if recs else None,
            "recommendation_id": primary,
        }

    def request_document_fallback(self, session_id: str) -> None:
        """Reopen a completed document route at C7 using its ranked candidate.

        C6 already ranked the fallback while producing the document message. The
        explicit asker action converts that dormant result into a real hand-off;
        recommendation rows are deliberately created later, as C7 streams.
        """

        with self._lock(session_id):
            session = self._session_factory()
            try:
                graph = self._graph(session)
                config = self._config(session_id)
                snapshot = graph.get_state(config)
                values = snapshot.values or {}
                if snapshot.next:
                    raise SessionConflict("session is already awaiting a response")
                if values.get("route") != "document" or not values.get("question_id"):
                    raise HandoffNotFound("no document fallback for this session")
                if values.get("self_answer_grounded"):
                    raise HandoffNotFound("the session was already answered")
                if not values.get("recommendations"):
                    raise SessionInvalid("no fallback responder is available")
                if not values.get("fallback_responder"):
                    raise SessionInvalid("no fallback responder is available")
                last_event = values.get("last_event") or {}
                if last_event.get("event") != "message":
                    raise HandoffNotFound("no completed document result for this session")

                # Pretend C6 just completed on the person route. LangGraph then
                # schedules the existing C7/answerability path without re-ranking.
                graph.update_state(
                    config,
                    {
                        "route": "person",
                        "route_reason": "文書で解決しなかったため、候補者へ取り次ぎます。",
                        "answer": None,
                        "document_id": None,
                        "fallback_responder": None,
                        "draft": None,
                        "outcome": None,
                        "last_event": None,
                        "document_fallback_requested": True,
                    },
                    as_node="c6_score",
                )
                with session_scope(self._session_factory) as write_session:
                    reopen_question_for_handoff(write_session, values["question_id"])
                ctx = self._reg_ensure(session_id)
                ctx.touched_at = self._clock()
            finally:
                session.close()

    # -- /ask : start a new question -------------------------------------- #
    def start_question(self, session_id: str, asker_id: int, question: str) -> None:
        """Queue a NEW question; reject (409) if the session is busy or paused.

        Validates ``asker_id`` exists (404) before persisting, so a bad id is a
        clean boundary error rather than a mid-flush FK ``IntegrityError``.
        """

        with self._lock(session_id):
            self._sweep()
            # Backpressure: shed a NEW question when the LLM run pool is saturated,
            # before persisting anything, so the caller gets a fast 503 (#180). Soft
            # check-then-run gate (may transiently overshoot under bursts); the hard
            # bound is vLLM's max-num-seqs. See Settings.max_concurrent_runs.
            self._reject_if_saturated()
            ctx = self._reg_get(session_id)
            if ctx is not None and ctx.pending is not None:
                raise SessionConflict("a run is already queued for this session")
            if self._next_nodes(session_id):
                raise SessionConflict("session is awaiting a resume; answer it first")

            now = self._now_factory()
            # Collision-free (uuid4) but ``api_``-prefixed so operational tooling
            # and test cleanup can target API-created questions by prefix.
            question_id = f"api_{uuid.uuid4().hex}"
            with session_scope(self._session_factory) as session:
                if not employee_exists(session, asker_id):
                    raise AskerNotFound(f"asker_id {asker_id} is not a known employee")
                persist_question(
                    session, question_id, asker_id, question, now, session_id=session_id
                )
            ctx = self._reg_ensure(session_id)
            ctx.pending = {
                "question": question,
                "asker": {"id": asker_id},
                "now": now,
                "question_id": question_id,
            }
            ctx.touched_at = self._clock()

    # -- /answer : resume a paused run ------------------------------------ #
    def submit_resume(
        self,
        session_id: str,
        *,
        outcome: str | None = None,
        reply: str | None = None,
        recommendation_id: int | None = None,
        answer_body: str | None = None,
    ) -> None:
        """Queue a resume, validating it matches the pending interrupt kind.

        The outcome target (``primary_recommendation_id``) is read from the durable
        state, not from memory — so it is recorded even after an eviction/restart.

        ``recommendation_id`` is an optional generation token: when supplied with an
        ``outcome`` it must equal the session's current primary, else the outcome is
        stale (a reroute moved the hand-off on, or a competing tab) and is rejected
        with :class:`SessionConflict` (409) — so it never binds to a new candidate
        (#94). ``None`` skips the check for older clients / clarification replies.

        ``answer_body`` (only on an ``accepted`` outcome) is the responder's answer
        text; it is persisted as an ``answers`` row so the accumulation loop closes
        (#274). It is captured on a best-effort basis — a failure to embed or store
        it never blocks the hand-off from resuming.
        """

        with self._lock(session_id):
            snapshot = self._snapshot(session_id)
            next_nodes = tuple(snapshot.next)
            if not next_nodes:
                raise SessionConflict("session is not awaiting a resume")
            ctx = self._reg_get(session_id)
            if ctx is not None and ctx.pending is not None:
                # A resume is already queued and not yet streamed (codex#5).
                raise SessionConflict("a resume is already queued for this session")

            node = next_nodes[0]
            if node == "ask":
                if reply is None:
                    raise SessionInvalid("this session expects a clarification 'reply'")
                resume_value: str = reply
            elif node == "send":
                if outcome is None:
                    raise SessionInvalid("this session expects a responder 'outcome'")
                # Stale-outcome guard (#94): if the client echoed the recommendation
                # id it was shown and the session has since moved to a different
                # primary (reroute / competing tab), reject so the outcome cannot
                # bind to the new candidate. The comparison happens inside
                # `_record_outcome`, against the RESOLVED primary (#94-3) — a
                # missing/unresolvable primary can't be validated, so it is let
                # through (degrade to prior behavior).
                # The DB is the source of truth for the outcome. ``_record_outcome``
                # writes it first-wins and returns the EFFECTIVE (persisted) value:
                # on a duplicate submission (e.g. a restart lost the in-memory
                # pending guard and the responder resubmitted) the stored outcome
                # wins. We resume the graph with that effective value so the
                # checkpoint always advances consistently with the DB — never
                # diverging, and never left permanently paused at ``send``.
                status, resume_value = self._record_outcome(
                    session_id,
                    snapshot.values,
                    outcome,
                    expected_recommendation_id=recommendation_id,
                )
                # #274: capture the answer only on a FRESHLY recorded accept (not a
                # duplicate "already"/"no_target"), in its own post-commit transaction
                # so it cannot roll back the accept. answer_body is pre-validated to
                # ride only on an accept, and route-level auth restricts it to the
                # responder (or admin).
                if status == "recorded" and outcome == "accepted" and answer_body:
                    self._capture_answer(snapshot.values, answer_body)
            else:  # pragma: no cover - the graph only ever interrupts at ask/send
                raise SessionConflict("session cannot be resumed from its current state")

            ctx = self._reg_ensure(session_id)
            ctx.pending = Command(resume=resume_value)
            ctx.touched_at = self._clock()

    def _resolve_primary(self, session: Session, values: dict[str, Any]) -> int | None:
        """The recommendation row the hand-off is CURRENTLY bound to (#94-3).

        ``primary_recommendation_id`` is written back into the checkpoint only at
        the END of a stream segment (``_persist_run_state``), while the graph
        checkpoints ``recommendations`` per node. A client disconnect in between —
        decline → ``reroute`` → C6 backfill → rows INSERTed → ``GeneratorExit`` —
        leaves the two disagreeing: ``recommendations[0]`` is the NEW candidate,
        ``primary_recommendation_id`` still points at the DECLINED row. Recording
        the next outcome on that id would attribute the answer to the wrong person
        and then wedge the hand-off (the declined row already carries an outcome,
        so every later submission returns ``"already"`` — a permanent 409-alike).

        So the SHOWN top candidate is the authority: when the stored id does not
        point at them, re-derive their newest row from the DB. Resolving by PERSON
        rather than by rank keeps this independent of the ``rank`` column, which is
        a projection maintained for ``GET /inbox`` (``reorder_recommendation_ranks``
        on reselect) — the outcome must land on the person the responder saw even if
        that projection lags, and "who was shown" is the invariant we actually mean.

        Returns ``None`` when the shown candidate has NO persisted row: attributing
        their answer to some other row would be worse than not recording it (the
        caller logs and resumes with ``no_target``).
        """

        primary = values.get("primary_recommendation_id")
        question_id = values.get("question_id")
        if question_id is None:
            return primary
        recs = values.get("recommendations") or []
        shown_person = recs[0].get("person_id") if recs else None
        if shown_person is None:
            # No shown candidate to cross-check against (e.g. a terminal with no
            # recommendations): keep the pre-existing durable fallback.
            return (
                primary
                if primary is not None
                else latest_primary_recommendation(session, question_id)
            )
        if primary is not None and recommendation_employee_id(session, primary) == shown_person:
            return primary
        rebound = latest_recommendation_for_person(session, question_id, shown_person)
        if rebound is None:
            # The shown candidate has no persisted row at all (e.g. the #351
            # document-fallback continuation reuses a PRIOR batch's ids and so never
            # inserts one for the fallback candidate). Falling back to `primary` here
            # would knowingly bind THIS person's answer to a row belonging to someone
            # else — it would land in `answers.responder_id` (#274) and light the
            # wrong employee's "回答が届きました". Recording nothing is the lesser
            # harm: the caller logs and resumes with `no_target`.
            logger.warning(
                "no recommendation row for the shown candidate (question %s, person %s); "
                "refusing to bind the outcome to %s",
                question_id,
                shown_person,
                primary,
            )
            return None
        if primary is not None and rebound != primary:
            logger.warning(
                "stale primary_recommendation_id %s for question %s; "
                "re-bound to %s (shown person %s)",
                primary,
                question_id,
                rebound,
                shown_person,
            )
        return rebound

    def _record_outcome(
        self,
        session_id: str,
        values: dict[str, Any],
        outcome: str,
        *,
        expected_recommendation_id: int | None = None,
        referred_to: int | None = None,
    ) -> tuple[str, str]:
        """Record the responder outcome on the durable primary recommendation.

        Returns ``(status, effective_outcome)`` where ``effective_outcome`` is the
        value the graph should resume with (the DB is authoritative):

        * ``("recorded", outcome)`` — fresh write of the submitted outcome;
        * ``("already", stored)`` — the primary already carried ``stored`` (a
          duplicate submission); the stored value wins, and the caller resumes the
          graph with it so the checkpoint stays consistent instead of diverging;
        * ``("no_target", outcome)`` — no recommendation to attach it to; resume
          with the submitted value (nothing to reconcile against).

        ``expected_recommendation_id`` is the generation token the client echoed
        (#94-2). It is compared against the RESOLVED primary, not the raw
        checkpoint value, so a stale checkpoint (#94-3) cannot flip the guard both
        ways — accepting the outcome for the superseded candidate and rejecting the
        one the responder is actually looking at.
        """

        question_id = values.get("question_id")
        with session_scope(self._session_factory) as session:
            primary = self._resolve_primary(session, values)
            if (
                expected_recommendation_id is not None
                and primary is not None
                and expected_recommendation_id != primary
            ):
                raise SessionConflict(
                    "this hand-off has moved to a different candidate; reload and try again"
                )
            if primary is None:
                logger.warning(
                    "no recommendation to record outcome for session %s (outcome=%s)",
                    session_id,
                    outcome,
                )
                return "no_target", outcome
            existing = recommendation_outcome(session, primary)
            if existing is not None:
                return "already", existing
            set_recommendation_outcome(session, primary, outcome, referred_to=referred_to)
            # An accepted hand-off resolves the question — stamp the runtime
            # resolution time (first-wins) for the dashboard's avg-resolution (#97).
            if outcome == "accepted" and question_id is not None:
                mark_question_resolved(session, question_id, self._now_factory())
                # NB: the #274 answer capture is deliberately NOT done here — it runs
                # in its own transaction AFTER this one commits (see submit_resume /
                # _capture_answer) so a capture failure can never roll back the accept.
                # Seed the chat with the asker's own request text (the hand-off
                # draft) as its first message, so the responder opens the thread
                # already seeing what they were asked — not an empty conversation.
                # Skipped for "direct" (no chat thread exists at all).
                if question_consult_method(session, question_id) != "direct":
                    draft = (values.get("draft") or "").strip()
                    asker_id = (values.get("asker") or {}).get("id")
                    if draft and asker_id is not None:
                        # Real wall-clock, not self._now_factory(): this message
                        # must sort before whatever a human sends next via
                        # POST /messages, which always timestamps with
                        # dt.datetime.now() (routes.py) regardless of the
                        # graph's (possibly injected/frozen) clock.
                        create_message(session, primary, asker_id, draft, dt.datetime.now())
                        # If both parties have linked Slack, this is also their
                        # very first look at the request — set up (or reuse)
                        # their shared Slack channel and post the SAME draft
                        # into it (#hand-off-chat), so accepting via TEKIJIN
                        # drops them straight into a normal-looking Slack
                        # conversation with the draft already "sent". Fire and
                        # forget: this method runs synchronously inside
                        # POST /answer's request/response cycle with no
                        # `BackgroundTasks` to defer to (see
                        # `schedule_channel_setup_and_draft`'s docstring).
                        parties = thread_parties(session, primary)
                        if parties is not None:
                            get_notifier().announce_handoff_accepted(
                                self._session_factory,
                                thread_id=primary,
                                parties=parties,
                                draft=draft,
                            )
            return "recorded", outcome

    def _capture_answer(self, values: dict[str, Any], body: str) -> None:
        """Persist the responder's answer text as an ``answers`` row (#274).

        Runs in its OWN transaction, AFTER the outcome has already been committed by
        ``_record_outcome`` — so this is genuinely best-effort: any failure here (a
        flush error, a DB hiccup, a NULL responder) is logged and swallowed, and the
        hand-off has already resumed regardless. The answer is attributed to the
        recommendation's employee via an authoritative DB lookup (not the external
        ``E###`` wire form) and tagged with the question's first predicted topic so
        ``answers_by_topic`` reuse can find it. The dense embedding is computed BEFORE
        the transaction opens (so a slow encode does not hold DB row locks) and
        degrades to ``None`` on failure — still BM25-reusable.
        """

        body = body.strip()
        if not body:
            return
        embedding = self._embed_answer(body)
        try:
            with session_scope(self._session_factory) as session:
                question_id = values.get("question_id")
                if question_id is None:
                    return
                # Same resolution as the outcome write (#94-3): attribute the
                # answer to the person the responder actually saw, never to a
                # superseded row left behind by a mid-reroute disconnect.
                primary = self._resolve_primary(session, values)
                if primary is None:
                    logger.warning(
                        "no recommendation to attribute the captured answer (question %s)",
                        question_id,
                    )
                    return
                responder_id = recommendation_employee_id(session, primary)
                if responder_id is None:
                    logger.warning(
                        "no responder to attribute the captured answer (question %s)",
                        question_id,
                    )
                    return
                topics = values.get("topics") or []
                create_answer(
                    session,
                    question_id=question_id,
                    responder_id=responder_id,
                    body=body,
                    topic=topics[0] if topics else None,
                    embedding=embedding,
                    created_at=self._now_factory(),
                )
        except Exception:  # pragma: no cover - defensive; exercised via logging path
            logger.exception("failed to capture the answer; the hand-off proceeds without it")

    def schedule_knowledge_embed(self) -> None:
        """Fire-and-forget: fill embeddings for any knowledge units missing one, so a
        unit just approved in Slack becomes searchable by ``knowledge_answer`` without a
        manual indexing pass (#531). Runs in a daemon thread — encoding can exceed
        Slack's interactivity budget — and swallows/logs its own errors (a missing
        embedding leaves the unit un-searchable, never crashes the approve click)."""

        def _run() -> None:
            try:
                with session_scope(self._session_factory) as session:
                    embed_knowledge_units(session, self._embedder)
            except Exception:  # noqa: BLE001 - background thread boundary
                logger.warning("scheduled knowledge embed failed", exc_info=True)

        threading.Thread(target=_run, daemon=True, name="knowledge-embed").start()

    def _embed_answer(self, body: str) -> list[float] | None:
        """Embed an answer body for dense reuse, or ``None`` if embedding fails.

        Kept non-fatal: the first ``encode`` may load a heavy model and can raise
        (missing ML deps, OOM). A NULL embedding still leaves the answer reusable
        via BM25 and visible to history/metrics, so a failure must not lose the
        answer or block the hand-off.
        """

        try:
            return self._embedder.encode([body], kind=PASSAGE)[0]
        except Exception:  # pragma: no cover - defensive; exercised via logging path
            logger.exception("failed to embed captured answer; storing without embedding")
            return None

    # -- /handoff : responder-facing view (product-spec 画面4) ------------- #
    def get_handoff(self, session_id: str) -> schemas.HandoffResponse:
        """Assemble the responder-facing payload for a ``send``-interrupt session.

        Read-only: reads the durable checkpoint (never advances the graph) plus
        two DB lookups (asker identity, responder reuse totals). Raises
        :class:`HandoffNotFound` (404) when there is no paused run at all
        (unknown / finished); :class:`SessionConflict` (409) when the run is
        paused elsewhere (a clarification is owed to the asker, not a responder
        outcome).

        The snapshot + pending check runs under the per-session lock (as
        ``submit_resume`` / ``stream_events`` do) so it is atomic against a
        concurrent dispatch: a reload cannot pass the "still awaiting" check on a
        stale snapshot while another reader is mid-way through advancing the graph.
        """

        with self._lock(session_id):
            snapshot = self._snapshot(session_id)
            next_nodes = tuple(snapshot.next)
            if not next_nodes:
                raise HandoffNotFound("no responder handoff for this session")
            if next_nodes[0] != "send":
                raise SessionConflict("session is not awaiting a responder outcome")
            # An outcome already queued (submitted, not yet consumed by an /events
            # reader) leaves the durable snapshot at ``send`` with the same, already
            # decided recommendation. Treat it as no-longer-offerable so a reload
            # does not re-render the form and invite a second submission (409).
            ctx = self._reg_get(session_id)
            if ctx is not None and ctx.pending is not None:
                raise HandoffNotFound("this handoff has already been answered")

        values = snapshot.values
        recs = values.get("recommendations") or []
        primary = recs[0] if recs else None
        asker_id = (values.get("asker") or {}).get("id")

        responder: schemas.Recommendation | None = None
        reuse = {"reuse_count": 0, "helpful_answer_count": 0}
        asker_name: str | None = None
        asker_dept: str | None = None
        with session_scope(self._session_factory) as session:
            # The generation token the responder will echo back on /answer must be
            # the SAME id the outcome will be recorded on — resolved, not the raw
            # checkpoint value. Otherwise a stale checkpoint (#94-3) would hand out
            # a superseded id and the guard would 409 the responder's own submission.
            recommendation_id = self._resolve_primary(session, values)
            if asker_id is not None:
                asker_name, asker_dept = employee_brief(session, asker_id)
            if primary is not None:
                # person_id crosses the boundary in the external "E###" form,
                # mirroring the recommend event (events.py / model-definition).
                responder = schemas.Recommendation(
                    **{**primary, "person_id": schemas.format_employee_id(primary["person_id"])}
                )
                reuse = responder_reuse_stats(session, primary["person_id"])
            # SQL, not the checkpoint: the asker's method choice must also be
            # visible to the inbox / chat gating, which never read the graph
            # state, so it lives only in Question.consult_method.
            consult_method = schemas.normalize_consult_method(
                question_consult_method(session, values.get("question_id"))
            )

        return schemas.HandoffResponse(
            session_id=session_id,
            question=values.get("question") or "",
            question_id=values.get("question_id"),
            asker=schemas.HandoffAsker(
                id=schemas.format_employee_id(asker_id) if asker_id is not None else "",
                name=asker_name,
                dept=asker_dept,
            ),
            topics=values.get("topics") or [],
            products=values.get("products") or [],
            situation=values.get("situation"),
            missing=values.get("missing") or [],
            responder=responder,
            draft=values.get("draft") or "",
            reuse_count=reuse["reuse_count"],
            helpful_answer_count=reuse["helpful_answer_count"],
            recommendation_id=recommendation_id,
            consult_method=consult_method,
        )

    def structure_question(self, session_id: str) -> schemas.QuestionStructureResponse:
        """Re-draft the asker's raw question into the four hand-off fields (#475).

        On-demand and READ-ONLY: called from the result screen when the asker asks
        the AI to tidy their question before sending. It reads the durable snapshot
        (the raw question + C1's ``situation``/``topics``) and runs the structurer
        OUTSIDE the graph, so it never touches the C1 critical path
        ([[tekijin-latency-and-streaming]]) and never advances or persists any state.

        Raises :class:`HandoffNotFound` (404) when the session has no question yet
        (unknown / not started). The structurer is always wired by the factory, so a
        missing one is a programming error surfaced as a 500 by the route, not a
        user-facing state.
        """

        if self._structurer is None:  # pragma: no cover - always wired by the factory
            raise RuntimeError("question structurer is not configured")
        snapshot = self._snapshot(session_id)
        values = getattr(snapshot, "values", None) or {}
        question = (values.get("question") or "").strip()
        if not question:
            raise HandoffNotFound("no question to structure for this session")
        result = self._structurer.structure(
            question,
            situation=values.get("situation"),
            topics=values.get("topics") or [],
        )
        return schemas.QuestionStructureResponse(
            session_id=session_id,
            summary=result.summary,
            environment=result.environment,
            tried=result.tried,
            blocker=result.blocker,
        )

    def save_handoff_draft(
        self,
        session_id: str,
        draft: str,
        consult_method: str,
        *,
        actor_id: int | None = None,
    ) -> None:
        """Persist the asker's edited hand-off ``draft`` (画面3 → 画面4) (#174) and
        their chosen consultation method (#245).

        ``consult_method`` goes to SQL (``Question.consult_method``), not the
        checkpoint: the inbox and the chat-thread gating are SQL-only, so that is
        the only place they can see it. It lives on the QUESTION rather than the
        recommendation so it survives a decline+reroute, which creates a new
        rank-1 recommendation row for the same question.

        Only the durable ``draft`` value is updated, never the recommendation ids
        or the accept/decline outcome, so the responder-side lifecycle (#94) is
        untouched. Validation mirrors :meth:`get_handoff` under the per-session
        lock: the run must be paused at ``send`` and not already answered, so an
        edit cannot be saved against a finished run, a clarification, or a
        hand-off whose outcome was already queued.

        Implicit C7 feedback (#237 Phase 1): when the sent text differs from the
        draft C7 generated, that edit is the one real "human corrected the AI" signal
        the runtime already had but discarded — it is recorded as ``c7`` feedback
        here (the generated vs. sent text in the payload). ``actor_id`` is who edited.
        """

        text = draft.strip()
        if not text:  # defense in depth; the request schema already rejects blanks
            raise SessionInvalid("draft must be a non-empty string")
        with self._lock(session_id):
            session = self._session_factory()
            try:
                graph = self._graph(session)
                config = self._config(session_id)
                # Snapshot + write share one graph/session under the lock, so the
                # validated state cannot change before update_state runs.
                state = graph.get_state(config)
                next_nodes = tuple(state.next)
                if not next_nodes:
                    raise HandoffNotFound("no responder handoff for this session")
                if next_nodes[0] != "send":
                    raise SessionConflict("session is not awaiting a responder outcome")
                ctx = self._reg_get(session_id)
                if ctx is not None and ctx.pending is not None:
                    raise HandoffNotFound("this handoff has already been answered")
                generated = state.values.get("draft")
                question_id = state.values.get("question_id")
                graph.update_state(config, {"draft": text})
                if question_id is not None:
                    update_question_consult_method(session, question_id, consult_method)
                    session.commit()
            finally:
                session.close()
        # Record the edit as C7 feedback OUTSIDE the graph session/lock: it is an
        # append-only learning signal, not part of the checkpointer transaction. The
        # draft save above is already committed, so a feedback-write failure must be
        # swallowed (not re-raised) — otherwise the caller would see a 500 for a
        # request whose primary effect already succeeded.
        if isinstance(generated, str) and generated.strip() != text:
            try:
                with session_scope(self._session_factory) as fb_session:
                    record_feedback(
                        fb_session,
                        stage="c7",
                        kind="draft_edited",
                        question_id=question_id if isinstance(question_id, str) else None,
                        session_id=session_id,
                        payload={"generated": generated, "sent": text},
                        actor_id=actor_id,
                    )
            except Exception:  # noqa: BLE001 - best-effort signal; never fail the committed save
                logger.exception("failed to record c7 draft feedback for session %s", session_id)

    def select_handoff_candidate(
        self, session_id: str, person_id: int
    ) -> schemas.HandoffSelectResponse:
        """Reselect the hand-off target among the currently shown candidates
        and regenerate the draft for them (#200/#A1/#204).

        Validation mirrors :meth:`save_handoff_draft`: the run must be paused at
        ``send`` and not already answered. Unlike a draft-only edit, this also
        reorders ``recommendations`` / ``recommendation_ids`` /
        ``primary_recommendation_id`` in the durable checkpoint (the selected
        candidate becomes primary), and syncs the DB ``Recommendation.rank`` so
        ``/inbox`` (SQL-only) reflects the new primary responder.
        """

        with self._lock(session_id):
            session = self._session_factory()
            try:
                graph = self._graph(session)
                config = self._config(session_id)
                next_nodes = tuple(graph.get_state(config).next)
                if not next_nodes:
                    raise HandoffNotFound("no responder handoff for this session")
                if next_nodes[0] != "send":
                    raise SessionConflict("session is not awaiting a responder outcome")
                ctx = self._reg_get(session_id)
                if ctx is not None and ctx.pending is not None:
                    raise HandoffNotFound("this handoff has already been answered")

                values = graph.get_state(config).values
                recs = list(values.get("recommendations") or [])
                rec_ids = list(values.get("recommendation_ids") or [])
                index = next((i for i, r in enumerate(recs) if r["person_id"] == person_id), None)
                if index is None:
                    raise SessionInvalid("person_id is not among the current recommendations")

                selected = recs.pop(index)
                reordered = [selected, *recs]
                if len(rec_ids) != len(reordered):
                    # Both lists are built from the same c6_score batch, so this
                    # cannot happen today. Fail loudly rather than reordering only
                    # `recommendations`: a silent desync would leave
                    # `primary_recommendation_id` on the OLD top pick, so the
                    # responder's accept/decline would land on a different row
                    # than the person the UI shows.
                    raise SessionInvalid(
                        "recommendation ids are out of sync with the shown recommendations"
                    )
                sid = rec_ids.pop(index)
                new_ids = [sid, *rec_ids]

                missing, known_values = draft_context(values)
                new_draft = self._draft.draft(
                    values["question"],
                    selected,
                    values.get("asker"),
                    missing,
                    situation=values.get("situation"),
                    topics=values.get("topics") or [],
                    known_values=known_values,
                )
                graph.update_state(
                    config,
                    {
                        "recommendations": reordered,
                        "recommendation_ids": new_ids,
                        "primary_recommendation_id": new_ids[0] if new_ids else None,
                        "draft": new_draft,
                    },
                )
            finally:
                session.close()

            if new_ids:
                with session_scope(self._session_factory) as write_session:
                    reorder_recommendation_ranks(write_session, new_ids)

        return schemas.HandoffSelectResponse(
            session_id=session_id,
            responder=schemas.Recommendation(
                **{**selected, "person_id": schemas.format_employee_id(selected["person_id"])}
            ),
            draft=new_draft,
            # NOT routed through `_resolve_primary` (unlike `/handoff`): the
            # `update_state` above just wrote `recommendations[0]` and
            # `primary_recommendation_id` from the SAME `new_ids`, so they agree by
            # construction here — there is no stale window to resolve away (#94-3).
            # This id is asker-side display; the responder's generation token always
            # comes from `/handoff`.
            recommendation_id=(new_ids[0] if new_ids else 0),
        )

    def exclude_handoff_target(
        self, session_id: str, person_id: int, *, actor_id: int | None = None
    ) -> None:
        """Asker excludes the current send target ("この人には聞かない") → reroute (#260).

        Queues the same reroute a responder decline drives (``_after_send`` →
        ``reroute`` → ``c6_score`` → ``c7_draft`` → ``send``), so the freshly-scored
        next candidate and its regenerated draft arrive over the open ``/events``
        stream — reusing the graph's decline machinery and its persistence
        unchanged. The excluded person's shown rank-1 row is stamped
        ``outcome="excluded"`` (distinct from a responder's ``declined``: nobody was
        actually asked), which — like any non-NULL outcome — drops it out of that
        person's ``/inbox`` (``pending_handoffs_for_responder`` filters ``outcome IS
        NULL``) and the dashboard's *pending* count, without counting against their
        acceptance rate. The exclusion is ALSO recorded as a ``c6`` feedback signal
        (#237 Phase 1) — the learning channel, separate from the hand-off lifecycle.

        Validation mirrors :meth:`select_handoff_candidate`: the run must be paused
        at ``send`` and not already answered / queued. ``person_id`` must be the
        current primary (the reroute path only ever declines the top pick), else
        422 — so an exclusion never silently reroutes a candidate other than the
        one the asker named (no mis-send).
        """

        with self._lock(session_id):
            snapshot = self._snapshot(session_id)
            next_nodes = tuple(snapshot.next)
            if not next_nodes:
                raise HandoffNotFound("no responder handoff for this session")
            if next_nodes[0] != "send":
                raise SessionConflict("session is not awaiting a responder outcome")
            ctx = self._reg_get(session_id)
            if ctx is not None and ctx.pending is not None:
                raise HandoffNotFound("this handoff has already been answered")

            recs = snapshot.values.get("recommendations") or []
            primary = recs[0] if recs else None
            if primary is None or primary["person_id"] != person_id:
                raise SessionInvalid("person_id is not the current hand-off target")

            question_id = snapshot.values.get("question_id")

            # Terminate the excluded person's shown rank-1 row so it stops showing
            # as a pending hand-off in THEIR /inbox and the dashboard-pending count
            # (first-wins, idempotent). Distinct value from a responder decline —
            # the asker withdrew before asking, so it must not skew acceptance rate.
            self._record_outcome(session_id, snapshot.values, "excluded")

            # Queue the reroute exactly as a responder decline does; the open
            # /events reader consumes it and re-scores / re-drafts the next pick.
            ctx = self._reg_ensure(session_id)
            ctx.pending = Command(resume="declined")
            ctx.touched_at = self._clock()

        # Record the exclusion as a c6 learning signal OUTSIDE the lock — append-only
        # and best-effort: the reroute is already queued, so a feedback-write failure
        # must not fail the request (mirrors the c7 draft-edit signal above).
        try:
            with session_scope(self._session_factory) as fb_session:
                record_feedback(
                    fb_session,
                    stage="c6",
                    kind="person_excluded",
                    question_id=question_id if isinstance(question_id, str) else None,
                    session_id=session_id,
                    target=schemas.format_employee_id(person_id),
                    actor_id=actor_id,
                )
        except Exception:  # noqa: BLE001 - best-effort signal; never fail the queued reroute
            logger.exception("failed to record c6 exclude feedback for session %s", session_id)

    def refer_handoff_target(
        self, session_id: str, referred_to: int, *, actor_id: int | None = None
    ) -> None:
        """Responder refers the hand-off to a person THEY name ("他に適任者がいる → X", #518).

        Records ``outcome="referred"`` (+ the named employee) on the current primary,
        seats the named person at rank 1 for the reroute via ``referred_responder_id``
        (c6_score's referral pin), then queues the same reroute a decline drives — so
        the newly-drafted hand-off to the named person arrives over the open
        ``/events`` stream and lands in THEIR ``/inbox``. Distinct outcome from a plain
        decline (``referred`` doesn't skew acceptance rate).

        Validation mirrors :meth:`exclude_handoff_target` (paused at ``send``, not
        already answered). 422 when the named person is unknown, is the asker (they
        cannot answer their own question), or is the current responder (a no-op
        self-referral). Object-level auth (the responder-only rule) is enforced at the
        route, as for accept/decline.
        """

        with self._lock(session_id):
            session = self._session_factory()
            try:
                graph = self._graph(session)
                config = self._config(session_id)
                snapshot = graph.get_state(config)
                next_nodes = tuple(snapshot.next)
                if not next_nodes:
                    raise HandoffNotFound("no responder handoff for this session")
                if next_nodes[0] != "send":
                    raise SessionConflict("session is not awaiting a responder outcome")
                ctx = self._reg_get(session_id)
                if ctx is not None and ctx.pending is not None:
                    raise HandoffNotFound("this handoff has already been answered")

                values = snapshot.values
                recs = values.get("recommendations") or []
                primary = recs[0] if recs else None
                if primary is None:
                    raise HandoffNotFound("no responder handoff for this session")
                asker_id = (values.get("asker") or {}).get("id")
                current_responder_id = primary["person_id"]

                if not employee_exists(session, referred_to):
                    raise SessionInvalid("referred employee not found")
                if referred_to == asker_id:
                    raise SessionInvalid("cannot refer to the asker")
                if referred_to == current_responder_id:
                    raise SessionInvalid("cannot refer to the current responder")
                # Someone who already declined this question cannot be re-seated: the
                # c6 pin would drop them (they're in declined_ids), so the referral
                # would silently no-op while still recording a misleading
                # outcome="referred". Reject up front instead (#518 review).
                if referred_to in (values.get("declined_ids") or []):
                    raise SessionInvalid("cannot refer to someone who already declined")

                question_id = values.get("question_id")

                # 'referred' (not 'declined') so the referrer's row drops out of their
                # inbox / pending count without counting against their acceptance rate,
                # and records WHO they named for the audit trail.
                self._record_outcome(session_id, values, "referred", referred_to=referred_to)
                # Seat the named person at rank 1 on the reroute (c6 referral pin).
                graph.update_state(config, {"referred_responder_id": referred_to})
                # Queue the reroute exactly as a decline does; the open /events reader
                # consumes it and re-scores (seating the referral) / re-drafts.
                ctx = self._reg_ensure(session_id)
                ctx.pending = Command(resume="declined")
                ctx.touched_at = self._clock()
            finally:
                session.close()

        # Record the referral as a c6 learning signal OUTSIDE the lock — best-effort,
        # append-only, mirrors the exclude signal above.
        try:
            with session_scope(self._session_factory) as fb_session:
                record_feedback(
                    fb_session,
                    stage="c6",
                    kind="person_referred",
                    question_id=question_id if isinstance(question_id, str) else None,
                    session_id=session_id,
                    target=schemas.format_employee_id(referred_to),
                    actor_id=actor_id,
                )
        except Exception:  # noqa: BLE001 - best-effort signal; never fail the queued reroute
            logger.exception("failed to record c6 refer feedback for session %s", session_id)

    def regenerate_handoff_draft(self, session_id: str, *, actor_id: int | None = None) -> None:
        """Asker asks the AI to regenerate the hand-off draft ("下書きの作り直し", #260).

        Queues a ``redraft`` resume that sends the graph back through ``c7_draft``
        for the current top candidate (``_after_send`` → ``c7_draft`` → ``send``), so
        the freshly-generated draft arrives over the open ``/events`` stream and
        rides the SSE thought-process like any other C7 run — discarding any manual
        edit the asker had saved (that is the point of "作り直し"). The regeneration
        is recorded as a ``c7`` feedback signal (``draft_regenerated``, #237).

        Validation mirrors :meth:`exclude_handoff_target`: the run must be paused at
        ``send`` and not already answered / queued.
        """

        with self._lock(session_id):
            snapshot = self._snapshot(session_id)
            next_nodes = tuple(snapshot.next)
            if not next_nodes:
                raise HandoffNotFound("no responder handoff for this session")
            if next_nodes[0] != "send":
                raise SessionConflict("session is not awaiting a responder outcome")
            ctx = self._reg_get(session_id)
            if ctx is not None and ctx.pending is not None:
                raise HandoffNotFound("this handoff has already been answered")

            question_id = snapshot.values.get("question_id")
            previous = snapshot.values.get("draft")

            # Queue the redraft loop; the open /events reader consumes it and
            # re-runs c7_draft, re-emitting a ``draft`` event before re-pausing.
            ctx = self._reg_ensure(session_id)
            ctx.pending = Command(resume="redraft")
            ctx.touched_at = self._clock()

        # Record the regeneration as a c7 learning signal OUTSIDE the lock —
        # append-only, best-effort (mirrors the c7 draft-edit signal above): the
        # redraft is already queued, so a feedback-write failure must not fail it.
        try:
            with session_scope(self._session_factory) as fb_session:
                record_feedback(
                    fb_session,
                    stage="c7",
                    kind="draft_regenerated",
                    question_id=question_id if isinstance(question_id, str) else None,
                    session_id=session_id,
                    payload={"previous": previous} if isinstance(previous, str) else None,
                    actor_id=actor_id,
                )
        except Exception:  # noqa: BLE001 - best-effort signal; never fail the queued redraft
            logger.exception("failed to record c7 redraft feedback for session %s", session_id)

    def correct_interpretation(
        self, session_id: str, supplement: str, *, actor_id: int | None = None
    ) -> None:
        """Asker corrects the AI's interpretation ("解釈の訂正", #260).

        Folds ``supplement`` into the question and re-runs the WHOLE pipeline from
        C1 — the same "add context, re-understand" move as the ``ask → c1_intent``
        clarification edge, but asker-initiated from the result screen. Implemented
        by queuing a fresh invoke input (a dict, not a ``Command``): on the next
        ``/events`` read that restarts the graph from ``reset`` (which clears the
        per-question control fields — declined_ids / recommendations / route — so
        the corrected question is scored cleanly), re-emitting understood → route →
        recommend → draft (or a fresh ``followup`` if the correction made C2 want
        one) over the stream.

        The previously handed-off primary is stamped ``outcome="superseded"`` first,
        so the abandoned recommendation stops showing as pending in that person's
        ``/inbox`` and the dashboard-pending count (same rationale as an exclude —
        distinct value since nobody declined). The correction is recorded as a
        ``c1`` feedback signal (#237).

        Validation mirrors :meth:`exclude_handoff_target`: the run must be paused at
        ``send`` and not already answered / queued.
        """

        text = supplement.strip()
        if not text:  # defense in depth; the request schema already rejects blanks
            raise SessionInvalid("supplement must be a non-empty string")

        with self._lock(session_id):
            snapshot = self._snapshot(session_id)
            next_nodes = tuple(snapshot.next)
            if not next_nodes:
                raise HandoffNotFound("no responder handoff for this session")
            if next_nodes[0] != "send":
                raise SessionConflict("session is not awaiting a responder outcome")
            ctx = self._reg_get(session_id)
            if ctx is not None and ctx.pending is not None:
                raise HandoffNotFound("this handoff has already been answered")

            values = snapshot.values
            question = values.get("question")
            asker = values.get("asker")
            question_id = values.get("question_id")
            if not isinstance(question, str) or not question.strip():
                # A paused-at-send run always has a question; guard defensively so a
                # corrupt checkpoint fails cleanly instead of enriching ``None``.
                raise SessionInvalid("session has no question to correct")

            enriched = f"{question} {text}".strip()
            # Persist the enriched body FIRST so /inbox and /history show the question
            # that was actually processed, not the pre-correction original (#268).
            # Ordering matters for the failure mode (#272 review): if this write
            # throws, nothing durable has changed yet (the supersede below has not
            # run and no re-run is queued), so the session stays cleanly at ``send``
            # and a retry heals it — rather than committing the supersede and then
            # failing, which would leave the checkpoint disagreeing with the DB.
            if isinstance(question_id, str):
                with session_scope(self._session_factory) as body_session:
                    update_question_body(body_session, question_id, enriched)

            # Terminate the abandoned hand-off so it drops out of the responder's
            # /inbox and the dashboard-pending count (see exclude_handoff_target).
            self._record_outcome(session_id, values, "superseded")

            # Queue a FRESH invoke (dict input) so the stream restarts from reset → C1.
            ctx = self._reg_ensure(session_id)
            ctx.pending = {
                "question": enriched,
                "asker": asker,
                "now": self._now_factory(),
                "question_id": question_id,
            }
            ctx.touched_at = self._clock()

        # Record the correction as a c1 learning signal OUTSIDE the lock — append-only
        # and best-effort (mirrors the c7 signals above): the re-run is already
        # queued, so a feedback-write failure must not fail the request.
        try:
            with session_scope(self._session_factory) as fb_session:
                record_feedback(
                    fb_session,
                    stage="c1",
                    kind="interpretation_corrected",
                    question_id=question_id if isinstance(question_id, str) else None,
                    session_id=session_id,
                    payload={"supplement": text},
                    actor_id=actor_id,
                )
        except Exception:  # noqa: BLE001 - best-effort signal; never fail the queued re-run
            logger.exception("failed to record c1 correction feedback for session %s", session_id)

    def discard_session(self, session_id: str) -> None:
        """Drop a one-shot run that no one will continue (Discord ``/ask``, #60).

        A run paused at an interrupt is never swept (``_sweep`` keeps every session
        a human is waited on), so a caller that only reads the first segment must
        end it explicitly. A pause at ``send`` has already persisted the rank-1
        recommendation, which the responder's ``/inbox`` lists as pending; stamp it
        ``outcome="unsent"`` (nobody was asked, like ``excluded``/``superseded``) so
        it drops out. Then delete the checkpoint, unlink the question from it (its
        ``/history`` result page would 404), and forget the registry entry and
        lock, so repeated one-shot runs do not accumulate.
        """

        with self._lock(session_id):
            snapshot = self._snapshot(session_id)
            if tuple(snapshot.next)[:1] == ("send",):
                self._record_outcome(session_id, snapshot.values, "unsent")
            self._checkpointer.delete_thread(session_id)
            with session_scope(self._session_factory) as session:
                detach_question_session(session, session_id)
            with self._registry_guard:
                self._registry.pop(session_id, None)
        with self._locks_guard:
            self._locks.pop(session_id, None)

    # -- /events : stream ------------------------------------------------- #
    def is_streamable(self, session_id: str) -> bool:
        """True if there is a queued run, a paused run, or a replayable terminal.

        The terminal case lets a client that dropped before receiving ``done`` /
        the terminal ``message`` reconnect and re-fetch it (codex#3). A genuinely
        unknown session has no checkpoint at all → not streamable → 404.
        """

        ctx = self._reg_get(session_id)
        if ctx is not None and ctx.pending is not None:
            return True
        snapshot = self._snapshot(session_id)
        return bool(snapshot.next) or snapshot.values.get("last_event") is not None

    def stream_events(self, session_id: str) -> Iterator[ServerSentEvent]:
        """Stream the queued run, or reconnect to a paused/mid-run state.

        Held under the per-session lock for the whole stream, so a second
        concurrent ``/events`` cannot start the graph a second time (which would
        double-insert recommendations): it blocks, then finds the input already
        consumed and reconnects instead. A fresh DB session is opened for the run
        and always closed; everything is inside try/except so any failure closes
        cleanly with one generic ``error`` event (details logged, never sent).
        """

        with self._lock(session_id):
            session = self._session_factory()
            try:
                yield from self._dispatch_stream(session, session_id)
            except Exception:
                logger.exception("SSE stream failed for session %s", session_id)
                yield _error_event()
            finally:
                session.close()

    def _dispatch_stream(self, session: Session, session_id: str) -> Iterator[ServerSentEvent]:
        graph = self._graph(session)
        config = self._config(session_id)
        ctx = self._reg_get(session_id)
        if ctx is not None and ctx.pending is not None:
            pending = ctx.pending
            ctx.pending = None
            ctx.touched_at = self._clock()
            yield from self._run(graph, config, pending)
            return

        state = graph.get_state(config)
        if ctx is not None:
            ctx.touched_at = self._clock()
        if not state.next:
            # Finished run: replay its terminal event so a reconnecting client can
            # still receive done / message (codex#3). Unknown session → nothing.
            replay = replay_terminal(state.values)
            if replay is not None:
                # #512: the progress FIRST, then the terminal — the same order the
                # live run produced, so a session revisited from the history list
                # renders the identical thinking steps instead of a bare outcome.
                # Gated on there being a terminal at all: without one this is not a
                # finished run but an unknown session, which must stay empty.
                yield from progress_events(state.values)
                yield replay
            return
        node = state.next[0]
        if node in _INTERRUPT_NODES:
            yield from progress_events(state.values)
            yield from reconnect_events(node, state.values)
        else:
            # codex#6: a client disconnected mid-run (before ask/send). The run is
            # parked at a normal node — continue it from the checkpoint.
            yield from self._run(graph, config, None)

    # -- internals -------------------------------------------------------- #
    def _graph(self, session: Session) -> Any:
        return build_agent(
            self._embedder,
            session,
            checkpointer=self._checkpointer,
            intent_model=self._intent,
            sufficiency_model=self._sufficiency,
            draft_model=self._draft,
            answerability_model=self._answerability,
            answerability_threshold=self._answerability_threshold,
            self_answer_model=self._self_answer,
            retriever=self._retriever,
            scorer=self._scorer,
            bm25_weight=self._bm25_weight,
            prior_answer_reuse_min=self._prior_answer_reuse_min,
            prior_answer_relevance_floor=self._prior_answer_relevance_floor,
            daily_evidence=self._daily_evidence,
            daily_knowledge_enabled=self._daily_knowledge_enabled,
            knowledge_answer_min_similarity=self._knowledge_answer_min_similarity,
            query_expansion_enabled=self._query_expansion_enabled,
            question_fit_enabled=self._question_fit_enabled,
            branch_constraint_enabled=self._branch_constraint_enabled,
            additive_self_answer_enabled=self._additive_self_answer_enabled,
            additive_self_answer_floor=self._additive_self_answer_floor,
            score_all_employees=self._score_all_employees,
            similar_askers_enabled=self._similar_askers_enabled,
        )

    def _run(
        self, graph: Any, config: dict[str, Any], agent_input: Any
    ) -> Iterator[ServerSentEvent]:
        """Stream one run segment, persisting topics/recommendations as it goes.

        ``question_id`` comes from the fresh input (a ``/ask`` dict) or, for a
        resume / mid-run continuation, from the durable state. After the segment,
        the shown recommendation ids AND the terminal event (if the run finished)
        are written back into the state — the former so ``/answer`` records the
        outcome durably, the latter so a reconnect can replay ``done`` / ``message``.
        """

        question_id = self._run_question_id(graph, config, agent_input)
        rec_ids: list[int] | None = None
        terminal: ServerSentEvent | None = None
        # Track the C5 route as it streams: on the document route, C6 runs only to
        # compute an inline person-fallback name in the answer (#279) — it is NOT a
        # live hand-off, so its recommendations must NOT be persisted or surfaced as
        # a `recommend` event (they could never resolve, permanently inflating the
        # pending-handoff KPI and dead-ending responder inboxes — #281 review).
        route: str | None = None
        # #70: when the answerability critic is wired, hold C6's recommend event
        # AND its persistence until the critic accepts — a rejected candidate set
        # must never surface or persist (it could never resolve; same phantom-row
        # harm as the document route, #281). Inert when no critic is wired.
        critique_wired = self._answerability is not None
        pending_recommend: ServerSentEvent | None = None
        pending_rec_data: dict[str, Any] | None = None
        # Per-stage timing for the latency KPI (#177): each real node's duration is
        # (this node's end) - (previous node's end within this segment). `prev`
        # starts at segment start, so a resume segment does not count the human wait.
        event_rows: list[tuple[str, dt.datetime, dt.datetime, dict[str, Any] | None]] = []
        prev = self._now_factory()
        # Occupy a backpressure slot only while the graph actually executes (#180);
        # released by the finally on completion or client disconnect.
        try:
            with self._run_slot():
                for update in graph.stream(agent_input, config, stream_mode="updates"):
                    for node, data in update.items():
                        if node == "c1_intent":
                            self._persist_topics(question_id, data)
                        elif node == "ask":
                            # A clarification reply enriched the question in-graph;
                            # persist it so /inbox and /history match the run (#268).
                            self._persist_question_body(question_id, data)
                        elif node == "knowledge_answer":
                            # #357 slice 4c: a grounded knowledge answer terminates
                            # BEFORE C5, so c5_route never persists a route. Stamp a
                            # synthetic "knowledge" route so dashboards that segment by
                            # route account for knowledge-answered questions instead of
                            # silently omitting them (self-resolution is still credited
                            # by _persist_self_answered on the shared self_answered node).
                            if (data or {}).get("self_answer_grounded"):
                                route = "knowledge"
                                self._persist_route(question_id, {"route": "knowledge"})
                        elif node == "c5_route":
                            route = (data or {}).get("route") or route
                            self._persist_route(question_id, data)
                        elif node == "self_answered":
                            # #291: a grounded self-answer resolves the question WITHOUT
                            # a hand-off — mark it self-resolved regardless of the
                            # underlying route (document already stamps at c5, but
                            # prior_answer would otherwise never get resolution credit,
                            # undercounting exactly the self-resolution KPI this feature
                            # exists to move). ``mark_self_resolved`` is first-wins.
                            self._persist_self_answered(question_id)
                        elif node == "c6_score" and route != "document":
                            # Defer persistence to the critic's verdict ONLY when C6
                            # actually produced candidates (a non-empty set routes to
                            # `answerability`; an empty one routes straight to
                            # `no_candidate`, bypassing the critic, so it must persist/
                            # emit exactly like the non-critique path — a no-op insert).
                            if critique_wired and (data or {}).get("recommendations"):
                                pending_rec_data = data
                            else:
                                rec_ids = self._persist_recommendations(question_id, data)
                        elif node == "c7_draft":
                            # An explicit document fallback resumes *after* C6. Turn
                            # its dormant ranking into the visible/persisted hand-off
                            # now (normal person runs already have rec_ids from C6).
                            values = graph.get_state(config).values
                            if values.get("document_fallback_requested") and not values.get(
                                "recommendation_ids"
                            ):
                                rec_data = {"recommendations": values.get("recommendations") or []}
                                # A disconnect can happen after SQL insert but
                                # before _persist_run_state. Reuse those ids instead
                                # of double-inserting on the continuation.
                                if question_id is not None:
                                    with session_scope(self._session_factory) as session:
                                        rec_ids = shown_recommendation_ids(session, question_id)
                                if not rec_ids:
                                    rec_ids = self._persist_recommendations(question_id, rec_data)
                                recommend = node_event("c6_score", rec_data)
                                if recommend is not None:
                                    yield recommend
                        if node == "__interrupt__":
                            event = interrupt_event(_interrupt_payload(data))
                        else:
                            # Record this stage's timing (all compute nodes, not just
                            # the ones that surface an SSE event).
                            now_dt = self._now_factory()
                            if question_id is not None:
                                event_rows.append((node, prev, now_dt, None))
                            prev = now_dt
                            # latency only when the segment actually recorded stages
                            # (question_id set) — else leave it None, not a bogus 0.
                            latency = (
                                _segment_latency_ms(event_rows)
                                if node in TERMINAL_NODES and event_rows
                                else None
                            )
                            event = node_event(node, data, latency_ms=latency)
                            # #279/#281: no `recommend` event on the document route —
                            # C6 there is a fallback note, not a hand-off in progress.
                            if node == "c6_score" and route == "document":
                                event = None
                            # #70: hold C6's recommend event until the critic decides
                            # (only when C6 produced candidates — see the persist guard
                            # above; an empty set bypasses the critic and emits normally).
                            elif (
                                node == "c6_score"
                                and critique_wired
                                and (data or {}).get("recommendations")
                            ):
                                pending_recommend = event
                                event = None
                            elif node == "answerability":
                                # Critic decided: on accept, persist the shown rows +
                                # release the recommend event; on reject, drop both (the
                                # `no_expert` terminal follows, nothing to surface).
                                if (data or {}).get("answerable"):
                                    # `pending_*` is None when C6 ran in a PRIOR segment
                                    # that the client disconnected before the critic ran
                                    # (reconnect resumes at `answerability`, codex#6). Re-
                                    # derive the shown recs from the durable state so the
                                    # accepted hand-off is still persisted + surfaced —
                                    # otherwise the outcome record is silently lost.
                                    if pending_rec_data is not None:
                                        rec_data = pending_rec_data
                                        recommend_event = pending_recommend
                                    else:
                                        values = graph.get_state(config).values
                                        rec_data = {
                                            "recommendations": values.get("recommendations") or []
                                        }
                                        recommend_event = node_event("c6_score", rec_data)
                                    rec_ids = self._persist_recommendations(question_id, rec_data)
                                    event = recommend_event
                                pending_recommend = None
                                pending_rec_data = None
                        if event is not None:
                            if event.event in TERMINAL_EVENTS:
                                terminal = event
                            yield event
                self._persist_run_state(graph, config, rec_ids, terminal)
        finally:
            # Flush collected stage timings even on a client disconnect / error
            # (GeneratorExit) — otherwise the whole segment's latency is lost and
            # the KPI under-counts exactly the reconnect-heavy sessions (#177 review).
            self._persist_events(question_id, event_rows)

    def _persist_run_state(
        self,
        graph: Any,
        config: dict[str, Any],
        rec_ids: list[int] | None,
        terminal: ServerSentEvent | None,
    ) -> None:
        updates: dict[str, Any] = {}
        if rec_ids:
            updates["recommendation_ids"] = rec_ids
            updates["primary_recommendation_id"] = rec_ids[0]
            updates["document_fallback_requested"] = False
        if terminal is not None:
            updates["last_event"] = {"event": terminal.event, "data": terminal.data}
        if updates:
            graph.update_state(config, updates)

    def _persist_events(
        self,
        question_id: str | None,
        event_rows: list[tuple[str, dt.datetime, dt.datetime, dict[str, Any] | None]],
    ) -> None:
        """Batch-write this segment's stage timings (latency KPI source, #177).

        Runs once after the stream segment (not in the hot loop), so the added DB
        write is a single insert per segment. No-op when there is nothing to record.
        """

        if question_id is None or not event_rows:
            return
        with session_scope(self._session_factory) as session:
            record_events(session, question_id, event_rows)

    def _run_question_id(self, graph: Any, config: dict[str, Any], agent_input: Any) -> str | None:
        if isinstance(agent_input, dict):
            return agent_input.get("question_id")
        return graph.get_state(config).values.get("question_id")

    def _persist_route(self, question_id: str | None, data: dict[str, Any] | None) -> None:
        if question_id is None:  # pragma: no cover - question_id always set via /ask
            logger.warning("c5 route with no question_id; skipping persist")
            return
        route = (data or {}).get("route")
        if route is None:  # pragma: no cover - c5 always sets a route
            return
        with session_scope(self._session_factory) as session:
            update_question_route(session, question_id, route)
            # A self-resolving route (document) ends the run with no human hand-off,
            # so the question is resolved the moment it is routed there (#97).
            if route == "document":
                mark_question_resolved(session, question_id, self._now_factory())

    def _persist_self_answered(self, question_id: str | None) -> None:
        """Record a grounded self-answer (#291) as a self-resolution (no hand-off).

        Sets ``resolution_kind="self"`` + ``resolved_at`` (first-wins), so the
        question counts toward the self-resolution rate and average resolution time
        for ANY route it self-answered on — not only ``document``.
        """

        if question_id is None:  # pragma: no cover - question_id always set via /ask
            return
        with session_scope(self._session_factory) as session:
            mark_self_resolved(session, question_id, self._now_factory())

    def _persist_topics(self, question_id: str | None, data: dict[str, Any] | None) -> None:
        if question_id is None:  # pragma: no cover - question_id always set via /ask
            logger.warning("c1 topics with no question_id; skipping persist")
            return
        topics = (data or {}).get("topics") or []
        with session_scope(self._session_factory) as session:
            update_question_topics(session, question_id, topics)

    def _persist_question_body(self, question_id: str | None, data: dict[str, Any] | None) -> None:
        """Persist a clarification-enriched question body (#268).

        The ``ask`` node folds the reply into ``question`` before looping back to
        C1; write it so ``/inbox`` and ``/history`` (which read ``Question.body``
        directly) reflect the question actually processed, not the pre-reply text.
        """

        if question_id is None:  # pragma: no cover - question_id always set via /ask
            logger.warning("enriched question with no question_id; skipping body persist")
            return
        body = (data or {}).get("question")
        if not isinstance(body, str) or not body.strip():
            return
        with session_scope(self._session_factory) as session:
            update_question_body(session, question_id, body)

    def _persist_recommendations(
        self, question_id: str | None, data: dict[str, Any] | None
    ) -> list[int] | None:
        recs = (data or {}).get("recommendations") or []
        if not recs:
            return None
        if question_id is None:  # pragma: no cover - question_id always set via /ask
            logger.warning("c6 recommendations with no question_id; skipping persist")
            return None
        with session_scope(self._session_factory) as session:
            return insert_shown_recommendations(session, question_id, recs)

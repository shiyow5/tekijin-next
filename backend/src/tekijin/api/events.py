"""LangGraph node updates -> SSE logical events (the mapping lives here only).

``graph.stream(stream_mode="updates")`` yields ``{node_name: partial_state}``
dicts plus a ``{"__interrupt__": (Interrupt,)}`` entry when the run pauses. This
module turns those into ``sse_starlette`` events per model-definition §4:

    c1_intent          -> event: understood
    ask (interrupt)    -> event: followup
    c5_route           -> event: route
    c6_score           -> event: recommend
    c7_draft           -> event: draft
    c8_update          -> event: done
    off_topic/document/unresolved_intent/no_candidate/no_expert -> event: message

All other nodes (reset, c2_sufficiency, c3_embed, c4_retrieve, prior_answer,
send, reroute) are internal and emit nothing.
"""

from __future__ import annotations

import json
from typing import Any

from sse_starlette import ServerSentEvent

from tekijin.api import schemas

# Terminal node -> the ``status`` reported on its ``message`` event.
_TERMINAL_STATUS: dict[str, str] = {
    "off_topic": "off_topic",
    "document": "document",
    "unresolved_intent": "unresolved",
    "no_candidate": "no_candidate",
    # #70: the answerability critic rejected — candidates scored, but their
    # in-house track record was judged insufficient (a graceful terminal, not a
    # hand-off). Only surfaces when the critic is wired (answerability_enabled).
    "no_expert": "no_expert",
    # #291: the assistant answered directly from retrieved data (past Q&A /
    # documents) with citations — no hand-off. Only when self_answer_enabled.
    "self_answered": "self_answered",
}

# Nodes that produce a visible SSE event (everything else is internal). Kept as a
# set for the single-source-of-truth of "which nodes surface to the client".
EVENT_NODES: frozenset[str] = frozenset(
    {"c1_intent", "c5_route", "c6_score", "c7_draft", "c8_update", *_TERMINAL_STATUS}
)

# Nodes whose event ends the run — ``done`` (c8_update) and the terminal messages.
# The service attaches ``latency_ms`` (this segment's processing time) to these.
TERMINAL_NODES: frozenset[str] = frozenset({"c8_update", *_TERMINAL_STATUS})


def _sse(event: str, data: schemas.BaseModel) -> ServerSentEvent:
    return ServerSentEvent(event=event, data=data.model_dump_json())


def node_event(
    node: str, update: dict[str, Any], *, latency_ms: int | None = None
) -> ServerSentEvent | None:
    """Map one node update to an SSE event, or ``None`` for internal nodes.

    ``latency_ms`` (this run segment's processing time) is attached to the terminal
    events (``done`` / ``message``) so the client can surface it; it is ignored for
    non-terminal nodes (#177).
    """

    update = update or {}
    if node == "c1_intent":
        return _sse(
            "understood",
            schemas.UnderstoodData(
                topics=update.get("topics", []),
                products=update.get("products", []),
                situation=update.get("situation"),
                question_type=update.get("question_type"),
                confidence=update.get("intent_confidence", 0.0),
                similar_asker_count=update.get("similar_asker_count", 0),
            ),
        )
    if node == "c5_route":
        return _sse(
            "route",
            schemas.RouteData(
                route=update.get("route", "person"),
                reason=update.get("route_reason", ""),
                confidence=update.get("route_confidence", 0.0),
            ),
        )
    if node == "additive_answer":
        # #413: additive cited answer on the person route. Emit a ``reference`` event
        # ONLY when one grounded (text present); otherwise this node is silent and the
        # run proceeds to the plain hand-off. It never replaces the recommend/hand-off.
        text = update.get("additive_answer_text")
        if not text:
            return None
        return _sse(
            "reference",
            schemas.ReferenceData(
                answer=text,
                citations=update.get("additive_citations") or [],
            ),
        )
    if node == "c6_score":
        # person_id crosses the boundary as the external "E###" string form
        # (model-definition §163-170), paired with the "E###" asker_id we accept.
        recs: list[Any] = [
            {**rec, "person_id": schemas.format_employee_id(rec["person_id"])}
            for rec in update.get("recommendations", [])
        ]
        return _sse("recommend", schemas.RecommendData(recommendations=recs))
    if node == "c7_draft":
        return _sse("draft", schemas.DraftData(draft=update.get("draft") or ""))
    if node == "c8_update":
        return _sse(
            "done",
            schemas.DoneData(status="sent", answer=update.get("answer"), latency_ms=latency_ms),
        )
    if node in _TERMINAL_STATUS:
        raw_fallback = update.get("fallback_responder")
        # Build the model here rather than handing MessageData a bare dict: the id
        # still has to be formatted, and constructing it eagerly means a malformed
        # fallback fails at the same place either way — but now with a type the
        # checker can follow (mirrors how service.py builds `responder`).
        fallback = (
            schemas.Recommendation(
                **{
                    **raw_fallback,
                    "person_id": schemas.format_employee_id(raw_fallback["person_id"]),
                }
            )
            if raw_fallback is not None
            else None
        )
        return _sse(
            "message",
            schemas.MessageData(
                status=_TERMINAL_STATUS[node],
                message=update.get("answer") or "",
                # Only the document terminal carries a doc id; harmless None elsewhere.
                doc_id=update.get("document_id"),
                fallback_responder=fallback,
                # #291: the self_answered terminal carries the sources it cited so
                # the chat renders a link per source; empty elsewhere.
                citations=update.get("self_answer_citations") or [],
                latency_ms=latency_ms,
            ),
        )
    return None  # internal node: no event


def progress_events(values: dict[str, Any]) -> list[ServerSentEvent]:
    """Rebuild, in graph order, the progress a run already produced (#512).

    The UI's thinking-progress list is derived entirely from ``understood`` /
    ``route`` / ``recommend`` / ``draft``. Those are streamed once, live — so a
    client that comes back to the session (from the history list, from the result
    screen, or by reloading) used to receive only the pending interrupt or the
    stored terminal, and rendered a screen with no progress on it at all. Every
    field is already in the durable checkpoint, so the sequence can simply be
    rebuilt from it: read-only, no re-run.

    **Emit only what actually ran.** ``reset`` seeds `route`/`recommendations`/
    `draft` with defaults BEFORE C1, so "emit whatever the state holds" would
    announce 「経路を判断しました: 人に取り次ぐ」 on a question that was never
    routed. Each source therefore has a marker that separates "did not run" from
    "ran and produced nothing":

    * ``route`` — ``route_reason``. ``reset`` leaves it ``""`` and every
      ``RouteDecision`` carries a non-empty reason, so non-empty ⟺ C5 ran.
    * ``draft`` — ``draft is not None``. ``reset`` leaves it ``None`` and C7
      always writes a string (possibly empty), mirroring the live event, which
      is emitted even for an empty draft.
    * ``recommend`` — a non-empty list. C6 writing ``[]`` (the no_candidate case)
      is indistinguishable from not having run, and the live empty ``recommend``
      only precedes a terminal ``message`` that says the same thing, so it is not
      worth replaying either way.
    * ``reference`` (#413) — non-empty text, exactly as the live node's own gate.

    ``understood`` is unconditional: every path that replays (a finished run, or
    one paused at ``ask``/``send``) is downstream of C1 by construction.
    """

    out: list[ServerSentEvent] = [
        _sse(
            "understood",
            schemas.UnderstoodData(
                topics=values.get("topics", []),
                products=values.get("products", []),
                situation=values.get("situation"),
                question_type=values.get("question_type"),
                confidence=values.get("intent_confidence", 0.0),
                similar_asker_count=values.get("similar_asker_count", 0),
            ),
        )
    ]
    if values.get("route_reason"):
        out.append(
            _sse(
                "route",
                schemas.RouteData(
                    route=values.get("route", "person"),
                    reason=values.get("route_reason", ""),
                    confidence=values.get("route_confidence", 0.0),
                ),
            )
        )
    if values.get("additive_answer_text"):
        out.append(
            _sse(
                "reference",
                schemas.ReferenceData(
                    answer=values.get("additive_answer_text") or "",
                    citations=values.get("additive_citations") or [],
                ),
            )
        )
    recs = values.get("recommendations") or []
    if recs:
        # person_id -> external "E###" form, mirroring the live recommend event.
        ext: list[Any] = [
            {**rec, "person_id": schemas.format_employee_id(rec["person_id"])} for rec in recs
        ]
        out.append(_sse("recommend", schemas.RecommendData(recommendations=ext)))
    if values.get("draft") is not None:
        out.append(_sse("draft", schemas.DraftData(draft=values.get("draft") or "")))
    return out


def reconnect_events(next_node: str, values: dict[str, Any]) -> list[ServerSentEvent]:
    """Re-emit the pending INTERRUPT event when a client reconnects to /events.

    A session paused at ``ask`` re-sends the ``followup`` (from the saved state).
    One paused at ``send`` needs no event of its own — the ``recommend`` and
    ``draft`` that reconstruct the hand-off are part of the run's progress and
    come from :func:`progress_events`, which the caller emits first (#512). That
    still satisfies the guarantee this used to carry on its own: a client that
    reconnects after another consumer drained the live segment gets the
    candidates, which ``ResultScreen`` reads only from ``recommend``.

    Any other pending node yields nothing.
    """

    if next_node == "ask":
        return [
            _sse(
                "followup",
                schemas.FollowupData(
                    question=values.get("followup_question") or "",
                    missing=values.get("missing", []),
                ),
            )
        ]
    return []


# SSE event names that represent a *terminal* run outcome (worth replaying on a
# reconnect after the run has already finished).
TERMINAL_EVENTS: frozenset[str] = frozenset({"done", "message"})


def replay_terminal(values: dict[str, Any]) -> ServerSentEvent | None:
    """Re-emit the stored terminal event (``done`` / ``message``) on reconnect.

    A run that finished commits its final event into ``last_event`` (see the
    service). If a client disconnected before receiving it, reconnecting to
    /events replays it verbatim — read-only, no re-run, no double insert. Returns
    ``None`` when there is no stored terminal event (a genuinely unknown session).
    """

    last = values.get("last_event")
    if not last:
        return None
    return ServerSentEvent(event=last["event"], data=last["data"])


def interrupt_event(payload: dict[str, Any]) -> ServerSentEvent | None:
    """Map an ``interrupt`` payload to an SSE event.

    A ``followup_question`` payload is the C2 clarification -> ``followup``. The
    ``send`` interrupt (draft/responder payload) needs no event: the ``draft``
    was already emitted and the client now POSTs an outcome to /answer.
    """

    if payload and "followup_question" in payload:
        return _sse(
            "followup",
            schemas.FollowupData(
                question=payload.get("followup_question") or "",
                missing=payload.get("missing", []),
            ),
        )
    return None


def without_reasons(event: ServerSentEvent) -> ServerSentEvent:
    """``event`` with every recommendation's ``reasons`` emptied.

    For a member's question the reasons come from evidence about the person
    (introductions, conversations…), shown only to the asker (ADR-0012, #107). A
    responder may open the same stream, so what reaches them goes through this.
    Events carrying no recommendation pass unchanged."""

    if event.event not in {"recommend", "message"} or not isinstance(event.data, str):
        return event
    try:
        data = json.loads(event.data)
    except ValueError:
        return event
    if not isinstance(data, dict):
        return event
    recs = data.get("recommendations")
    if isinstance(recs, list):
        data["recommendations"] = [{**r, "reasons": []} if isinstance(r, dict) else r for r in recs]
    fallback = data.get("fallback_responder")
    if isinstance(fallback, dict):
        data["fallback_responder"] = {**fallback, "reasons": []}
    return ServerSentEvent(event=event.event, data=json.dumps(data, ensure_ascii=False))

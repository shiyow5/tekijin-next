"""Collapse one run's SSE events into a single chat-friendly outcome (Discord D1, #60).

The web client renders the event stream step by step. A chat command gets ONE reply,
so this reads the same events (``api/events.py``) and keeps only what the reply needs:
the answer and its sources, the recommended people, or the clarification question.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

from tekijin.api.schemas import Recommendation, SourceCitation

AskKind = Literal["answered", "handoff", "followup", "no_answer", "error", "unlinked"]

# Terminal ``message`` statuses that carry an answer the asker can use as-is.
_ANSWER_STATUSES = frozenset({"self_answered", "document"})


class AskOutcome(BaseModel):
    kind: AskKind
    text: str = ""
    citations: list[SourceCitation] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)


def summarize_events(events: Iterable[tuple[str, dict[str, Any]]]) -> AskOutcome:
    """Reduce ``(event, data)`` pairs from one run segment to an :class:`AskOutcome`.

    A segment ends in exactly one of: a terminal ``message``, ``done``, a ``followup``
    interrupt, the ``send`` interrupt after ``recommend``/``draft`` (no event of its
    own), or ``error``. An empty stream means the run never produced anything usable.
    """

    reference = ""
    reference_citations: list[SourceCitation] = []
    recommendations: list[Recommendation] = []
    for event, data in events:
        if event == "error":
            return AskOutcome(kind="error")
        if event == "followup":
            return AskOutcome(kind="followup", text=data.get("question") or "")
        if event == "message":
            return _from_message(data)
        if event == "done":
            return AskOutcome(kind="answered", text=data.get("answer") or "")
        if event == "reference":
            reference = data.get("answer") or ""
            reference_citations = [SourceCitation(**c) for c in data.get("citations") or []]
        elif event == "recommend":
            recommendations = [Recommendation(**r) for r in data.get("recommendations", [])]
    if recommendations:
        return AskOutcome(
            kind="handoff",
            text=reference,
            citations=reference_citations,
            recommendations=recommendations,
        )
    return AskOutcome(kind="error")


def _from_message(data: dict[str, Any]) -> AskOutcome:
    status = data.get("status") or ""
    text = data.get("message") or ""
    if status not in _ANSWER_STATUSES:
        return AskOutcome(kind="no_answer", text=text)
    citations = [SourceCitation(**c) for c in data.get("citations") or []]
    doc_id = data.get("doc_id")
    if doc_id and all(c.source_id != doc_id for c in citations):
        citations.append(SourceCitation(source_id=doc_id, kind="document"))
    return AskOutcome(kind="answered", text=text, citations=citations)

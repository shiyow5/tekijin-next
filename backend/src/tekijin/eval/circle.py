"""Run the fictional circle's questions through the graph and score them (#108).

``scripts/circle_eval.py`` is the command-line entry point: it builds the graph the
API builds (real LLM nodes, ``ChunkSearch``, ``DbExpertFinder``) over the circle the
database holds and prints :func:`format_summary`. The pieces live here so the
scoring is tested without a model.

What each row records, per the design doc §4.7:

* ``out_of_scope`` — C1 refused the question.
* ``seeks_person`` — C1 read the question as asking for a person (#108).
* ``followup`` — the run paused at ``ask`` (a follow-up question is pending).
* ``handoff`` — the run paused at ``send`` (a hand-off draft is waiting).
* ``grounded`` — the run ended with a cited answer from the sources (#291).
* ``additive`` — a cited answer was shown next to a hand-off (#413).
* ``cite_ok`` — the answer shown cites at least one of the expected sources.
* ``contains`` — the answer shown contains every expected phrase.
* ``hit_at_3`` — for a ``person`` question, a correct member is among the first
  three recommendations.
* ``knowledge`` — the answer came from verified knowledge (#113). Its citations
  name the documents (``src_``) and Discord messages (``msg_``) the knowledge was
  made from; a message counts as the conversation it belongs to.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from tekijin.agent.graph import build_agent
from tekijin.config import Settings
from tekijin.data.circle import GUILD_ID, Circle, external_id
from tekijin.data.circle import _discord_id as circle_discord_id
from tekijin.data.conversations import conversation_key
from tekijin.llm.factory import make_llm_nodes
from tekijin.models.tables import ChannelMessage, Identity, Organization, SourceItem
from tekijin.retrieval.embedding import Embedder
from tekijin.retrieval.experts import DbExpertFinder
from tekijin.retrieval.sources import ChunkSearch, source_citation_id

KINDS = ("answer", "abstain", "person")
_COUNTED = (
    "out_of_scope",
    "seeks_person",
    "followup",
    "handoff",
    "grounded",
    "additive",
    "cite_ok",
    "contains",
    "hit_at_3",
    "knowledge",
)


@dataclass(frozen=True, slots=True)
class CircleIds:
    """How the fixture's references map onto the stored rows."""

    # A question's ``sources`` reference ("doc:…" / "conv:…") -> "src_<id>".
    citation_ids: Mapping[str, str]
    # Stored member id -> fixture member key ("m04").
    member_keys: Mapping[int, str]
    # A knowledge answer's message citation ("msg_<id>") -> its conversation's
    # "src_<id>" (#113).
    message_sources: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RowResult:
    id: str
    kind: str
    route: str | None
    out_of_scope: bool
    seeks_person: bool
    followup: bool
    handoff: bool
    grounded: bool
    additive: bool
    cite_ok: bool
    contains: bool
    hit_at_3: bool
    top3: tuple[str, ...]
    answer: str
    knowledge: bool = False


def read_only(session: Session) -> None:
    """Make the session's transaction refuse every write (the caller rolls back)."""

    session.execute(text("SET TRANSACTION READ ONLY"))


def circle_org_id(session: Session) -> int:
    """The id of the organization the circle was seeded into."""

    org_id: int = session.scalars(
        select(Organization.id).where(Organization.discord_guild_id == GUILD_ID)
    ).one()
    return org_id


def build_circle_agent(
    session: Session,
    embedder: Embedder,
    settings: Settings,
    nodes: tuple[Any, ...] | None = None,
    *,
    expert_finder: Any | None = None,
    knowledge_floor: float | None = None,
) -> Any:
    """The graph the API builds for a member's question, on a ``MemorySaver``.

    Mirrors ``api/factory.py`` and ``AgentService._graph`` for the switches that
    reach a member's question; ``nodes`` defaults to the configured LLM backend and
    ``expert_finder`` to ``DbExpertFinder`` with its calibrated floors.
    ``knowledge_floor`` wires the knowledge answer with that floor (#113), as
    ``knowledge_retrieval_enabled`` does in the API; ``None`` leaves it out.
    """

    intent, sufficiency, draft, answerability, self_answer = nodes or make_llm_nodes(settings)
    return build_agent(
        embedder,
        session,
        checkpointer=MemorySaver(),
        intent_model=intent,
        sufficiency_model=sufficiency,
        draft_model=draft,
        answerability_model=answerability if settings.answerability_enabled else None,
        answerability_threshold=settings.answerability_threshold,
        self_answer_model=self_answer if settings.self_answer_enabled else None,
        bm25_weight=settings.bm25_weight,
        prior_answer_reuse_min=settings.prior_answer_reuse_min,
        prior_answer_relevance_floor=settings.prior_answer_relevance_floor,
        query_expansion_enabled=settings.query_expansion_enabled,
        question_fit_enabled=settings.question_fit_enabled,
        additive_self_answer_enabled=settings.additive_self_answer_enabled,
        additive_self_answer_floor=settings.additive_self_answer_floor,
        similar_askers_enabled=settings.similar_askers_enabled,
        source_search=ChunkSearch(session),
        expert_finder=expert_finder or DbExpertFinder(session),
        knowledge_answer_min_similarity=knowledge_floor,
    )


def circle_ids(session: Session, circle: Circle, org_id: int) -> CircleIds:
    """Read the stored ids of the circle's sources and members (read only)."""

    stored = {
        row.external_id: row.id
        for row in session.scalars(select(SourceItem).where(SourceItem.org_id == org_id))
    }
    refs = {ref for q in circle.questions for ref in q.get("sources", [])}
    citation_ids = {
        ref: source_citation_id(stored[external_id(circle, ref)])
        for ref in refs
        if external_id(circle, ref) in stored
    }
    by_discord = {
        row.external_id: row.employee_id
        for row in session.scalars(select(Identity).where(Identity.org_id == org_id))
    }
    member_keys = {
        by_discord[circle_discord_id(m["key"])]: m["key"]
        for m in circle.members
        if circle_discord_id(m["key"]) in by_discord
    }
    messages = session.execute(
        select(
            ChannelMessage.id,
            ChannelMessage.channel_id,
            ChannelMessage.thread_id,
            ChannelMessage.created_at,
        ).where(ChannelMessage.org_id == org_id)
    ).all()
    message_sources = {
        f"msg_{mid}": source_citation_id(stored[key])
        for mid, channel, thread, at in messages
        if (key := conversation_key(channel, thread, at)) in stored
    }
    return CircleIds(
        citation_ids=citation_ids, member_keys=member_keys, message_sources=message_sources
    )


def score_row(
    question: Mapping[str, Any],
    values: Mapping[str, Any],
    next_nodes: Sequence[str],
    ids: CircleIds,
) -> RowResult:
    """Score one finished (or paused) run against the question's expectations."""

    grounded = bool(values.get("self_answer_grounded"))
    additive_text = values.get("additive_answer_text") or ""
    if grounded:
        shown, citations = values.get("answer") or "", values.get("self_answer_citations") or []
    else:
        shown, citations = additive_text, values.get("additive_citations") or []
    cited = {ids.message_sources.get(c.get("source_id"), c.get("source_id")) for c in citations}
    wanted = {ids.citation_ids[r] for r in question.get("sources", []) if r in ids.citation_ids}
    phrases = question.get("contains", [])
    top3 = tuple(
        ids.member_keys.get(r["person_id"], str(r["person_id"]))
        for r in (values.get("recommendations") or [])[:3]
    )
    return RowResult(
        id=str(question["id"]),
        kind=str(question["kind"]),
        route=values.get("route"),
        out_of_scope=bool(values.get("out_of_scope")),
        seeks_person=bool(values.get("seeks_person")),
        followup=tuple(next_nodes) == ("ask",),
        handoff=tuple(next_nodes) == ("send",),
        grounded=grounded,
        additive=bool(additive_text),
        cite_ok=bool(cited & wanted),
        contains=bool(shown) and bool(phrases) and all(p in shown for p in phrases),
        hit_at_3=bool(set(top3) & set(question.get("members", []))),
        top3=top3,
        answer=(values.get("answer") or additive_text or "").replace("\n", " "),
        knowledge=bool(values.get("knowledge_article_ids")),
    )


def run_circle(
    agent: Any,
    circle: Circle,
    org_id: int,
    ids: CircleIds,
    *,
    now: dt.datetime,
    asker: dict[str, Any] | None = None,
    thread_prefix: str = "circle",
) -> list[RowResult]:
    """Ask every question once, in order, each on its own thread."""

    rows = []
    for question in circle.questions:
        config = {"configurable": {"thread_id": f"{thread_prefix}-{question['id']}"}}
        agent.invoke(
            {"question": question["question"], "asker": asker, "org_id": org_id, "now": now},
            config,
        )
        # A paused run (ask / send) returns only part of the state from invoke, so
        # the checkpoint is read for the whole of it.
        snapshot = agent.get_state(config)
        rows.append(score_row(question, snapshot.values, snapshot.next, ids))
    return rows


def summarize(rows: Iterable[RowResult]) -> dict[str, dict[str, Any]]:
    """Per kind: the row count, each counted flag, and how the routes split."""

    every = list(rows)
    out: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        mine = [r for r in every if r.kind == kind]
        out[kind] = {
            "n": len(mine),
            **{flag: sum(bool(getattr(r, flag)) for r in mine) for flag in _COUNTED},
            "routes": dict(sorted(Counter(r.route or "-" for r in mine).items())),
        }
    return out


def format_summary(summary: Mapping[str, Mapping[str, Any]]) -> str:
    """A plain table, one line per kind."""

    head = ["kind", "n", *_COUNTED, "routes"]
    lines = [" | ".join(head)]
    for kind, row in summary.items():
        routes = ", ".join(f"{k}={v}" for k, v in row["routes"].items())
        lines.append(" | ".join([kind, str(row["n"]), *(str(row[f]) for f in _COUNTED), routes]))
    return "\n".join(lines)


def format_rows(rows: Iterable[RowResult]) -> str:
    """One line per question, for explaining a changed row."""

    def flag(value: bool, mark: str) -> str:
        return mark if value else "-"

    lines = []
    for r in rows:
        flags = "".join(
            [
                flag(r.out_of_scope, "O"),
                flag(r.seeks_person, "S"),
                flag(r.followup, "F"),
                flag(r.handoff, "H"),
                flag(r.grounded, "G"),
                flag(r.additive, "A"),
                flag(r.cite_ok, "C"),
                flag(r.contains, "P"),
                flag(r.hit_at_3, "R"),
                flag(r.knowledge, "K"),
            ]
        )
        lines.append(
            f"{r.id} {r.kind:<7} {r.route or '-':<9} {flags} top3={','.join(r.top3) or '-'}"
            f" {r.answer[:60]}"
        )
    return "\n".join(lines)

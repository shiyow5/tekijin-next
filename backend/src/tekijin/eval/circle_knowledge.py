"""Build the circle's knowledge for measuring the knowledge answer (#113).

``scripts/circle_eval.py --build-knowledge`` runs this with the real LLM: every
document of the circle is drafted into procedures and facts, and every message
that reads as the asker closing their question is tried as a solved conversation
(the same rule the live bot uses; the fixtures carry no reactions). The drafts are
then marked verified, which only an admin does in real use, so the measurement
shows what the knowledge answer does once someone has checked them. The ``wip``
drafts (no answer) stay unverified.

Everything here writes, and only to the circle's organization: :func:`circle_org`
refuses any other, and :func:`reset_circle_knowledge` deletes only that
organization's knowledge. :func:`floor_sweep` reads only, and shows how many
questions of each kind verified knowledge would take at each floor without
running the graph.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from tekijin.data.circle import GUILD_ID, ORG_NAME, Circle
from tekijin.data.knowledge import article_sources, search_verified, verify_article
from tekijin.data.visibility import PUBLIC
from tekijin.eval.circle import KINDS, CircleIds
from tekijin.knowledge.conversations import draft_conversation_case
from tekijin.knowledge.documents import DocumentExtractor, document_item_ids, draft_document
from tekijin.knowledge.extract import Extractor
from tekijin.knowledge.index import embed_articles
from tekijin.knowledge.solve_signals import is_closing_utterance
from tekijin.models.tables import ChannelMessage, KnowledgeArticle, Organization
from tekijin.retrieval.embedding import QUERY, Embedder
from tekijin.retrieval.sources import source_citation_id

#: Floors :func:`floor_sweep` tries by default.
DEFAULT_FLOORS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


class NotTheCircle(RuntimeError):
    """The organization is not the fictional circle; nothing is written."""


@dataclass(frozen=True, slots=True)
class BuildReport:
    documents: int = 0
    document_drafts: int = 0
    document_errors: int = 0
    signals: int = 0
    cases: int = 0
    case_errors: int = 0
    verified: int = 0
    wip: int = 0


def circle_org(session: Session) -> int:
    """The circle's organization id, after checking it is the circle."""

    org = session.scalars(
        select(Organization).where(Organization.discord_guild_id == GUILD_ID)
    ).first()
    if org is None or org.name != ORG_NAME:
        raise NotTheCircle("the circle is not seeded (python -m tekijin.data.circle)")
    return org.id


def reset_circle_knowledge(session: Session) -> int:
    """Delete the circle's knowledge (and only the circle's). Returns how many."""

    org_id = circle_org(session)
    result = session.execute(delete(KnowledgeArticle).where(KnowledgeArticle.org_id == org_id))
    return int(result.rowcount or 0)  # type: ignore[attr-defined]


def build_circle_knowledge(
    session: Session,
    embedder: Embedder,
    *,
    case_extractor: Extractor,
    document_extractor: DocumentExtractor,
    now: dt.datetime,
    verify: bool = True,
) -> BuildReport:
    """Draft the circle's documents and solved conversations, verify, embed."""

    org_id = circle_org(session)
    documents = document_item_ids(session, org_id=org_id)
    document_drafts = document_errors = 0
    for item_id in documents:
        report = draft_document(
            session, org_id=org_id, item_id=item_id, extractor=document_extractor, now=now
        )
        document_drafts += report.stored
        document_errors += report.errored
    messages = session.execute(
        select(ChannelMessage.message_id, ChannelMessage.content)
        .where(ChannelMessage.org_id == org_id, ChannelMessage.hidden_at.is_(None))
        .order_by(ChannelMessage.created_at, ChannelMessage.message_id)
    ).all()
    signals = [m for m, content in messages if is_closing_utterance(content)]
    cases = case_errors = 0
    for message_id in signals:
        found = draft_conversation_case(
            session, org_id=org_id, message_id=message_id, extractor=case_extractor, now=now
        )
        if found is not None:
            cases += found.stored
            case_errors += found.errored
    verified = wip = 0
    for row in session.scalars(
        select(KnowledgeArticle).where(KnowledgeArticle.org_id == org_id)
    ).all():
        if row.state == "draft" and verify:
            verify_article(session, row.id, org_id=org_id, verifier_id=None, now=now)
            verified += 1
        elif row.state == "wip":
            wip += 1
    ids = list(
        session.scalars(select(KnowledgeArticle.id).where(KnowledgeArticle.org_id == org_id))
    )
    embed_articles(session, embedder, article_ids=ids, only_missing=False)
    return BuildReport(
        documents=len(documents),
        document_drafts=document_drafts,
        document_errors=document_errors,
        signals=len(signals),
        cases=cases,
        case_errors=case_errors,
        verified=verified,
        wip=wip,
    )


@dataclass(frozen=True, slots=True)
class SweepRow:
    floor: float
    # Per question kind: how many questions the top verified article would take.
    taken: Mapping[str, int]
    # Of the ``answer`` questions taken, how many cite an expected source.
    cite_ok: int


def _article_citations(session: Session, org_id: int, article_id: int, ids: CircleIds) -> set[str]:
    sources = article_sources(session, [article_id], org_id=org_id, scope=PUBLIC)
    cited: set[str] = set()
    for source in sources.get(article_id, []):
        if source.source_item_id is not None:
            cited.add(source_citation_id(source.source_item_id))
        elif source.channel_message_id is not None:
            key = f"msg_{source.channel_message_id}"
            cited.add(ids.message_sources.get(key, key))
    return cited


def floor_sweep(
    session: Session,
    embedder: Embedder,
    circle: Circle,
    ids: CircleIds,
    *,
    floors: Iterable[float] = DEFAULT_FLOORS,
    top_n: int = 3,
) -> list[SweepRow]:
    """For each floor, the questions verified knowledge would answer (read only).

    A question is taken when its nearest verified article clears the floor, as in
    the graph's knowledge step (which also skips questions C1 reads as asking for
    a person, so ``person`` here is the risk before that guard). ``cite_ok``
    counts ``answer`` questions whose taken articles (up to ``top_n``) cite one of
    the expected sources.
    """

    org_id = circle_org(session)
    questions = list(circle.questions)
    vectors = embedder.encode([q["question"] for q in questions], kind=QUERY)
    nearest: list[tuple[str, list[tuple[int, float]], set[str]]] = []
    for question, vector in zip(questions, vectors, strict=True):
        hits = search_verified(session, org_id=org_id, query_vec=vector, scope=PUBLIC, top_k=top_n)
        wanted = {ids.citation_ids[r] for r in question.get("sources", []) if r in ids.citation_ids}
        nearest.append((question["kind"], [(a.id, sim) for a, sim in hits], wanted))
    citations: dict[int, set[str]] = {}
    rows = []
    for floor in floors:
        taken = dict.fromkeys(KINDS, 0)
        cite_ok = 0
        for kind, scored, wanted in nearest:
            chosen = [aid for aid, sim in scored if sim >= floor]
            if not chosen:
                continue
            taken[kind] = taken.get(kind, 0) + 1
            if kind == "answer":
                cited: set[str] = set()
                for aid in chosen:
                    if aid not in citations:
                        citations[aid] = _article_citations(session, org_id, aid, ids)
                    cited |= citations[aid]
                cite_ok += bool(cited & wanted)
        rows.append(SweepRow(floor=floor, taken=taken, cite_ok=cite_ok))
    return rows


def format_sweep(rows: Sequence[SweepRow]) -> str:
    head = "floor | answer taken | answer cite_ok | abstain taken | person taken"
    lines = [head]
    for r in rows:
        lines.append(
            f"{r.floor:.2f} | {r.taken.get('answer', 0)} | {r.cite_ok} | "
            f"{r.taken.get('abstain', 0)} | {r.taken.get('person', 0)}"
        )
    return "\n".join(lines)


def format_build(report: BuildReport) -> str:
    return (
        f"documents={report.documents} document_drafts={report.document_drafts} "
        f"document_errors={report.document_errors} signals={report.signals} "
        f"cases={report.cases} case_errors={report.case_errors} "
        f"verified={report.verified} wip={report.wip}"
    )


def as_dict(report: BuildReport) -> dict[str, Any]:
    return {f: getattr(report, f) for f in BuildReport.__dataclass_fields__}

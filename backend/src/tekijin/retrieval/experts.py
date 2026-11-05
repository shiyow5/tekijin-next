"""Who in an organization can answer a question: evidence first, then its authors (#107, R5).

The company scorer (``scorer/``) needs a roster, certifications and a topic
vocabulary, none of which an organization has. Instead, the evidence closest to
the question is found first and credit goes to whoever wrote it, as in Balog et
al.'s Model 2 (SIGIR 2006, https://doi.org/10.1145/1148170.1148181). Four kinds
of evidence are used:

* ``self_intro``: a member's self-introduction (``data/self_intros.py``).
* ``document``: a document C4 found, credited to its author.
* ``conversation``: a Discord conversation C4 found, credited to the people who
  spoke in it by their share of its characters. Only when the organization's
  ``expertise_from_messages`` is on (ADR-0012).
* ``answer``: an answer given through TEKIJIN to a member of the organization.

Each member's score is ``sum(weight[kind] * share / (RRF_K + rank))`` over their
evidence, where ``rank`` is the evidence's place in its own kind's list. Rank-based
credit carries no notion of "close enough", so an evidence item counts only when
its cosine similarity to the question reaches its kind's floor (``FLOORS``);
otherwise an unrelated question would still recommend whoever ranked first.

The asker, members who opted out or left, and anyone excluded by the caller
(declined, already shown) are never ranked. What the reader's scope hides is never
read: C4's hits are already filtered, and self-introductions are searched with
the same scope, so hidden evidence neither counts nor appears in a reason.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, aliased

from tekijin.data.conversations import DISCORD, conversation_filter
from tekijin.data.self_intros import SLIDES, is_intro_document, search_self_intros
from tekijin.data.visibility import SourceScope, visible
from tekijin.models.tables import (
    Answer,
    ChannelMessage,
    Chunk,
    Employee,
    Organization,
    Question,
    SourceItem,
)
from tekijin.retrieval.fragments import clip_text
from tekijin.retrieval.sources import source_citation_id

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids agent<-retrieval cycle)
    from tekijin.agent.state import SourceHit

SELF_INTRO = "self_intro"
DOCUMENT = "document"
CONVERSATION = "conversation"
ANSWER = "answer"

# Balog-style rank credit, the same k as RRF elsewhere.
RRF_K = 60
# How much one piece of each kind counts. A self-introduction, a document and a
# TEKIJIN answer each say directly what a person knows; a conversation is mostly
# small talk, and the person may only have asked, so it counts less.
WEIGHTS: Mapping[str, float] = MappingProxyType(
    {SELF_INTRO: 1.0, DOCUMENT: 1.0, CONVERSATION: 0.5, ANSWER: 1.0}
)
# The cosine similarity an evidence item needs to count. Measured on evo-x2 with
# Nemotron-3-Embed-1B on the fictional circle's 36 questions (#107, 2026-11-05):
#
# * self_intro 0.20: the correct member's introduction scored 0.213-0.328 on the 8
#   person questions; the best introduction on the 6 abstain questions scored
#   0.024-0.198. Any floor in 0.18-0.21 kept recall@3 at 8/8. The margin is
#   thin, so re-measure on a real organization (R7).
# * document 0.25: the correct author's document scored 0.259-0.485; the abstain
#   questions' best documents scored 0.065-0.480, so no floor separates them.
#   0.24-0.30 all kept 8/8; 0.25 drops the four unrelated abstain questions.
# * conversation 0.32: with expertise_from_messages on, 0.32-0.34 kept 8/8 and
#   recommended someone on 2 of 6 abstain questions; with no floor, small talk
#   pushed a correct member to 4th (7/8).
# * answer 0.20: not measured (the circle has no TEKIJIN answers yet); set with
#   self_intro, the other kind written by a person about what they know.
FLOORS: Mapping[str, float] = MappingProxyType(
    {SELF_INTRO: 0.20, DOCUMENT: 0.25, CONVERSATION: 0.32, ANSWER: 0.20}
)
MAX_REASONS = 3
# How many self-introductions and TEKIJIN answers are read per question.
INTRO_TOP_K = 20
ANSWER_TOP_K = 20
_INTRO_CHARS = 60
_TITLE_CHARS = 40
# What an ``answer`` reason says. Never the question or the answer itself.
ANSWER_REASON = "TEKIJINで似た質問に答えた"
# The web gauge shows ``score / 1.8`` (frontend/src/lib/fit.ts), the company
# scorer's ceiling. A member whose evidence is first in two kinds fills it.
_SCORE_CEILING = 1.8
_FULL_SCORE = 2 * max(WEIGHTS.values()) / (RRF_K + 1)
# Kinds whose cosine is a signal about people rather than about the documents.
_PERSON_SIGNAL_KINDS = frozenset({SELF_INTRO, ANSWER})
# Kinds that make a recommendation 高: the person wrote the document or answered.
_STRONG_KINDS = frozenset({DOCUMENT, ANSWER})
_KIND_ORDER = {SELF_INTRO: 0, DOCUMENT: 1, ANSWER: 2, CONVERSATION: 3}


@dataclass(frozen=True, slots=True)
class Evidence:
    """One piece of evidence for one member."""

    kind: str
    member_id: int
    # 1-based place in this kind's list for the question.
    rank: int
    # Cosine similarity of the evidence to the question.
    cosine: float
    # What the asker is shown: ``type``, ``detail`` and, for a source, ``source_id``.
    reason: Mapping[str, str]
    # The member's part of the credit (a conversation is split among its speakers).
    share: float = 1.0


@dataclass(frozen=True, slots=True)
class Expert:
    """One ranked member."""

    member_id: int
    name: str
    dept: str | None
    raw_score: float
    reasons: tuple[Mapping[str, str], ...]
    kinds: frozenset[str]

    def as_recommendation(self) -> dict[str, Any]:
        """The shape C6's scorer returns, read by C7, the API and Discord."""

        return {
            "person_id": self.member_id,
            "name": self.name,
            "dept": self.dept,
            "score": round(_SCORE_CEILING * min(self.raw_score / _FULL_SCORE, 1.0), 4),
            "confidence": "高" if self.kinds & _STRONG_KINDS else "低",
            "reasons": [dict(reason) for reason in self.reasons],
        }


@dataclass(frozen=True, slots=True)
class ExpertResult:
    experts: list[Expert]
    # C5 reads this as ``people_confidence``: the best cosine of a counted
    # self-introduction or TEKIJIN answer. Documents and conversations are left
    # out: they are C4's own hits, whose cosine already is ``document_confidence``,
    # and counting them again would turn every well-documented question into a
    # hand-off instead of an answer from the documents (measured: 12 of the
    # circle's 28 answer/abstain questions changed route when they counted).
    confidence: float


def rank_experts(
    evidence: Iterable[Evidence],
    members: Mapping[int, tuple[str, str | None]],
    *,
    weights: Mapping[str, float] = WEIGHTS,
    floors: Mapping[str, float] = FLOORS,
    k: int = RRF_K,
) -> ExpertResult:
    """Rank ``members`` (id -> (name, dept)) by the evidence that counts for them.

    Evidence of anyone not in ``members``, or below its kind's floor, is ignored.
    Best first; ties break on the member id.
    """

    totals: dict[int, float] = {}
    credited: dict[int, list[tuple[float, Evidence]]] = {}
    best = 0.0
    for item in evidence:
        if item.member_id not in members or item.cosine < floors.get(item.kind, 0.0):
            continue
        credit = weights.get(item.kind, 0.0) * item.share / (k + item.rank)
        totals[item.member_id] = totals.get(item.member_id, 0.0) + credit
        credited.setdefault(item.member_id, []).append((credit, item))
        if item.kind in _PERSON_SIGNAL_KINDS:
            best = max(best, item.cosine)
    experts = [
        Expert(
            member_id=member_id,
            name=members[member_id][0],
            dept=members[member_id][1],
            raw_score=total,
            reasons=_reasons(credited[member_id]),
            kinds=frozenset(item.kind for _, item in credited[member_id]),
        )
        for member_id, total in totals.items()
    ]
    experts.sort(key=lambda e: (-e.raw_score, e.member_id))
    return ExpertResult(experts=experts, confidence=best)


def _reasons(credited: Sequence[tuple[float, Evidence]]) -> tuple[Mapping[str, str], ...]:
    ordered = sorted(credited, key=lambda c: (-c[0], _KIND_ORDER.get(c[1].kind, 9), c[1].rank))
    reasons: list[Mapping[str, str]] = []
    seen: set[str] = set()
    for _, item in ordered:
        detail = item.reason.get("detail", "")
        if detail in seen:
            continue
        seen.add(detail)
        reasons.append(MappingProxyType(dict(item.reason)))
        if len(reasons) == MAX_REASONS:
            break
    return tuple(reasons)


def intro_excerpt(text: str, *, origin: str) -> str:
    """What a reason quotes of a self-introduction. A slide's first line is its
    heading (「スライド4 岩田颯太(機材係)」), which says nothing the name does not."""

    lines = text.strip().split("\n")
    if origin == SLIDES and len(lines) > 1:
        lines = lines[1:]
    return clip_text(" ".join(lines), _INTRO_CHARS)


# --------------------------------------------------------------------------- #
# Reading the evidence from the database
# --------------------------------------------------------------------------- #
class DbExpertFinder:
    """:class:`~tekijin.agent.protocols.ExpertFinder` over one session."""

    def __init__(
        self,
        session: Session,
        *,
        weights: Mapping[str, float] = WEIGHTS,
        floors: Mapping[str, float] = FLOORS,
        k: int = RRF_K,
        expertise_from_messages: bool | None = None,
    ) -> None:
        self._session = session
        self._weights = weights
        self._floors = floors
        self._k = k
        # ``None`` reads the organization's setting; a bool overrides it (measurement).
        self._from_messages = expertise_from_messages

    def find(
        self,
        *,
        org_id: int,
        question: str,
        query_vector: Sequence[float],
        scope: SourceScope,
        source_hits: Sequence[SourceHit],
        asker_id: int | None,
        exclude: Iterable[int] = (),
    ) -> ExpertResult:
        members = self._members(org_id, {*exclude, *([asker_id] if asker_id is not None else [])})
        if not members:
            return ExpertResult(experts=[], confidence=0.0)
        # People left out could fill the top of a kind's list, so introductions and
        # answers are read only from people who may be recommended.
        evidence = [
            *self._intros(org_id, question, query_vector, scope, members),
            *self._sources(org_id, query_vector, scope, source_hits),
            *self._answers(org_id, query_vector, members),
        ]
        return rank_experts(
            evidence, members, weights=self._weights, floors=self._floors, k=self._k
        )

    def describe(self, org_id: int, member_ids: Iterable[int]) -> list[Expert]:
        """Members who may be recommended, with no evidence (a pinned referral)."""

        ids = set(member_ids)
        return [
            Expert(member_id=m, name=name, dept=dept, raw_score=0.0, reasons=(), kinds=frozenset())
            for m, (name, dept) in self._members(org_id, set()).items()
            if m in ids
        ]

    # -- members ------------------------------------------------------------- #
    def _members(self, org_id: int, excluded: set[int]) -> dict[int, tuple[str, str | None]]:
        rows = self._session.execute(
            select(Employee.id, Employee.name, Employee.department).where(
                Employee.org_id == org_id,
                Employee.opted_out_at.is_(None),
                Employee.is_active.is_(True),
            )
        ).all()
        return {row.id: (row.name, row.department) for row in rows if row.id not in excluded}

    # -- self-introductions -------------------------------------------------- #
    def _intros(
        self,
        org_id: int,
        question: str,
        query_vector: Sequence[float],
        scope: SourceScope,
        members: Mapping[int, object],
    ) -> list[Evidence]:
        if not query_vector:
            return []
        hits = search_self_intros(
            self._session,
            org_id=org_id,
            query=question,
            query_vector=query_vector,
            scope=scope,
            top_k=INTRO_TOP_K,
            member_ids=list(members),
        )
        return [
            Evidence(
                kind=SELF_INTRO,
                member_id=hit.member_id,
                rank=rank,
                cosine=hit.cosine,
                reason={
                    "type": SELF_INTRO,
                    "detail": f"自己紹介: {intro_excerpt(hit.text, origin=hit.origin)}",
                },
            )
            for rank, hit in enumerate(hits, start=1)
        ]

    # -- documents and conversations C4 found -------------------------------- #
    def _sources(
        self,
        org_id: int,
        query_vector: Sequence[float],
        scope: SourceScope,
        hits: Sequence[SourceHit],
    ) -> list[Evidence]:
        order = list(dict.fromkeys(hit["source_item_id"] for hit in hits))
        if not order:
            return []
        # Read again under the scope: what the reader may not see gives no credit.
        items = {
            item.id: item
            for item in self._session.scalars(
                select(SourceItem).where(
                    SourceItem.id.in_(order),
                    SourceItem.org_id == org_id,
                    visible(org_id, scope),
                )
            ).all()
        }
        cosine = self._item_cosines([hit["chunk_id"] for hit in hits], query_vector)
        documents = [
            items[i]
            for i in order
            if i in items
            and items[i].provider != DISCORD
            and items[i].author_id is not None
            # The introduction slides are credited through the self-introductions.
            and not is_intro_document(items[i].provider, items[i].title)
        ]
        evidence = [
            Evidence(
                kind=DOCUMENT,
                member_id=item.author_id,
                rank=rank,
                cosine=cosine.get(item.id, 0.0),
                reason={
                    "type": DOCUMENT,
                    "detail": f"『{clip_text(item.title or '資料', _TITLE_CHARS)}』を書いた",
                    "source_id": source_citation_id(item.id),
                },
            )
            for rank, item in enumerate(documents, start=1)
            if item.author_id is not None
        ]
        if self._messages_count(org_id):
            conversations = [items[i] for i in order if i in items and items[i].provider == DISCORD]
            for rank, item in enumerate(conversations, start=1):
                evidence.extend(self._speakers(org_id, item, rank, cosine.get(item.id, 0.0)))
        return evidence

    def _item_cosines(
        self, chunk_ids: Sequence[int], query_vector: Sequence[float]
    ) -> dict[int, float]:
        """Per source item, the best cosine of its hit chunks to the question."""

        if not chunk_ids or not query_vector:
            return {}
        distance = func.min(Chunk.embedding.cosine_distance(query_vector))
        rows = self._session.execute(
            select(Chunk.source_item_id, distance)
            .where(Chunk.id.in_(list(chunk_ids)), Chunk.embedding.isnot(None))
            .group_by(Chunk.source_item_id)
        ).all()
        return {item_id: 1.0 - float(dist) for item_id, dist in rows}

    def _messages_count(self, org_id: int) -> bool:
        if self._from_messages is not None:
            return self._from_messages
        return bool(
            self._session.scalar(
                select(Organization.expertise_from_messages).where(Organization.id == org_id)
            )
        )

    def _speakers(self, org_id: int, item: SourceItem, rank: int, cosine: float) -> list[Evidence]:
        chars = func.sum(func.length(ChannelMessage.content))
        rows = self._session.execute(
            select(ChannelMessage.author_id, chars)
            .where(
                ChannelMessage.org_id == org_id,
                ChannelMessage.hidden_at.is_(None),
                ChannelMessage.author_id.isnot(None),
                *conversation_filter(item.external_id),
            )
            .group_by(ChannelMessage.author_id)
        ).all()
        total = sum(int(n or 0) for _, n in rows)
        if total == 0:
            return []
        reason = {
            "type": CONVERSATION,
            "detail": f"{clip_text(item.title or '会話', _TITLE_CHARS)} の会話で発言した",
            "source_id": source_citation_id(item.id),
        }
        return [
            Evidence(
                kind=CONVERSATION,
                member_id=author_id,
                rank=rank,
                cosine=cosine,
                share=int(n or 0) / total,
                reason=reason,
            )
            for author_id, n in rows
            if n
        ]

    # -- answers given through TEKIJIN --------------------------------------- #
    def _answers(
        self, org_id: int, query_vector: Sequence[float], members: Mapping[int, object]
    ) -> list[Evidence]:
        if not query_vector or not members:
            return []
        asker = aliased(Employee)
        distance = Answer.embedding.cosine_distance(query_vector).label("distance")
        rows = self._session.execute(
            select(Answer.responder_id, distance)
            .join(Question, Question.id == Answer.question_id)
            .join(asker, asker.id == Question.asker_id)
            .where(
                asker.org_id == org_id,
                Answer.embedding.isnot(None),
                Answer.responder_id.in_(list(members)),
            )
            .order_by(distance, Answer.id)
            .limit(ANSWER_TOP_K)
        ).all()
        return [
            Evidence(
                kind=ANSWER,
                member_id=row.responder_id,
                rank=rank,
                cosine=1.0 - float(row.distance),
                # A fixed text: the question came from another member's /ask and may
                # be private, so neither it nor the answer is quoted to this asker.
                reason={"type": ANSWER, "detail": ANSWER_REASON},
            )
            for rank, row in enumerate(rows, start=1)
        ]

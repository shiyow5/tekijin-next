"""C6 expertise scorer: rank candidate people with explainable reasons.

Deterministic, LLM-free. Given a question topic and the candidate people C4
retrieved, it recomputes each person's topic edge from raw evidence and combines
it with recency, answer quality, proximity, and a load penalty into a single
score (technical-spec §5). It returns the top-k with a per-term ``reasons``
breakdown — the explanation the UI shows and the project's answer to "誤推薦対策".

``now`` is injected (never read from the clock) so the load window and recency
are reproducible; ties break on ``person_id`` so the ranking is stable.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import TypedDict

from tekijin.data.dto import (
    AnswerDTO,
    CertificationDTO,
    OfflineConsultDTO,
    ProjectMembershipDTO,
    SkillDTO,
)
from tekijin.data.repository import Repository
from tekijin.scorer.evidence import Evidence, collect_topic_evidence, edge_weight
from tekijin.scorer.features import (
    answer_quality,
    confidence_label,
    load,
    proximity,
    recency,
)
from tekijin.scorer.weights import (
    DAYS_PER_MONTH,
    DEFAULT_WEIGHTS,
    LOAD_WINDOW_DAYS,
    QUESTION_FIT_REASON_FLOOR,
    RECENCY_SOURCE_TYPES,
    REGION_OF_BRANCH,
    Weights,
)


class ScoredReason(TypedDict):
    """One explainable contribution to a candidate's score (the UI's evidence)."""

    type: str
    detail: str


class ScoredCandidate(TypedDict):
    """A ranked candidate. ``person_id`` is the internal int; the API boundary
    renders it as the external ``"E###"`` form (events.py / schemas)."""

    person_id: int
    name: str
    dept: str | None
    score: float
    confidence: str
    reasons: list[ScoredReason]


class RankResult(TypedDict):
    """The scorer's public return: the top-k candidates, best first."""

    recommendations: list[ScoredCandidate]


# Fixed order to break reason-contribution ties deterministically. ``self`` is a
# self-declared skill, ``skill`` an inferred one (distinct provenance wording).
_REASON_TYPE_ORDER = {
    "cert": 0,
    "self": 1,
    "skill": 2,
    "answers": 3,
    "project": 4,
    "question_fit": 6,  # #405: question↔past-answer semantic match
    "recency": 7,
    "proximity": 8,
    "load": 9,
}


class ExpertiseScorer:
    """Ranks candidate people for a topic from behavioural evidence."""

    def __init__(
        self,
        repository: Repository,
        *,
        weights: Weights = DEFAULT_WEIGHTS,
    ) -> None:
        self._repo = repository
        self._weights = weights

    def rank(
        self,
        topics: str | Sequence[str],
        candidate_ids: Sequence[int],
        asker_id: int | None,
        now: dt.datetime,
        *,
        top_k: int = 3,
        question_similarity: Mapping[int, float] | None = None,
    ) -> RankResult:
        """Return ``{"recommendations": [...]}`` — the top-``top_k`` candidates.

        ``topics`` may be a single topic or several — for a multi-topic question
        the topic_fit / recency / answer_quality evidence is aggregated across all
        of them (load / proximity are topic-independent). Only ids in
        ``candidate_ids`` that resolve to a real employee are scored; unknown ids
        are dropped. The asker (``asker_id``) is removed — never recommend a
        person to themselves.

        ``question_similarity`` (#405) maps a candidate's ``person_id`` to the max
        cosine of the QUESTION against that person's past answers (from C4's answer
        dense channel). When supplied, ``weights.question_fit * qsim`` is added to
        each candidate's score, lifting the expert whose answers actually match the
        question even when the topic tag saturates or C1 mispredicts the topic.
        ``None`` (the default) leaves the score byte-identical to develop — the
        feature is gated at the graph node by ``question_fit_enabled``.

        ``now`` must be timezone-naive: stored ``created_at`` values are naive, so
        an aware ``now`` would raise on comparison/subtraction.
        """

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        # A ValueError (not assert): fail fast even under python -O, and match the
        # top_k check above.
        if now.tzinfo is not None:
            raise ValueError("now must be naive (matches stored timestamps)")

        topic_list = [topics] if isinstance(topics, str) else list(dict.fromkeys(topics))

        # De-dupe (keep first order), then drop the asker (only when known).
        candidates = [
            pid for pid in dict.fromkeys(candidate_ids) if asker_id is None or pid != asker_id
        ]

        # Load window is [now - 7d, now], both ends inclusive: the upper bound
        # keeps offline eval's replayed ``now`` from counting future rows.
        since = now - dt.timedelta(days=LOAD_WINDOW_DAYS)
        rec_counts = self._repo.recent_recommendation_counts(since, now, candidates)
        ans_counts = self._repo.recent_answer_counts(since, now, candidates)
        answers_by_person = self._group_answers_by_person(self._repo.answers_by_topics(topic_list))
        asker_branch = self._asker_branch(asker_id)

        # Batch every per-candidate evidence lookup into one query each (was O(4·N)
        # queries in the loop — #58). The scorer reads them via ``.get(id, [])``.
        employees = self._repo.employees_by_ids(candidates)
        certs_by_person = self._repo.certifications_for_many(candidates)
        skills_by_person = self._repo.skills_for_many(candidates)
        memberships_by_person = self._repo.project_memberships_for_many(candidates)
        # #247: 直接相談のふりかえり. NOT flag-gated — these rows only ever come from
        # runtime submissions, so the fixtures contain none and the query returns {}
        # on every eval/CI run (byte-identical scores). Gating it would make the
        # feature dead on arrival with nothing to measure against.
        consults_by_person = self._repo.offline_consults_for_many(candidates)

        scored: list[tuple[float, int, ScoredCandidate]] = []
        for person_id in candidates:
            employee = employees.get(person_id)
            if employee is None:
                continue
            record, score = self._score_person(
                topics=topic_list,
                employee_id=person_id,
                employee_name=employee.name,
                employee_dept=employee.department,
                employee_branch=employee.branch,
                certifications=certs_by_person.get(person_id, []),
                skills=skills_by_person.get(person_id, []),
                memberships=memberships_by_person.get(person_id, []),
                answers=answers_by_person.get(person_id, []),
                offline_consults=consults_by_person.get(person_id, []),
                load_count=rec_counts.get(person_id, 0) + ans_counts.get(person_id, 0),
                asker_branch=asker_branch,
                now=now,
                question_fit_score=(
                    question_similarity.get(person_id, 0.0)
                    if question_similarity is not None
                    else None
                ),
            )
            scored.append((score, person_id, record))

        # Descending score, then ascending person_id for a stable tiebreak.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return {"recommendations": [record for _, _, record in scored[:top_k]]}

    # ------------------------------------------------------------------ #
    # per-person scoring
    # ------------------------------------------------------------------ #
    def _score_person(
        self,
        *,
        topics: Sequence[str],
        employee_id: int,
        employee_name: str,
        employee_dept: str | None,
        employee_branch: str | None,
        certifications: Sequence[CertificationDTO],
        skills: Sequence[SkillDTO],
        memberships: Sequence[ProjectMembershipDTO],
        answers: Sequence[AnswerDTO],
        offline_consults: Sequence[OfflineConsultDTO] = (),
        load_count: int,
        asker_branch: str | None,
        now: dt.datetime,
        question_fit_score: float | None = None,
    ) -> tuple[ScoredCandidate, float]:
        # Evidence is pre-fetched in a batch by ``rank`` (no per-candidate query, #58).
        evidence = collect_topic_evidence(
            topics, certifications, skills, memberships, answers, offline_consults
        )
        weights = self._weights

        topic_fit = edge_weight(evidence)
        recency_score = recency(now, self._recency_moments(evidence, now))
        helpful_count = sum(1 for a in answers if a.was_helpful is True)
        reuse_total = sum(a.reuse_count or 0 for a in answers)
        quality = answer_quality(helpful_count, reuse_total, len(answers))
        proximity_score = proximity(asker_branch, employee_branch)
        load_score = load(load_count)

        score = (
            weights.topic_fit * topic_fit
            + weights.recency * recency_score
            + weights.answer_quality * quality
            + weights.proximity * proximity_score
            - weights.load * load_score
        )
        # #405: additive question↔past-answer term. ``None`` -> dormant (develop
        # byte-identical); a float (incl. 0.0) adds ``weights.question_fit * qsim``.
        if question_fit_score is not None:
            score += weights.question_fit * question_fit_score

        reasons = self._build_reasons(
            evidence=evidence,
            topic_fit=topic_fit,
            recency_score=recency_score,
            quality=quality,
            proximity_score=proximity_score,
            load_score=load_score,
            load_count=load_count,
            answers=answers,
            helpful_count=helpful_count,
            employee_branch=employee_branch,
            asker_branch=asker_branch,
            now=now,
            question_fit_score=question_fit_score,
        )
        record: ScoredCandidate = {
            "person_id": employee_id,
            "name": employee_name,
            "dept": employee_dept,
            "score": round(score, 4),
            "confidence": confidence_label(e.source_type for e in evidence),
            "reasons": reasons,
        }
        return record, score

    # ------------------------------------------------------------------ #
    # reasons
    # ------------------------------------------------------------------ #
    def _build_reasons(
        self,
        *,
        evidence: Sequence[Evidence],
        topic_fit: float,
        recency_score: float,
        quality: float,
        proximity_score: float,
        load_score: float,
        load_count: int,
        answers: Sequence[AnswerDTO],
        helpful_count: int,
        employee_branch: str | None,
        asker_branch: str | None,
        now: dt.datetime,
        question_fit_score: float | None = None,
    ) -> list[ScoredReason]:
        weights = self._weights
        base_total = sum(e.base_score for e in evidence)
        entries: list[tuple[float, ScoredReason]] = []

        def topic_fit_share(source_types: tuple[str, ...]) -> float:
            # Only ever called from a block guarded by matching evidence, so
            # ``base_total`` is strictly positive here.
            share = sum(e.base_score for e in evidence if e.source_type in source_types)
            return weights.topic_fit * topic_fit * (share / base_total)

        cert_names = [e.detail for e in evidence if e.source_type == "cert"]
        if cert_names:
            entries.append(
                (topic_fit_share(("cert",)), {"type": "cert", "detail": "、".join(cert_names)})
            )

        # Skill evidence: surface it so cold-start candidates (chosen on a skill
        # alone) are explained by their expertise signal, not just load — worded
        # by provenance (an inferred skill is never called "declared") and naming
        # the actual skill topics (multi-topic accurate; e.detail is the topic).
        self_topics = [e.detail for e in evidence if e.source_type == "self"]
        if self_topics:
            entries.append(
                (
                    topic_fit_share(("self",)),
                    {"type": "self", "detail": "自己申告スキル: " + "、".join(self_topics)},
                )
            )
        inferred_topics = [e.detail for e in evidence if e.source_type == "inferred"]
        if inferred_topics:
            entries.append(
                (
                    topic_fit_share(("inferred",)),
                    {"type": "skill", "detail": "推定スキル: " + "、".join(inferred_topics)},
                )
            )

        if answers:
            detail = f"類似の質問に過去{len(answers)}件回答"
            if helpful_count:
                detail += f"（うち有用と評価{helpful_count}件）"
            answers_weight = topic_fit_share(("answer",)) + weights.answer_quality * quality
            entries.append((answers_weight, {"type": "answers", "detail": detail}))

        project_count = sum(1 for e in evidence if e.source_type == "project")
        if project_count:
            entries.append(
                (
                    topic_fit_share(("project",)),
                    {"type": "project", "detail": f"同種の案件を{project_count}件担当"},
                )
            )

        # #405: explain the question-fit boost. Shown only above a noise floor —
        # below it the cosine is not a meaningful "match" so the assertive detail
        # would mislead (the tiny continuous score nudge is still applied above).
        if question_fit_score is not None and question_fit_score >= QUESTION_FIT_REASON_FLOOR:
            entries.append(
                (
                    weights.question_fit * question_fit_score,
                    {"type": "question_fit", "detail": "質問内容が過去の回答と一致"},
                )
            )

        recency_detail = self._recency_detail(evidence, now) if recency_score > 0.0 else ""
        if recency_detail:
            entries.append(
                (weights.recency * recency_score, {"type": "recency", "detail": recency_detail})
            )

        entries.append(
            (
                weights.proximity * proximity_score,
                {
                    "type": "proximity",
                    "detail": self._proximity_detail(asker_branch, employee_branch),
                },
            )
        )
        entries.append(
            (
                -weights.load * load_score,
                {"type": "load", "detail": f"今週の対応件数: {self._load_label(load_count)}"},
            )
        )

        # Highest contribution first; stable tiebreak on the fixed type order.
        entries.sort(key=lambda item: (-item[0], _REASON_TYPE_ORDER[item[1]["type"]]))
        return [reason for _, reason in entries]

    @staticmethod
    def _recency_moments(
        evidence: Sequence[Evidence], now: dt.datetime
    ) -> list[dt.date | dt.datetime]:
        """Timestamps of the time-decaying evidence (projects/answers).

        Certifications are omitted on purpose — a qualification does not fade with
        age, so it must not feed the recency term (RECENCY_SOURCE_TYPES). An
        ongoing project (``timestamp is None``, i.e. no end_date) is treated as
        current work: its moment is ``now``, so live engagements stay fresh rather
        than decaying from their start date.
        """

        moments: list[dt.date | dt.datetime] = []
        for e in evidence:
            if e.source_type not in RECENCY_SOURCE_TYPES:
                continue
            if e.source_type == "project" and e.timestamp is None:
                moments.append(now)  # ongoing project = current
            elif e.timestamp is not None:
                moments.append(e.timestamp)
        return moments

    @classmethod
    def _recency_detail(cls, evidence: Sequence[Evidence], now: dt.datetime) -> str:
        moments = cls._recency_moments(evidence, now)
        if not moments:
            return ""
        newest = max(
            (m if isinstance(m, dt.datetime) else dt.datetime(m.year, m.month, m.day))
            for m in moments
        )
        months = max((now - newest).total_seconds() / 86400.0 / DAYS_PER_MONTH, 0.0)
        if months < 1.0:
            return "直近1か月以内の関連実績あり"
        return f"直近の関連実績: 約{round(months)}か月前"

    @staticmethod
    def _proximity_detail(asker_branch: str | None, employee_branch: str | None) -> str:
        if asker_branch and employee_branch and asker_branch == employee_branch:
            return f"同じ拠点（{employee_branch}）"
        if (
            asker_branch
            and employee_branch
            and REGION_OF_BRANCH.get(asker_branch) is not None
            and REGION_OF_BRANCH.get(asker_branch) == REGION_OF_BRANCH.get(employee_branch)
        ):
            return "同じエリア"
        return "全社から選定"

    @staticmethod
    def _load_label(load_count: int) -> str:
        if load_count <= 1:
            return "少なめ"
        if load_count <= 4:
            return "やや多め"
        return "多め"

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _group_answers_by_person(answers: Sequence[AnswerDTO]) -> dict[int, list[AnswerDTO]]:
        grouped: dict[int, list[AnswerDTO]] = {}
        for answer in answers:
            grouped.setdefault(answer.responder_id, []).append(answer)
        return grouped

    def _asker_branch(self, asker_id: int | None) -> str | None:
        if asker_id is None:
            return None
        asker = self._repo.get_employee(asker_id)
        return asker.branch if asker is not None else None

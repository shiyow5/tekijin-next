"""Read-oriented repository over the TEKIJIN schema.

A thin object that wraps a SQLAlchemy :class:`~sqlalchemy.orm.Session` and
exposes ``find``-style queries used by the search and scorer components. Every
method returns immutable :mod:`tekijin.data.dto` snapshots rather than live ORM
rows, keeping callers insulated from the session lifecycle and from accidental
mutation.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, joinedload, selectinload

from tekijin.data.dto import (
    AnswerDTO,
    CertificationDTO,
    DailyReportDTO,
    DocumentDTO,
    EmployeeDTO,
    OfflineConsultDTO,
    ProfileDTO,
    ProjectMembershipDTO,
    ProjectWithMembersDTO,
    QuestionDTO,
    SkillDTO,
)
from tekijin.models.tables import (
    Answer,
    Certification,
    DailyReport,
    Document,
    Employee,
    EmployeeProfile,
    OfflineConsult,
    Project,
    ProjectMember,
    Question,
    Recommendation,
    Skill,
)

# Who may be recommended: still in the organization (#506) and not opted out (#66).
_RECOMMENDABLE = (Employee.is_active.is_(True), Employee.opted_out_at.is_(None))


class Repository:
    """Read-only data access over an active session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- employees -------------------------------------------------------- #
    def list_employees(self) -> list[EmployeeDTO]:
        """Everyone who can still be recommended (#506).

        This is the candidate pool the scorer runs over, so someone who has left
        must not appear — a hand-off to them cannot be delivered, their Slack
        link having been removed by the same departure. ``get_employee`` is
        deliberately NOT filtered: their name still has to resolve, or the
        questions and answers they left behind lose their author.
        """

        rows = self._session.scalars(
            select(Employee).where(*_RECOMMENDABLE).order_by(Employee.id)
        ).all()
        return [EmployeeDTO.from_row(r) for r in rows]

    def active_employee_ids(self) -> set[int]:
        """Ids of everyone who may still be recommended (#506).

        The retriever needs this rather than :meth:`list_employees` because it
        builds its candidate list out of answers and profiles, not out of the
        roster — filtering the roster alone left the real pool untouched.
        """

        return set(self._session.scalars(select(Employee.id).where(*_RECOMMENDABLE)))

    def get_employee(self, employee_id: int) -> EmployeeDTO | None:
        row = self._session.get(Employee, employee_id)
        return EmployeeDTO.from_row(row) if row is not None else None

    def employees_by_ids(self, employee_ids: Sequence[int]) -> dict[int, EmployeeDTO]:
        """Resolve several employees in one query (scorer N+1 fix, #58).

        Keyed by id; unknown ids are simply absent. Empty ids → ``{}`` (no query).
        """

        if not employee_ids:
            return {}
        rows = self._session.scalars(select(Employee).where(Employee.id.in_(employee_ids)))
        return {row.id: EmployeeDTO.from_row(row) for row in rows}

    def get_profile(self, employee_id: int) -> ProfileDTO | None:
        row = self._session.get(EmployeeProfile, employee_id)
        return ProfileDTO.from_row(row) if row is not None else None

    def list_profiles(self) -> list[ProfileDTO]:
        """Every employee profile, ordered by employee id.

        Used to build the BM25 profile corpus so people whose expertise terms
        live only in their self-description are reachable by lexical search.
        """

        stmt = select(EmployeeProfile).order_by(EmployeeProfile.employee_id)
        return [ProfileDTO.from_row(r) for r in self._session.scalars(stmt)]

    # -- expertise evidence ---------------------------------------------- #
    def certifications_for(self, employee_id: int) -> list[CertificationDTO]:
        stmt = (
            select(Certification)
            .where(Certification.employee_id == employee_id)
            .order_by(Certification.id)
        )
        return [CertificationDTO.from_row(r) for r in self._session.scalars(stmt)]

    def certifications_for_many(
        self, employee_ids: Sequence[int]
    ) -> dict[int, list[CertificationDTO]]:
        """Certifications for several employees in one query (scorer N+1 fix, #58).

        Grouped by ``employee_id``; each list keeps the same ``id`` order as the
        per-employee :meth:`certifications_for`. Empty ids → ``{}`` (no query);
        employees with none are simply absent (callers use ``.get(id, [])``).
        """

        if not employee_ids:
            return {}
        stmt = (
            select(Certification)
            .where(Certification.employee_id.in_(employee_ids))
            .order_by(Certification.employee_id, Certification.id)
        )
        out: dict[int, list[CertificationDTO]] = {}
        for row in self._session.scalars(stmt):
            out.setdefault(row.employee_id, []).append(CertificationDTO.from_row(row))
        return out

    def skills_for(self, employee_id: int) -> list[SkillDTO]:
        stmt = select(Skill).where(Skill.employee_id == employee_id).order_by(Skill.id)
        return [SkillDTO.from_row(r) for r in self._session.scalars(stmt)]

    def skills_for_many(self, employee_ids: Sequence[int]) -> dict[int, list[SkillDTO]]:
        """Skills for several employees in one query (scorer N+1 fix, #58)."""

        if not employee_ids:
            return {}
        stmt = (
            select(Skill)
            .where(Skill.employee_id.in_(employee_ids))
            .order_by(Skill.employee_id, Skill.id)
        )
        out: dict[int, list[SkillDTO]] = {}
        for row in self._session.scalars(stmt):
            out.setdefault(row.employee_id, []).append(SkillDTO.from_row(row))
        return out

    def daily_reports_for_many(
        self, employee_ids: Sequence[int]
    ) -> dict[int, list[DailyReportDTO]]:
        """Daily reports for several employees in one query (scorer evidence, #355).

        Only reports carrying at least one precomputed topic are returned (a
        report with no topic is not evidence). Ordered newest first so the
        ``DAILY_EVIDENCE_CAP`` in the scorer keeps the most recent activity.
        """

        if not employee_ids:
            return {}
        stmt = (
            select(DailyReport)
            .where(
                DailyReport.employee_id.in_(employee_ids),
                DailyReport.topics.isnot(None),
            )
            .order_by(DailyReport.employee_id, DailyReport.report_date.desc(), DailyReport.id)
        )
        out: dict[int, list[DailyReportDTO]] = {}
        for row in self._session.scalars(stmt):
            if row.topics:
                out.setdefault(row.employee_id, []).append(DailyReportDTO.from_row(row))
        return out

    def offline_consults_for_many(
        self, employee_ids: Sequence[int]
    ) -> dict[int, list[OfflineConsultDTO]]:
        """直接相談のふりかえり for several responders in one query (#247).

        Keyed by ``responder_id`` — the person the retrospective is evidence FOR,
        not the asker who wrote it. Only rows carrying at least one topic are
        returned (an untagged retrospective cannot be joined to a topic). Ordered
        newest first so ``OFFLINE_CONSULT_EVIDENCE_CAP`` keeps the most recent
        consultations. Unresolved rows are NOT filtered here: the scorer drops
        them (``collect_topic_evidence``), and keeping the read side dumb means
        the accumulation metrics can count them without a second query.
        """

        if not employee_ids:
            return {}
        stmt = (
            select(OfflineConsult)
            .where(
                OfflineConsult.responder_id.in_(employee_ids),
                OfflineConsult.topics.isnot(None),
            )
            .order_by(
                OfflineConsult.responder_id,
                OfflineConsult.created_at.desc(),
                OfflineConsult.id,
            )
        )
        out: dict[int, list[OfflineConsultDTO]] = {}
        for row in self._session.scalars(stmt):
            if row.topics:
                out.setdefault(row.responder_id, []).append(OfflineConsultDTO.from_row(row))
        return out

    # -- questions & answers --------------------------------------------- #
    def list_questions(self) -> list[QuestionDTO]:
        rows = self._session.scalars(select(Question).order_by(Question.id)).all()
        return [QuestionDTO.from_row(r) for r in rows]

    def answers_by_topic(self, topic: str) -> list[AnswerDTO]:
        """Answers relevant to ``topic``, matched two ways.

        The fixture-only ``answers.topic`` column is NULL for answers created at
        runtime, so the topic of a runtime answer lives on its linked question
        (``questions.topics`` is an array). This matches on either signal:
        ``answers.topic == topic`` OR ``topic = ANY(questions.topics)``.
        """

        stmt = (
            select(Answer)
            .join(Question, Answer.question_id == Question.id)
            .where(or_(Answer.topic == topic, Question.topics.any(topic)))  # type: ignore[arg-type]
            .order_by(Answer.id)
        )
        return [AnswerDTO.from_row(r) for r in self._session.scalars(stmt)]

    def answers_by_topics(self, topics: Sequence[str]) -> list[AnswerDTO]:
        """Answers that are genuine evidence for ANY of ``topics``, in one query.

        The strict subtopic rule (see the scorer): an answer counts when its OWN
        ``topic`` is one of ``topics``, OR its ``topic`` is NULL and its question's
        ``topics`` array overlaps ``topics`` (the intended question-topics
        fallback). An answer tagged for a *different* subtopic is excluded. This
        replaces the per-topic ``answers_by_topic`` fan-out (N+1) with a single
        ``IN`` / array-overlap query. Ordered by ``id`` (deterministic); an empty
        ``topics`` yields ``[]`` without a query.
        """

        topic_list = list(topics)
        if not topic_list:
            return []
        stmt = (
            select(Answer)
            .join(Question, Answer.question_id == Question.id)
            .where(
                or_(
                    Answer.topic.in_(topic_list),
                    and_(Answer.topic.is_(None), Question.topics.overlap(topic_list)),
                )
            )
            .order_by(Answer.id)
        )
        return [AnswerDTO.from_row(r) for r in self._session.scalars(stmt)]

    def answers_for_question(self, question_id: str) -> list[AnswerDTO]:
        stmt = select(Answer).where(Answer.question_id == question_id).order_by(Answer.id)
        return [AnswerDTO.from_row(r) for r in self._session.scalars(stmt)]

    def list_answers(self) -> list[AnswerDTO]:
        """Every answer, ordered by id. Used to build the BM25 answer corpus."""

        rows = self._session.scalars(select(Answer).order_by(Answer.id)).all()
        return [AnswerDTO.from_row(r) for r in rows]

    def answers_by_ids(self, ids: Sequence[str]) -> dict[str, AnswerDTO]:
        """Resolve several answers by id in one query (#69 fragment rehydration).

        Keyed by id; unknown ids are simply absent. Empty ids → ``{}`` (no query).
        """

        if not ids:
            return {}
        rows = self._session.scalars(select(Answer).where(Answer.id.in_(ids)))
        return {row.id: AnswerDTO.from_row(row) for row in rows}

    def questions_by_ids(self, ids: Sequence[str]) -> dict[str, QuestionDTO]:
        """Resolve several questions by id in one query (#69 fragment rehydration)."""

        if not ids:
            return {}
        rows = self._session.scalars(select(Question).where(Question.id.in_(ids)))
        return {row.id: QuestionDTO.from_row(row) for row in rows}

    # -- documents -------------------------------------------------------- #
    def list_documents(self) -> list[DocumentDTO]:
        rows = self._session.scalars(select(Document).order_by(Document.id)).all()
        return [DocumentDTO.from_row(r) for r in rows]

    def documents_by_ids(self, ids: Sequence[str]) -> dict[str, DocumentDTO]:
        """Resolve several documents by id in one query (#69 fragment rehydration)."""

        if not ids:
            return {}
        rows = self._session.scalars(select(Document).where(Document.id.in_(ids)))
        return {row.id: DocumentDTO.from_row(row) for row in rows}

    def daily_reports_by_ids(self, ids: Sequence[int]) -> dict[int, DailyReportDTO]:
        """Resolve several daily reports by id in one query (#433 knowledge source)."""

        if not ids:
            return {}
        rows = self._session.scalars(select(DailyReport).where(DailyReport.id.in_(ids)))
        return {row.id: DailyReportDTO.from_row(row) for row in rows}

    # -- projects --------------------------------------------------------- #
    def list_projects_with_members(self) -> list[ProjectWithMembersDTO]:
        stmt = select(Project).options(selectinload(Project.members)).order_by(Project.id)
        rows = self._session.scalars(stmt).all()
        return [ProjectWithMembersDTO.from_row(r) for r in rows]

    def project_memberships_for(self, employee_id: int) -> list[ProjectMembershipDTO]:
        """Projects an employee is on (with role, product, dates) for scoring.

        Ordered by ``project_id`` so the result is deterministic.
        """

        stmt = (
            select(ProjectMember)
            .options(joinedload(ProjectMember.project))
            .where(ProjectMember.employee_id == employee_id)
            .order_by(ProjectMember.project_id)
        )
        return [ProjectMembershipDTO.from_member(m) for m in self._session.scalars(stmt)]

    def project_memberships_for_many(
        self, employee_ids: Sequence[int]
    ) -> dict[int, list[ProjectMembershipDTO]]:
        """Project memberships for several employees in one query (scorer N+1 fix, #58).

        Grouped by ``employee_id``; each list keeps the ``project_id`` order of the
        per-employee :meth:`project_memberships_for`. Empty ids → ``{}`` (no query).
        """

        if not employee_ids:
            return {}
        stmt = (
            select(ProjectMember)
            .options(joinedload(ProjectMember.project))
            .where(ProjectMember.employee_id.in_(employee_ids))
            .order_by(ProjectMember.employee_id, ProjectMember.project_id)
        )
        out: dict[int, list[ProjectMembershipDTO]] = {}
        for member in self._session.scalars(stmt):
            out.setdefault(member.employee_id, []).append(ProjectMembershipDTO.from_member(member))
        return out

    # -- load (recency windows) ------------------------------------------ #
    def recent_recommendation_counts(
        self, since: dt.datetime, until: dt.datetime, employee_ids: Sequence[int]
    ) -> dict[int, int]:
        """Per-employee recommendation count within ``[since, until]`` (inclusive).

        Feeds the scorer's ``load`` penalty (technical-spec §5: "直近7日の推薦件数",
        which is outcome-independent). ``declined`` recommendations ARE counted:
        a decline lowers only availability (余裕度), never expertise — the spec
        keeps declines out of the expertise evidence, not out of the load window.

        The upper bound ``until`` (the scorer's ``now``) is required so that
        offline evaluation replaying a historical ``now`` (#33) never lets rows
        created *after* that moment leak into the window. Both ends are inclusive.
        Scoped to ``employee_ids``; an empty list yields ``{}`` without a query.
        """

        if not employee_ids:
            return {}
        stmt = (
            select(Recommendation.employee_id, func.count())
            .where(Recommendation.created_at >= since)
            .where(Recommendation.created_at <= until)
            .where(Recommendation.employee_id.in_(employee_ids))
            .group_by(Recommendation.employee_id)
        )
        return {employee_id: count for employee_id, count in self._session.execute(stmt)}

    def recent_answer_counts(
        self, since: dt.datetime, until: dt.datetime, responder_ids: Sequence[int]
    ) -> dict[int, int]:
        """Per-responder answer count within ``[since, until]`` (inclusive; load).

        ``until`` (the scorer's ``now``) bounds the window on the top end so a
        replayed historical ``now`` never counts later answers. Scoped to
        ``responder_ids``; an empty list yields ``{}`` without a query.
        """

        if not responder_ids:
            return {}
        stmt = (
            select(Answer.responder_id, func.count())
            .where(Answer.created_at >= since)
            .where(Answer.created_at <= until)
            .where(Answer.responder_id.in_(responder_ids))
            .group_by(Answer.responder_id)
        )
        return {responder_id: count for responder_id, count in self._session.execute(stmt)}

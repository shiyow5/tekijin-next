"""SQLAlchemy 2.0 ORM table definitions for the TEKIJIN schema.

The schema follows ``docs/specs/db-schema.md`` and is organised in three layers:

* **A. Input data** — employees, profiles, chat, daily reports, projects.
* **B. Runtime** — certifications, skills, questions, answers, recommendations,
  events, project members, documents.
* **C. Expertise graph** — person/topic edges and their evidence.

``embedding`` columns use :class:`pgvector.sqlalchemy.Vector`; they are
``nullable`` because vectors are computed at ingestion time (a later component),
so fixtures seed the rows with ``NULL`` embeddings.

Every foreign key points back at :class:`Employee` (the "who"), matching the ER
diagram. Primary-key types mirror the synthetic fixtures verbatim: integer keys
for employees/projects/chat/reports, string keys for the id-prefixed entities
(``cert_0001``, ``q_0001`` …).
"""

from __future__ import annotations

import datetime as dt

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tekijin.config import get_settings
from tekijin.data.db import Base

# Width of every pgvector column, taken from settings so the schema and the
# embedding model can never drift apart.
EMBEDDING_DIM: int = get_settings().embedding_dim


# --------------------------------------------------------------------------- #
# A. Input data layer
# --------------------------------------------------------------------------- #
class Organization(Base):
    """One organization TEKIJIN serves (#66, ADR-0012).

    For now an organization is one Discord server (``discord_guild_id``). The
    first deployment holds a single row, but members and identities already carry
    ``org_id`` so a second organization does not need a schema change.
    ``expertise_from_messages`` gates using members' Discord messages as evidence
    of what they know (Discord Developer Policy §16, see ADR-0012); off by default.
    """

    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    discord_guild_id: Mapped[str | None] = mapped_column(String(32), unique=True)
    expertise_from_messages: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false")
    )
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class Employee(Base):
    """A member of the organization. The "who" every other table refers to.

    Still named ``employees`` from the company-only design; the rename to
    ``members`` is planned for R3, together with dropping the company-only
    columns (department, branch, ...). A member created from Discord has only a
    name, an organization and an identity: ``email`` is optional.
    """

    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255), unique=True)
    org_id: Mapped[int | None] = mapped_column(ForeignKey("organizations.id"), index=True)
    # ``/tekijin join``: the member registered and may sign in to the web UI.
    joined_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    # ``/tekijin optout``: never recommended, and evidence about them is dropped.
    opted_out_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    department: Mapped[str | None] = mapped_column(String(255))
    section: Mapped[str | None] = mapped_column(String(255))
    position: Mapped[str | None] = mapped_column(String(255))
    branch: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[str | None] = mapped_column(String(255))
    hire_date: Mapped[dt.date | None] = mapped_column(Date)
    department_history: Mapped[list | None] = mapped_column(JSONB)
    # PBKDF2-encoded login password (#241). Nullable so a pre-auth DB row (or a
    # fixture that omits it) is valid; a NULL/blank hash simply never verifies, so
    # such an account cannot log in until seeded. Never exposed via EmployeeDTO.
    password_hash: Mapped[str | None] = mapped_column(String(255))
    # Whether this person is still with the company (#506). Departure is NOT a
    # delete: questions, answers and evidence all reference this row, and
    # removing it would tear the history out of the product. Instead they leave
    # the CANDIDATE POOL, so the recommender stops offering someone whose
    # hand-off can no longer be delivered. Defaults to true — the synthetic
    # roster predates the column, and a default of false would empty the pool.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))


class Identity(Base):
    """A member's account on an external service (#66).

    ``provider`` is ``discord`` for now (``slack``/``google``/``notion`` later;
    Slack still uses ``slack_links`` until it is generalized). One external
    account maps to exactly one member within an organization.
    """

    __tablename__ = "identities"
    __table_args__ = (
        UniqueConstraint("org_id", "provider", "external_id", name="uq_identities_external"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    provider: Mapped[str] = mapped_column(String(16))
    external_id: Mapped[str] = mapped_column(String(64))
    display_name: Mapped[str | None] = mapped_column(String(255))
    linked_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class SourceItem(Base):
    """One thing ingested from outside, whatever its origin (#70, R2).

    ``provider`` says where it came from: ``manual`` (``/tekijin add`` or the web),
    and later ``discord``, ``google``, ``notion``. ``external_id`` identifies it there,
    so re-ingesting the same thing updates this row instead of adding a copy.
    ``content_hash`` lets a re-send of an unchanged body skip re-embedding.
    """

    __tablename__ = "source_items"
    __table_args__ = (
        UniqueConstraint("org_id", "provider", "external_id", name="uq_source_items_external"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    provider: Mapped[str] = mapped_column(String(16))
    external_id: Mapped[str] = mapped_column(String(2048))
    url: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    author_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id"), index=True)
    content_hash: Mapped[str] = mapped_column(String(64))
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class Chunk(Base):
    """A passage of a :class:`SourceItem`, embedded on its own (#70, R2)."""

    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("source_item_id", "seq", name="uq_chunks_seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    source_item_id: Mapped[int] = mapped_column(
        ForeignKey("source_items.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    # Read by re-embedding, which needs the title the chunk was embedded with.
    source_item: Mapped[SourceItem] = relationship()


class EmployeeProfile(Base):
    """Free-text self-description of an employee (one row per employee)."""

    __tablename__ = "employee_profiles"

    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id"), primary_key=True, index=True
    )
    description: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    updated_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class SlackLink(Base):
    """One employee's linked Slack account (one row per employee).

    Only the Slack user/team id is stored — no per-user OAuth token. Every
    Slack-side action (posting into a thread's channel, inviting a member)
    goes through the app's own bot token instead, so there is nothing here to
    refresh or revoke on Slack's side beyond the identity mapping itself.
    """

    __tablename__ = "slack_links"

    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id"), primary_key=True, index=True
    )
    slack_user_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    slack_team_id: Mapped[str] = mapped_column(String(32))
    linked_at: Mapped[dt.datetime] = mapped_column(DateTime)


class SlackChannelLink(Base):
    """The shared private Slack channel for one PAIR of employees (#hand-off-chat).

    Created once, the first time a "chat" hand-off between two Slack-linked
    employees is accepted (a private channel containing the bot plus both of
    them), then REUSED for every later hand-off between the same pair — one
    channel per relationship, not one per question, so consulting the same
    person again doesn't pile up a fresh channel every time.

    ``current_thread_id`` is which TEKIJIN thread an inbound Slack message in
    this channel is attributed to. It is stamped to the new thread whenever
    the channel is reused for another hand-off, so it always points at the
    pair's most recent consultation — a "most recent thread wins" heuristic:
    if the same two people have two hand-offs open at once, a Slack reply
    lands on whichever was accepted last, even if a human meant it for the
    other. Naming the row by the pair (not by ``thread_id``, an earlier
    design) is what makes channel reuse possible at all.
    """

    __tablename__ = "slack_channel_links"

    # Canonicalized so the pair (A, B) and (B, A) are always the same row:
    # employee_low_id < employee_high_id, enforced by the writer, not the DB.
    employee_low_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), primary_key=True)
    employee_high_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), primary_key=True)
    slack_channel_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    slack_team_id: Mapped[str] = mapped_column(String(32))
    current_thread_id: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime)


class SlackMessageAnchor(Base):
    """Which TEKIJIN thread a specific Slack message belongs to (#476/#508).

    A pair channel is REUSED across sequential hand-offs (see
    :class:`SlackChannelLink`), so ``current_thread_id`` only names the pair's most
    recent thread — it cannot tell which thread an OLD message was part of. Solve
    capture (#476) is triggered by a ✅ reaction on a specific message, so it needs
    per-message provenance: this records, at the moment a message is mirrored, the
    thread that was current then, keyed by ``(channel_id, slack_ts)`` (a Slack ts is
    unique within a channel). The reaction handler resolves ``item.ts`` here to
    attribute the capture to the RIGHT thread, not merely the latest one (#508).
    """

    __tablename__ = "slack_message_anchors"

    slack_channel_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    slack_ts: Mapped[str] = mapped_column(String(32), primary_key=True)
    # No index: every read goes through the composite PK (channel, ts). Add one only
    # when a by-thread query exists (e.g. bulk-purging a thread's anchors) (#511 review).
    thread_id: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime)


class AiChatHistory(Base):
    """AI <-> employee conversation log (one row per message)."""

    __tablename__ = "ai_chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"))
    speaker: Mapped[str] = mapped_column(String(32))
    content: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class EmployeeChatHistory(Base):
    """Employee-to-employee chat log (Slack/Teams style)."""

    __tablename__ = "employee_chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sender_employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"))
    receiver_employee_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id"))
    channel: Mapped[str | None] = mapped_column(Text)
    message: Mapped[str | None] = mapped_column(Text)
    sent_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class OfflineConsult(Base):
    """One retrospective the ASKER wrote after a 直接相談 (#247).

    A "direct" consultation (#245) happens face to face, so unlike a chat hand-off
    it leaves no text behind and F-10 (回答を索引に追加し専門性の推定を更新) has
    nothing to work with. This table is that missing record, written by the asker.

    Deliberately HEARSAY: ``answer_body`` is the asker's paraphrase of what the
    responder said, so the scorer weights it below a self-declared skill
    (``BASE_SCORE_OFFLINE_CONSULT``). ``topics`` comes from ``TOPIC_VOCABULARY``,
    validated at the API boundary, so the scorer can join on it without a runtime
    keyword vocabulary — the same contract as ``daily_reports.topics`` (#355).

    ``resolution`` is ``resolved`` / ``partial`` / ``unresolved``. ``unresolved``
    is stored but contributes NO expertise evidence and never subtracts — the same
    rule as a decline (db-schema.md: 断り≠非専門).

    ONE row per question, enforced by a unique constraint: exactly one hand-off per
    question is ever accepted, so "the consultation" is singular. Without it the
    asker could write the same consultation up ``OFFLINE_CONSULT_EVIDENCE_CAP``
    times and reach the full cap from a single real conversation.
    """

    __tablename__ = "offline_consults"
    __table_args__ = (UniqueConstraint("question_id", name="uq_offline_consults_question"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # NOT NULL: a write-up with no question is not weaker evidence, it is
    # unverifiable evidence, so there is nothing to record (the route answers 404).
    # It also makes the unique constraint above bite — Postgres allows any number
    # of NULLs in a unique column.
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"), index=True)
    # Who answered (the person this becomes evidence for) and who is reporting it.
    responder_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    asker_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id"), index=True)
    topics: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    asked: Mapped[str | None] = mapped_column(Text)
    answer_body: Mapped[str | None] = mapped_column(Text)
    resolution: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[dt.datetime | None] = mapped_column(DateTime, server_default=func.now())


class DailyReport(Base):
    """Daily report submitted by an employee."""

    __tablename__ = "daily_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    report_date: Mapped[dt.date | None] = mapped_column(Date)
    content: Mapped[str | None] = mapped_column(Text)
    issue: Mapped[str | None] = mapped_column(Text)
    # #355: precomputed topics (build_fixtures ``match_topics``), so the scorer can
    # use daily reports as topic evidence without a runtime keyword vocabulary.
    topics: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    created_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    # #433: dense embedding of ``issue + content`` so a daily report can be a
    # SEARCHABLE knowledge source for System 1 (self-answer / #413 additive), not
    # just topic-overlap evidence for the scorer. NULL until ``make embed`` fills
    # it (fresh DBs via create_all, existing via _apply_schema_upgrades).
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))


class Project(Base):
    """Sales/consulting project. Membership lives in ``project_members``."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    subject: Mapped[str | None] = mapped_column(Text)
    client_company: Mapped[str | None] = mapped_column(String(255))
    industry: Mapped[str | None] = mapped_column(String(255))
    company_size: Mapped[str | None] = mapped_column(String(255))
    client_issue: Mapped[str | None] = mapped_column(Text)
    product: Mapped[str | None] = mapped_column(String(255))
    negotiation_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str | None] = mapped_column(String(64))
    remarks: Mapped[str | None] = mapped_column(Text)
    start_date: Mapped[dt.date | None] = mapped_column(Date)
    end_date: Mapped[dt.date | None] = mapped_column(Date)

    members: Mapped[list[ProjectMember]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


# --------------------------------------------------------------------------- #
# B. Runtime layer
# --------------------------------------------------------------------------- #
class Certification(Base):
    """Certification held by an employee (strongest evidence, base_score 0.6)."""

    __tablename__ = "certifications"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    acquired_at: Mapped[dt.date | None] = mapped_column(Date)


class Skill(Base):
    """Self-declared / inferred skill (base_score 0.3 for ``self``)."""

    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    topic: Mapped[str] = mapped_column(String(255))
    level: Mapped[str | None] = mapped_column(String(64))
    source: Mapped[str | None] = mapped_column(String(64))


class Question(Base):
    """A question asked by an employee (the "asking" side)."""

    __tablename__ = "questions"
    # GIN index on the topics array: ``answers_by_topic`` filters with
    # ``topic = ANY(questions.topics)``, which a GIN index accelerates.
    __table_args__ = (Index("ix_questions_topics", "topics", postgresql_using="gin"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    asker_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    body: Mapped[str | None] = mapped_column(Text)
    topics: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    status: Mapped[str | None] = mapped_column(String(64))
    # The graph ``thread_id`` (== the client's ``session_id``) for an API-created
    # question. The pending-handoff view lives in the checkpointer keyed by this,
    # not in SQL, so the responder inbox joins on it to deep-link into
    # ``/answer/{session_id}``. NULL for seeded/pre-session questions.
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    # C5 route the run took (person / prior_answer / document). Persisted by the
    # API when C5 emits, so the dashboard can report the self-resolution rate
    # (補助経路で人を介さず解決した割合). NULL for pre-seeded/unrouted questions.
    route: Mapped[str | None] = mapped_column(String(32))
    # The asker's chosen consultation method ("direct" | "chat"), set via
    # POST /handoff/draft when confirming the hand-off. Lives on the question
    # (not the recommendation) so it survives a decline+reroute, which creates
    # a NEW Recommendation row for the same question. NULL (never set) is
    # treated as "chat" everywhere — the implicit pre-existing default.
    consult_method: Mapped[str | None] = mapped_column(String(32))
    # When the question was resolved at runtime — a responder accepted the hand-off
    # (C8) or a self-resolving terminal (document) was reached. Set first-wins by
    # the API so the dashboard's average resolution time reflects live traffic, not
    # only seeded ``answers`` rows (#97). NULL while unresolved / pre-seeded.
    resolved_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    # HOW the question was resolved when it did NOT go through a live hand-off:
    # ``"self"`` = the asker marked it solved without asking a person (#159, e.g. a
    # document answer or a past answer sufficed). NULL otherwise (still pending, or
    # resolved by a person — that is derived from the accepted recommendation, not
    # stored here). Feeds the dashboard's self-resolution rate alongside the route.
    resolution_kind: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))


class Answer(Base):
    """An answer to a question (fuel for reuse path F-10 and learning)."""

    __tablename__ = "answers"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"), index=True)
    responder_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    body: Mapped[str | None] = mapped_column(Text)
    # Fixtures set this explicitly; runtime answers may omit it, so the DB stamps
    # the insert time (mirrors ``recommendations.created_at``).
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    reuse_count: Mapped[int | None] = mapped_column(Integer)
    was_helpful: Mapped[bool | None] = mapped_column(Boolean)
    # Not in the base ER but present in fixtures; drives ``answers_by_topic``.
    topic: Mapped[str | None] = mapped_column(String(255), index=True)


class Recommendation(Base):
    """Recommendation result and its outcome (the core of learning)."""

    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"), index=True)
    employee_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    rank: Mapped[int | None] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Float)
    reasons: Mapped[dict | None] = mapped_column(JSONB)
    outcome: Mapped[str | None] = mapped_column(String(32))
    # Needed by the scorer's ``load`` calc (recommendations in the last 7 days,
    # technical-spec §5). DB stamps it at insert; indexed for the recency window.
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    # When the asker acknowledged this decline in the notification bell (#E7).
    # NULL means "not yet seen"; only ever set on a rank==1, outcome=='declined'
    # row (the shape a decline notification is derived from).
    declined_seen_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    # #509: when the asker acknowledged this acceptance in the notification bell.
    # NULL means "not yet seen"; only ever set on an outcome=='accepted' row (at
    # most one per question — see ``Message`` below).
    accepted_seen_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    # #509: when the responder acknowledged this incoming request in the
    # notification bell. NULL means "not yet seen"; only ever set on the same
    # rank==1, outcome IS NULL shape the inbox (`data.inbox`) lists as pending.
    request_seen_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    # #518: when this row's outcome is 'referred', the employee the responder named
    # as more suitable ("他に適任者がいる → この人"). NULL for every other outcome.
    # Auditable record of the referral relationship (who referred whom); the actual
    # reroute seats the named person via c6_score's referral pin.
    referred_to_employee_id: Mapped[int | None] = mapped_column(
        ForeignKey("employees.id"), nullable=True, index=True
    )


class Message(Base):
    """A chat message on an accepted recommendation (承諾後のやり取り, #224).

    ``recommendation_id`` doubles as the thread id: a decline+reroute creates a
    NEW rank-1 recommendation row rather than mutating the old one, and only
    one recommendation per question ever reaches ``outcome == "accepted"``, so
    the accepted ``Recommendation.id`` is a stable, unique key for "this
    accepted collaboration's chat" (no separate thread table needed).
    """

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recommendation_id: Mapped[int] = mapped_column(ForeignKey("recommendations.id"), index=True)
    sender_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), index=True)


class EvalRun(Base):
    """A stored offline-evaluation snapshot (dashboard 推薦精度 source).

    Written by ``python -m tekijin.eval`` so the dashboard can surface the latest
    Top-1 / Recall@3 without re-running the (heavy) evaluation on every request.
    """

    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now(), index=True)
    top1_accuracy: Mapped[float | None] = mapped_column(Float)
    recall_at_3: Mapped[float | None] = mapped_column(Float)
    mrr: Mapped[float | None] = mapped_column(Float)
    route_accuracy: Mapped[float | None] = mapped_column(Float)
    n_ranked: Mapped[int | None] = mapped_column(Integer)
    n_routed: Mapped[int | None] = mapped_column(Integer)


class Event(Base):
    """Per-stage measurement row (p50/p95 latency KPI)."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"))
    stage: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    ended_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    meta: Mapped[dict | None] = mapped_column(JSONB)


class Feedback(Base):
    """One "the AI got this wrong / I changed it" signal from the asking side (#237).

    The first-class learning signal the runtime used to throw away: the asker's
    correction of the AI's interpretation (C1), recommendation (C6), or draft (C7).
    Kept SEPARATE from ``events`` (which is latency measurement) on purpose. ``stage``
    is ``c1`` / ``c6`` / ``c7``; ``kind`` names the correction (e.g. ``draft_edited``);
    ``target`` is the thing corrected (a field name, a person id); ``payload`` carries
    the specifics (e.g. the generated vs. sent draft). ``actor_id`` is who gave it.
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[str | None] = mapped_column(ForeignKey("questions.id"), index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    stage: Mapped[str] = mapped_column(String(8))
    kind: Mapped[str] = mapped_column(String(32))
    target: Mapped[str | None] = mapped_column(String(64))
    payload: Mapped[dict | None] = mapped_column(JSONB)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id"), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


class ProjectMember(Base):
    """Project membership with role (lead 0.8 / member 0.5).

    PK is ``(project_id, employee_id)`` — an employee holds exactly one role per
    project, so ``role`` is a plain column, not part of the key. Keeping ``role``
    out of the PK prevents the same person being both lead and member on one
    project (which would double-count evidence in the C-layer graph).
    """

    __tablename__ = "project_members"
    __table_args__ = (
        CheckConstraint("role IN ('lead', 'member')", name="ck_project_members_role"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), primary_key=True)
    employee_id: Mapped[int] = mapped_column(
        ForeignKey("employees.id"), primary_key=True, index=True
    )
    role: Mapped[str] = mapped_column(String(32))

    project: Mapped[Project] = relationship(back_populates="members")


class Document(Base):
    """Internal document (demotion path, low priority)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(255))
    updated_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))


# --------------------------------------------------------------------------- #
# C. Expertise graph layer
# --------------------------------------------------------------------------- #
class PersonTopicEdge(Base):
    """Weighted person x topic expertise edge (recomputable from evidence)."""

    __tablename__ = "person_topic_edges"

    person_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), primary_key=True, index=True)
    topic_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    weight: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    evidence_count: Mapped[int | None] = mapped_column(Integer)
    last_updated: Mapped[dt.datetime | None] = mapped_column(DateTime)


class Evidence(Base):
    """A single piece of evidence backing an expertise edge."""

    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[int] = mapped_column(ForeignKey("employees.id"), index=True)
    topic_id: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str | None] = mapped_column(String(32))
    base_score: Mapped[float | None] = mapped_column(Float)
    weight_contrib: Mapped[float | None] = mapped_column(Float)
    ts: Mapped[dt.datetime | None] = mapped_column(DateTime)


# --------------------------------------------------------------------------- #
# D. Knowledge layer (#357 — tacit → explicit)
# --------------------------------------------------------------------------- #
class KnowledgeUnit(Base):
    """A structured unit of formalised knowledge extracted from raw data (#357).

    The product pivot ([[tekijin-product-direction]]) is to turn tacit knowledge
    into *explicit, reusable* units rather than searching raw text. The PoC unit is
    the **case** (``kind='case'``): ``problem`` (状況・課題) → ``action`` (打ち手) →
    ``result`` (結果). ``procedure`` / ``decision`` are reserved in the check but not
    yet produced.

    Every unit keeps its ``(source_type, source_id)`` provenance back to the raw
    record it was extracted from — a unit is never stored without it, so a
    hallucinated unit has no home. ``(source_type, source_id)`` is UNIQUE, making
    re-extraction an idempotent upsert (one source record → at most one unit for
    the PoC). ``review_status`` gates a unit into the retrieval/self-answer path:
    only ``approved`` units are trusted (the human-review導線 is #354). ``topics``
    reuses the same 22-word vocabulary as the eval gold, so knowledge retrieval and
    the existing scorer speak one language. ``embedding`` indexes the *unit* (the
    structure), not the raw text — filled by a later ingestion slice, hence NULL
    at insert.
    """

    __tablename__ = "knowledge_units"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('case', 'procedure', 'decision')", name="ck_knowledge_units_kind"
        ),
        CheckConstraint(
            "review_status IN ('unreviewed', 'approved', 'rejected')",
            name="ck_knowledge_units_review_status",
        ),
        # Provenance is unique so re-extraction upserts rather than duplicates.
        Index(
            "uq_knowledge_units_source",
            "source_type",
            "source_id",
            unique=True,
        ),
        # ``knowledge_units_by_topics`` filters with ``topics && :topics``; GIN
        # accelerates the array-overlap the same way it does for ``questions``.
        Index("ix_knowledge_units_topics", "topics", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16))
    problem: Mapped[str | None] = mapped_column(Text)
    action: Mapped[str | None] = mapped_column(Text)
    result: Mapped[str | None] = mapped_column(Text)
    topics: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    industry: Mapped[str | None] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float | None] = mapped_column(Float)
    review_status: Mapped[str] = mapped_column(String(16), server_default="unreviewed")
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())


__all__ = [
    "EMBEDDING_DIM",
    "Employee",
    "EmployeeProfile",
    "AiChatHistory",
    "EmployeeChatHistory",
    "DailyReport",
    "Project",
    "Certification",
    "Skill",
    "Question",
    "Answer",
    "Recommendation",
    "Message",
    "EvalRun",
    "Event",
    "ProjectMember",
    "Document",
    "PersonTopicEdge",
    "Evidence",
    "KnowledgeUnit",
]

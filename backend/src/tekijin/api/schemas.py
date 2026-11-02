"""Pydantic v2 request / response / SSE-data contracts for the API boundary.

Every value crossing the HTTP boundary is validated through one of these models
(model-definition §4). ``asker_id`` is an ``int`` to match the DB. The SSE data
models mirror the events emitted by :mod:`tekijin.api.events`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from tekijin.scorer.topics import TOPIC_VOCABULARY

Outcome = Literal["accepted", "declined"]

# The asker's chosen consultation method. "chat" is the implicit default: an
# unset/legacy value is coalesced to it everywhere it is read.
ConsultMethod = Literal["direct", "chat"]

# #247: how far the 直接相談 got. "unresolved" is recorded but is NOT expertise
# evidence and never subtracts (断り≠非専門) — see collect_topic_evidence.
ConsultResolution = Literal["resolved", "partial", "unresolved"]


def normalize_consult_method(value: str | None) -> ConsultMethod:
    """Narrow a raw ``questions.consult_method`` value onto the API contract.

    The column is a bare ``VARCHAR(32)`` with no CHECK constraint, so an older
    client, a manual fix-up, or a rolled-back future value can leave something
    else in it. Without this, such a row 500s on GET /handoff and GET /inbox —
    the response model rejects it — for a hand-off that is otherwise perfectly
    serviceable. Everything downstream already reads the column as a two-way
    branch (``COALESCE(consult_method, 'chat') != 'direct'`` in
    :mod:`tekijin.data.messages`), so "anything that is not 直接相談 behaves as
    チャット" is the existing semantics, not a new rule.
    """

    return "direct" if value == "direct" else "chat"


# session_id doubles as the ``/events/{session_id}`` path segment and the graph
# ``thread_id``; constrain it to path-safe characters (no ``/``) so a created
# session is always reachable over GET /events.
_SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


def format_employee_id(employee_id: int) -> str:
    """Render an internal int employee id as the external ``"E###"`` string form.

    Model-definition §163-170: the external contract uses zero-padded string ids
    (``"E017"``). Internal/DB stays int; this is applied only at the API boundary
    (recommend events), mirroring the ``"E###"`` asker_id we accept on input.
    """

    return f"E{int(employee_id):03d}"


def coerce_employee_id(value: object) -> int:
    """Public wrapper: accept an int or the spec's ``"E###"`` employee id.

    Used by query-param endpoints (e.g. ``GET /inbox?responder_id=E017``) that
    need the same lenient coercion the request bodies apply to ``asker_id``.
    """

    return _coerce_asker_id(value)


def _coerce_asker_id(value: object) -> int:
    """Accept an int employee id or the spec's ``"E###"`` string form.

    The DB stores ``asker_id`` as an int; the product spec writes employee ids as
    ``"E200"``. We accept both at the boundary and normalise to int (the DB form),
    stripping an optional leading ``E``/``e``.
    """

    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        raise ValueError("asker_id must be an integer or 'E###' string")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ("E", "e"):
            text = text[1:]
        if text.isdigit():
            return int(text)
    raise ValueError("asker_id must be an integer or 'E###' string")


# --------------------------------------------------------------------------- #
# requests
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    """Start (or restart) a question for a session."""

    asker_id: int
    question: str = Field(min_length=1)
    session_id: str = Field(pattern=_SESSION_ID_PATTERN)

    @field_validator("asker_id", mode="before")
    @classmethod
    def _accept_e_prefixed_id(cls, value: object) -> int:
        # DB is int; ``"E200"`` (spec form) is accepted and converted here.
        return _coerce_asker_id(value)

    @field_validator("question")
    @classmethod
    def _trim_nonempty(cls, value: str) -> str:
        # Trim and reject whitespace-only questions at the boundary (422), so the
        # empty query never reaches C3 and surfaces as an SSE error.
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("question must not be blank")
        return trimmed


class ResumeRequest(BaseModel):
    """Resume a paused run: a responder ``outcome`` OR a clarification ``reply``.

    Exactly one of the two must be supplied — ``outcome`` answers a ``send``
    interrupt (accept/decline), ``reply`` answers a ``followup`` interrupt.
    """

    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    outcome: Outcome | None = None
    reply: str | None = None
    # Generation token for the outcome: the recommendation id the responder was
    # shown (from GET /handoff). If it no longer matches the session's current
    # primary (a decline→reroute moved the hand-off on, or a competing tab), the
    # outcome is stale and rejected with 409 so it cannot bind to a new candidate
    # (#94). ``None`` skips the check (older clients / clarification replies).
    recommendation_id: int | None = None
    # The responder's answer text, captured when they accept the hand-off (#274).
    # Persisted as an ``answers`` row so it fuels reuse (self-answer / prior_answer),
    # the asker's "回答が届きました" history, and the accumulation dashboard. Optional
    # and only meaningful with ``outcome == "accepted"`` — a decline carries no
    # answer, and older clients / the "direct" consult method may accept without one.
    # Bounded (matches ``supplement`` / message ``body``): the text is embedded and
    # stored in an unbounded Text column, so an unbounded body is a storage/CPU
    # foot-gun.
    answer_body: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _exactly_one(self) -> ResumeRequest:
        provided = [v for v in (self.outcome, self.reply) if v is not None]
        if len(provided) != 1:
            raise ValueError("provide exactly one of 'outcome' or 'reply'")
        if self.reply is not None and not self.reply.strip():
            raise ValueError("'reply' must be non-empty")
        # The generation token only qualifies an outcome; it is meaningless on a
        # clarification reply (matches the frontend's discriminated union) (#94).
        if self.reply is not None and self.recommendation_id is not None:
            raise ValueError("'recommendation_id' is only valid with an 'outcome'")
        # An answer body only belongs on an ACCEPTED hand-off (#274): a decline
        # or a clarification reply never carries one, so reject the mismatch
        # rather than silently dropping it.
        if self.answer_body is not None and self.outcome != "accepted":
            raise ValueError("'answer_body' is only valid with outcome 'accepted'")
        return self

    @property
    def clean_answer_body(self) -> str | None:
        """The trimmed answer body, or ``None`` when blank (treated as no answer)."""

        if self.answer_body is None:
            return None
        stripped = self.answer_body.strip()
        return stripped or None

    @property
    def resume_value(self) -> str:
        return self.outcome if self.outcome is not None else (self.reply or "")


class DocumentFallbackRequest(BaseModel):
    """Turn a completed document result into a person hand-off (#351)."""

    session_id: str = Field(pattern=_SESSION_ID_PATTERN)


class HandoffDraftRequest(BaseModel):
    """Persist the asker's edited hand-off draft for a session paused at ``send``.

    The asker reviews/edits the generated draft on 画面3 and confirms; this saves
    the edited text into the durable state so the responder (画面4, GET /handoff)
    reads the edited version rather than the original generation. Draft-only —
    it never touches the recommendation ids or the accept/decline outcome (#174).
    """

    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    draft: str = Field(min_length=1)
    # Defaults to "chat" so a client that predates this field (or an asker who
    # cancels the method popup) keeps the pre-existing implicit behaviour.
    consult_method: ConsultMethod = "chat"

    @field_validator("draft")
    @classmethod
    def _trim_nonempty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("draft must not be blank")
        return trimmed


class QuestionStructureRequest(BaseModel):
    """Ask the AI to tidy the session's raw question into the four hand-off fields
    (#475 Screen 01). On-demand and read-only — it reshapes the already-stored
    question, so the client only needs to name the session."""

    session_id: str = Field(pattern=_SESSION_ID_PATTERN)


class QuestionStructureResponse(BaseModel):
    """The re-drafted question (#475): 起きていること / 環境 / 試したこと / 詰まっている点.

    Every field can be empty — the model leaves a field blank rather than inventing
    what the question never stated (an invented 環境 would mislead the responder), and
    the asker fills or edits it before sending. Purely presentational: nothing here is
    persisted or fed back into routing.
    """

    session_id: str
    summary: str
    environment: str
    tried: str
    blocker: str


class HandoffSelectRequest(BaseModel):
    """Asker picks a different (of the currently shown) candidate as the
    hand-off target, reordering it to primary and regenerating the draft for
    them (#200)."""

    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    person_id: int

    @field_validator("person_id", mode="before")
    @classmethod
    def _accept_e_prefixed_person_id(cls, value: object) -> int:
        return _coerce_asker_id(value)


class HandoffExcludeRequest(BaseModel):
    """Asker excludes the current send target ("この人には聞かない"), rerouting to a
    freshly-scored next candidate (#260).

    ``person_id`` must be the current primary (the person being handed off): the
    reroute path declines the top pick and re-scores excluding them, so excluding
    a non-target shown candidate is rejected (422) rather than silently declining
    the target. The new candidate arrives over the open ``/events`` stream (like a
    responder decline), so the POST only acks.
    """

    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    person_id: int

    @field_validator("person_id", mode="before")
    @classmethod
    def _accept_e_prefixed_person_id(cls, value: object) -> int:
        return _coerce_asker_id(value)


class HandoffReferRequest(BaseModel):
    """Responder refers the hand-off to a person THEY name ("他に適任者がいる → X", #518).

    ``person_id`` is any employee (not restricted to the shown candidates): the
    responder personally vouches for them, so they are seated at rank 1 for the
    reroute regardless of how they'd score. The new hand-off + draft arrive over the
    open ``/events`` stream (like a decline), so the POST only acks. Rejected (422)
    when the named person is the asker or the current responder.
    """

    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    person_id: int

    @field_validator("person_id", mode="before")
    @classmethod
    def _accept_e_prefixed_person_id(cls, value: object) -> int:
        return _coerce_asker_id(value)


class HandoffCorrectRequest(BaseModel):
    """Asker corrects the AI's interpretation of their question ("解釈の訂正", #260).

    The ``supplement`` is folded into the question and the whole pipeline re-runs
    from C1 (re-understand → re-route → re-score → re-draft), exactly as the
    ``ask → c1_intent`` clarification edge does — but asker-initiated from the
    result screen rather than in response to a C2 followup. The re-run streams over
    ``/events``; the response only acks.
    """

    session_id: str = Field(pattern=_SESSION_ID_PATTERN)
    supplement: str = Field(min_length=1, max_length=2000)

    @field_validator("supplement")
    @classmethod
    def _trim_nonempty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("supplement must not be blank")
        return trimmed


class HandoffRedraftRequest(BaseModel):
    """Asker asks the AI to regenerate the hand-off draft ("下書きの作り直し", #260).

    Regenerates the draft for the CURRENT send target (no ``person_id`` — use
    ``/handoff/select`` or ``/handoff/exclude`` to change who), discarding any
    saved manual edit. The new draft arrives over the open ``/events`` stream, so
    the response only acks.
    """

    session_id: str = Field(pattern=_SESSION_ID_PATTERN)


class NotificationAckRequest(BaseModel):
    """Mark notifications as seen (#225 decline, #509 accepted/request_received).

    ``kind`` picks which ``Recommendation.<kind>_seen_at`` column — and which
    ownership check — the ack applies: "declined"/"accepted" are asker-side
    (``asker_id``), "request_received" is responder-side (``employee_id``).
    """

    kind: Literal["declined", "accepted", "request_received"] = "declined"
    asker_id: int | None = None
    employee_id: int | None = None
    ids: list[int] = Field(min_length=1)

    @field_validator("asker_id", "employee_id", mode="before")
    @classmethod
    def _accept_e_prefixed_id(cls, value: object) -> int | None:
        if value is None:
            return None
        return _coerce_asker_id(value)

    @model_validator(mode="after")
    def _owner_matches_kind(self) -> NotificationAckRequest:
        if self.kind in ("declined", "accepted"):
            if self.asker_id is None or self.employee_id is not None:
                raise ValueError(f"exactly 'asker_id' is required for kind {self.kind!r}")
        elif self.employee_id is None or self.asker_id is not None:
            raise ValueError("exactly 'employee_id' is required for kind 'request_received'")
        return self


# --------------------------------------------------------------------------- #
# responses
# --------------------------------------------------------------------------- #
class AckResponse(BaseModel):
    """Acknowledgement for /ask and /answer (the stream flows over /events)."""

    session_id: str
    status: str


class DeleteQuestionResponse(BaseModel):
    """Acknowledgement for DELETE /questions/{id} (#207)."""

    question_id: str
    deleted: bool


class ResolveQuestionResponse(BaseModel):
    """Acknowledgement for POST /questions/{id}/resolve (#159 self-resolution)."""

    question_id: str
    resolved: bool


class FeedbackRequest(BaseModel):
    """The asking side's correction of an AI output (#237 Phase 1).

    ``stage`` is which pipeline output was wrong (c1 interpretation / c6
    recommendation / c7 draft). ``actor_id`` is NOT accepted from the body — it is
    taken from the authenticated principal so a caller cannot attribute feedback to
    someone else.
    """

    stage: Literal["c1", "c6", "c7"]
    kind: str = Field(min_length=1, max_length=32)
    question_id: str | None = Field(default=None, max_length=64)
    session_id: str | None = Field(default=None, max_length=64)
    target: str | None = Field(default=None, max_length=64)
    payload: dict[str, Any] | None = None

    @field_validator("payload")
    @classmethod
    def _cap_payload_size(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        # Bound the JSONB blob so an authenticated caller cannot grow the table
        # unbounded (storage-exhaustion DoS). 16KB is ample for a correction note
        # or a generated-vs-sent draft pair.
        if value is not None:
            import json

            if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 16384:
                raise ValueError("payload is too large (max 16KB)")
        return value


class TopicVocabularyResponse(BaseModel):
    """The closed topic list the scorer joins on (#247).

    Served rather than duplicated in the frontend: the retrospective form makes the
    asker pick from this vocabulary, and a hard-coded copy would drift from
    ``scorer/topics.py`` silently — a topic that no longer exists matches no
    evidence and does nothing (#116).
    """

    topics: list[str]


class ConsultRetrospectiveRequest(BaseModel):
    """The asker's write-up of a face-to-face 直接相談 (#247).

    A "direct" consultation leaves no text behind, so F-10 (回答を索引に追加し
    専門性の推定を更新) has nothing to work with. This is that record.

    ``asker_id`` is NOT accepted from the body — it comes from the authenticated
    principal, the same rule as :class:`FeedbackRequest`. That matters more here
    than for feedback: this row becomes expertise EVIDENCE for ``responder_id``,
    so an attributable author is what stops it being a way to fabricate someone's
    standing.

    ``responder_id`` IS accepted from the body, but it is not trusted: the route
    requires it to equal the employee who ACCEPTED this question's hand-off. It is
    a confirmation of what the client was shown, not a choice — an author-only
    check ("who may write") would still have left "whom may they write about" open,
    which is a way to grant anyone up to ``OFFLINE_CONSULT_EVIDENCE_CAP`` × the
    offline-consult base score on any topic.

    ``topics`` is validated against ``TOPIC_VOCABULARY`` because the scorer JOINS
    on these strings — a free-text topic would match no evidence and silently do
    nothing (#116). ``asked`` is optional (#247 の項目2); the rest are required.
    """

    question_id: str = Field(min_length=1, max_length=64)
    responder_id: str = Field(min_length=1, max_length=32)
    # At most 3 of the 22-topic vocabulary. One consultation is about one thing;
    # a wide list would spread a single conversation's evidence across most of the
    # vocabulary, which is how the offline-consult cap gets reached without the
    # consultations behind it.
    topics: list[str] = Field(min_length=1, max_length=3)
    asked: str | None = Field(default=None, max_length=2000)
    answer_body: str = Field(min_length=1, max_length=4000)
    resolution: ConsultResolution

    @field_validator("topics")
    @classmethod
    def _topics_in_vocabulary(cls, value: list[str]) -> list[str]:
        unknown = [t for t in value if t not in TOPIC_VOCABULARY]
        if unknown:
            raise ValueError(f"未知のトピックです: {', '.join(unknown)}")
        # De-duplicate, keeping order: the same topic twice must not count twice.
        return list(dict.fromkeys(value))

    @field_validator("answer_body")
    @classmethod
    def _answer_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("得られた回答は必須です")
        return stripped


class ConsultResponder(BaseModel):
    """The person a retrospective may be written about (#247)."""

    person_id: str
    name: str


class ConsultRetrospectiveContext(BaseModel):
    """What GET /consult-retrospective/{session_id} tells the write-up form (#247).

    Deliberately NOT ``HandoffResponse``: that payload is the pending hand-off view
    and 404s as soon as the responder records an outcome — i.e. it stops existing
    exactly when the face-to-face consultation becomes possible. This one is read
    from SQL and stays valid afterwards.

    ``responder`` is ``None`` until someone accepts, ``already_recorded`` flips once
    a write-up exists; between them the client can tell "not yet consulted", "ready
    to write" and "already written" apart without guessing from error codes.
    """

    session_id: str
    question_id: str
    question: str
    consult_method: ConsultMethod
    responder: ConsultResponder | None = None
    already_recorded: bool = False


class ConsultRetrospectiveAck(BaseModel):
    """Acknowledgement for POST /consult-retrospective (#247)."""

    status: str
    consult_id: int


class FeedbackAck(BaseModel):
    """Acknowledgement for POST /feedback (#237)."""

    status: str


class EmployeeSummary(BaseModel):
    """One employee for the current-user switcher (id / name / dept).

    ``id`` is the external ``"E###"`` form (see :func:`format_employee_id`), the
    same shape accepted back as ``asker_id`` and used as the responder id for the
    inbox — so the frontend can round-trip the selected user without conversion.
    """

    id: str  # external "E###" form (see format_employee_id)
    name: str
    dept: str | None = None


class EmployeeListResponse(BaseModel):
    """Employee directory for the current-user switcher (no auth in the prototype)."""

    employees: list[EmployeeSummary] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# domain models (shared by SSE data and final response)
# --------------------------------------------------------------------------- #
class Reason(BaseModel):
    type: str
    detail: str


class Recommendation(BaseModel):
    person_id: str  # external "E###" form (see format_employee_id)
    name: str
    dept: str | None = None
    score: float
    confidence: str
    reasons: list[Reason] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# handoff (GET /handoff/{session_id}) — responder-facing view (product-spec 画面4)
# --------------------------------------------------------------------------- #
class HandoffAsker(BaseModel):
    """The asking employee, enriched for the responder-facing handoff view."""

    id: str  # external "E###" form (see format_employee_id)
    name: str | None = None
    dept: str | None = None


class HandoffResponse(BaseModel):
    """Responder-facing payload for a session paused at the ``send`` interrupt.

    Assembled from the durable checkpoint (question / asker / slots /
    recommendations / draft) plus DB aggregates (the responder's past-answer
    reuse). Read-only: fetching it does NOT advance the graph — the responder
    acts via ``POST /answer`` (outcome=accepted|declined).
    """

    session_id: str
    question: str
    # #247: the durable question id. The retrospective form needs it to attribute
    # the write-up, and it is the only identifier the asker's client can reach from
    # a session — the checkpoint carries it, so this is a projection, not a lookup.
    question_id: str | None = None
    asker: HandoffAsker
    topics: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    situation: str | None = None
    missing: list[str] = Field(default_factory=list)
    # The primary (handed-off) candidate — the person being asked — with the
    # selection reasons. ``None`` only in the degenerate no-candidate case.
    responder: Recommendation | None = None
    draft: str = ""
    reuse_count: int = 0
    helpful_answer_count: int = 0
    # The primary recommendation's id — a generation token the client echoes back
    # on POST /answer so a stale outcome (after a reroute / from a competing tab)
    # is rejected instead of binding to a new candidate (#94). None if no primary.
    recommendation_id: int | None = None
    # The asker's chosen consultation method; "chat" until they choose otherwise
    # via POST /handoff/draft.
    consult_method: ConsultMethod = "chat"


class HandoffSelectResponse(BaseModel):
    """The new primary responder + regenerated draft after a reselect (#200/#A1/#204)."""

    session_id: str
    responder: Recommendation
    draft: str
    recommendation_id: int


# --------------------------------------------------------------------------- #
# inbox (GET /inbox) — responder-facing list of pending handoffs (#123)
# --------------------------------------------------------------------------- #
class InboxItem(BaseModel):
    """One pending handoff awaiting the responder.

    ``session_id`` deep-links to ``/answer/{session_id}``; the payload is just
    enough to preview the question in the list (the full handoff loads on the
    answer screen).
    """

    session_id: str
    question_id: str
    question: str
    topics: list[str] = Field(default_factory=list)
    asker: HandoffAsker
    # Shown as a badge in the list so the responder can tell, BEFORE accepting,
    # whether accepting opens a chat or means they will be approached directly
    # (#245). Absent on the question = the implicit default, "chat".
    consult_method: ConsultMethod = "chat"
    created_at: str | None = None


class InboxResponse(BaseModel):
    """Questions currently awaiting a given responder (newest first)."""

    items: list[InboxItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# asker history (GET /questions?asker_id) — "最近あなたが解決した質問" (#125)
# --------------------------------------------------------------------------- #
class RecentQuestionItem(BaseModel):
    """One of the asker's own recent questions, with its resolution state.

    ``resolution`` says HOW it was resolved (or that it is still pending):
    ``"person"`` a responder took it (accepted rec / answer row),
    ``"self"`` the asker marked it solved without asking anyone (#159),
    ``"document"`` the document route self-resolved it (no human),
    ``"pending"`` still awaiting a hand-off. ``resolved`` is the boolean shortcut
    ``resolution != "pending"``.
    """

    question_id: str
    title: str
    resolved: bool = False
    resolution: Literal["person", "self", "document", "pending"] = "pending"
    responder_name: str | None = None
    # Deep-link target for re-viewing the result (/session/{session_id} replays
    # the run over /events). ``None`` for seeded history with no live session.
    session_id: str | None = None
    created_at: str | None = None


class RecentQuestionsResponse(BaseModel):
    """The asker's most recent questions (newest first)."""

    items: list[RecentQuestionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# knowledge list (GET /knowledge) — company-wide, not scoped to one asker
# (#293, #301)
# --------------------------------------------------------------------------- #
class KnowledgeItem(BaseModel):
    """One piece of accumulated knowledge: an answered question OR an internal
    document — the same two kinds a self-answer (#291) cites.

    ``source_id`` is exactly a citation's ``SourceCitation.source_id`` for the
    same ``kind`` (``Answer.id`` for ``"qa"``, ``Document.id`` for
    ``"document"``), so a chat citation and a knowledge-list item can point at
    the same stable entity. ``resolved_at`` is the item's own timestamp — the
    ANSWER's (when it was given, not when the question was asked) for
    ``"qa"``, the document's ``updated_at`` for ``"document"``. The
    ``"document"``-only fields (``question_id``, ``session_id``, responder,
    topics) are ``None``/empty for that kind.
    """

    source_id: str
    kind: Literal["qa", "document", "knowledge"]
    title: str
    summary: str = ""
    topics: list[str] = Field(default_factory=list)
    responder_name: str | None = None
    responder_department: str | None = None
    resolved_at: str | None = None
    question_id: str | None = None
    session_id: str | None = None


class KnowledgeSummary(BaseModel):
    """Side-panel aggregate stats — reuses existing dashboard aggregates.

    Per-responder aggregates are deliberately NOT included — that view belongs
    to ``/dashboard``, not a knowledge browser (PR #340 review).
    """

    total_items: int
    self_resolution_rate: float


class KnowledgeListResponse(BaseModel):
    """GET /knowledge payload: one page of (filtered) items plus a global summary.

    ``total_matching`` is the count of items matching the current filters
    BEFORE the ``offset``/``limit`` page cut — what the frontend paginates a
    search's results with (#293, #301).
    """

    items: list[KnowledgeItem] = Field(default_factory=list)
    total_matching: int = 0
    summary: KnowledgeSummary


# --------------------------------------------------------------------------- #
# notification bell (GET /notifications, POST /notifications/ack)
# (#E7 declined, #509 accepted / request_received)
# --------------------------------------------------------------------------- #
class Notification(BaseModel):
    """One not-yet-seen notification (newest first).

    ``kind``-specific fields are optional and only populated for their own
    kind: ``declined_person_name`` (declined), ``responder_name`` +
    ``consult_method`` (accepted — the latter decides whether the frontend
    links to the chat thread or the session directly), ``asker_name``
    (request_received).
    """

    kind: Literal["declined", "accepted", "request_received"]
    id: int  # the Recommendation row's id (also the ack target)
    question_id: str
    session_id: str | None = None
    message: str
    created_at: str | None = None
    declined_person_name: str | None = None
    responder_name: str | None = None
    consult_method: ConsultMethod | None = None
    asker_name: str | None = None


class NotificationsResponse(BaseModel):
    items: list[Notification] = Field(default_factory=list)


class NotificationAckResponse(BaseModel):
    acknowledged: int


# --------------------------------------------------------------------------- #
# chat (GET/POST /messages) — accepted-recommendation threads (#224)
# --------------------------------------------------------------------------- #
class MessageSendRequest(BaseModel):
    """Send one chat message on an accepted-recommendation thread."""

    thread_id: int
    sender_id: int
    # Unlike /ask and /handoff/draft (one per session), chat is a repeat-send
    # surface, so an unbounded body is a storage-growth foot-gun rather than a
    # theoretical one. 2000 chars is well past any real message.
    body: str = Field(min_length=1, max_length=2000)

    @field_validator("sender_id", mode="before")
    @classmethod
    def _accept_e_prefixed_id(cls, value: object) -> int:
        return _coerce_asker_id(value)

    @field_validator("body")
    @classmethod
    def _trim_nonempty(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("body must not be blank")
        return trimmed


class MessageItem(BaseModel):
    """One chat message, sender in the external ``"E###"`` form."""

    id: int
    thread_id: int
    sender_id: str
    body: str
    created_at: str


class MessageThreadSummary(BaseModel):
    """One accepted thread for the chat list (newest activity first)."""

    thread_id: int
    question_id: str
    question_title: str
    counterpart: HandoffAsker
    last_message: str | None = None
    last_message_at: str | None = None
    created_at: str


class MessageThreadListResponse(BaseModel):
    items: list[MessageThreadSummary] = Field(default_factory=list)


class MessageThreadDetail(BaseModel):
    """One thread's full history, oldest first."""

    thread_id: int
    question_id: str
    question_title: str
    counterpart: HandoffAsker
    messages: list[MessageItem] = Field(default_factory=list)
    # Deep link to this pair's shared Slack channel (#hand-off-chat) — present
    # only once one exists (both parties linked Slack and a "chat" hand-off
    # between them was accepted). The channel is shared across every thread
    # between the two, so this link is the same regardless of which of their
    # threads it's fetched from.
    slack_channel_url: str | None = None


# --------------------------------------------------------------------------- #
# document detail (GET /documents/{doc_id}) — the cited internal document (#143)
# --------------------------------------------------------------------------- #
class DocumentDetail(BaseModel):
    """One internal document's full content, for the document viewer."""

    id: str
    title: str | None = None
    body: str | None = None
    source: str | None = None
    updated_at: str | None = None


# --------------------------------------------------------------------------- #
# SSE event data
# --------------------------------------------------------------------------- #
class UnderstoodData(BaseModel):
    topics: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    situation: str | None = None
    question_type: str | None = None
    confidence: float = 0.0
    # #475 Screen 01: how many OTHER askers previously asked in the same topic area
    # (reassurance). 0 when the feature is off or nobody else has — the client hides
    # the message at 0.
    similar_asker_count: int = 0


class FollowupData(BaseModel):
    question: str
    missing: list[str] = Field(default_factory=list)


class RouteData(BaseModel):
    route: str
    reason: str
    confidence: float


class RecommendData(BaseModel):
    recommendations: list[Recommendation] = Field(default_factory=list)


class DraftData(BaseModel):
    draft: str


class DoneData(BaseModel):
    status: str
    answer: str | None = None
    # This run segment's processing time in ms (technical-spec §7 / #177). None
    # when unavailable (e.g. a replayed terminal has no live timing).
    latency_ms: int | None = None


class SourceCitation(BaseModel):
    """One source a self-answer (#291) drew from, so the chat can show where it came from.

    The frontend branches on ``kind``, and its fallback branch reads 「過去の回答」 —
    so an unlisted kind is not merely unstyled, it is MISLABELLED as a past answer
    (#366). Keep this comment and ``SourceCitation.kind`` in
    ``frontend/src/lib/api-types.ts`` in step.
    """

    source_id: str
    # "qa" (past Q&A) | "document" (internal doc) | "daily" (daily report, #433)
    # | "knowledge" (structured knowledge unit, #357 — emitted by `knowledge_answer`)
    # | "source" (something the organization ingested, #81)
    kind: str
    # Only a "source" has these.
    title: str | None = None
    url: str | None = None


class ReferenceData(BaseModel):
    """#413: a cited answer surfaced ALONGSIDE a person hand-off ("参考: 過去の類似
    回答"). Emitted on the person route before ``recommend`` when a grounded past
    answer exists — additive, never a substitute for the hand-off."""

    answer: str
    citations: list[SourceCitation] = Field(default_factory=list)


class MessageData(BaseModel):
    status: str
    message: str
    # For the ``document`` route: the id of the cited document, so the client can
    # open it (GET /documents/{doc_id}). ``None`` for every non-document terminal.
    doc_id: str | None = None
    # Ranked during the document route but not persisted/shown as a hand-off until
    # the asker explicitly chooses this person. None means there is no safe CTA.
    fallback_responder: Recommendation | None = None
    # #291: for the ``self_answered`` terminal, the sources the answer cited — the
    # chat renders a link per entry. Empty for every other terminal.
    citations: list[SourceCitation] = Field(default_factory=list)
    # This run segment's processing time in ms (#177); None on a replay.
    latency_ms: int | None = None


class ErrorData(BaseModel):
    error: str


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #
class ResponderLoad(BaseModel):
    employee_id: int
    name: str
    answer_count: int


class TopicCount(BaseModel):
    topic: str
    count: int


class OutcomeCounts(BaseModel):
    """Aggregate recommendation outcomes (no per-record enumeration)."""

    accepted: int = 0
    declined: int = 0
    pending: int = 0  # shown but not yet accepted/declined


class EvalSnapshot(BaseModel):
    """The latest stored offline-evaluation metrics (dashboard 推薦精度)."""

    top1_accuracy: float | None = None
    recall_at_3: float | None = None
    mrr: float | None = None
    route_accuracy: float | None = None
    created_at: str | None = None


class ProcessingLatency(BaseModel):
    """p50/p95 of per-question AI processing time in ms (product-spec §9 / #177).

    Computed from the ``events`` table (sum of stage durations per question, so
    human-wait gaps are excluded). ``sample_size`` is the number of questions with
    recorded timing; p50/p95 are None until there is any.
    """

    p50_ms: int | None = None
    p95_ms: int | None = None
    sample_size: int = 0


class FeedbackByStage(BaseModel):
    """Feedback counts per pipeline stage + total (#237 — どの段でどれだけずれているか)."""

    c1: int = 0
    c6: int = 0
    c7: int = 0
    total: int = 0


class MonthlyCount(BaseModel):
    """One point of the accumulation trend (``"2026-09"``, count)."""

    month: str
    count: int


class KnowledgeAccumulation(BaseModel):
    """How much tacit knowledge became explicit, and whether the loop is closing (#294).

    Counts only what the RUNTIME produced — captured answers (#274) and 直接相談
    retrospectives (#247) — never the seeded corpus, so a freshly seeded database
    reads 0 rather than flattering itself with fixture rows.

    ``capture_rate`` is the recovery rate (暗黙知の回収率): of the hand-offs accepted
    this month, the share that left knowledge behind. Raw counts only ever grow;
    this is the one that can fall, which is what makes it worth showing.
    """

    this_month: int = 0
    last_month: int = 0
    captured_answers: int = 0
    consult_retrospectives: int = 0
    accepted_handoffs: int = 0
    capture_rate: float = 0.0
    monthly: list[MonthlyCount] = Field(default_factory=list)


class DashboardResponse(BaseModel):
    """Aggregate-only view (counts / distributions / ratios).

    Deliberately carries NO per-record listing (product-spec §241-251: the
    dashboard summarises usage, it is not a monitoring/audit log of individual
    recommendations).
    """

    total_employees: int
    total_questions: int
    total_answers: int
    recommendation_count: int
    recommendation_outcomes: OutcomeCounts = Field(default_factory=OutcomeCounts)
    acceptance_rate: float = 0.0  # accepted / (accepted + declined), 0 when none decided
    # product-spec 画面5 headline metrics.
    self_resolution_rate: float = 0.0  # 補助経路で解決した割合（route 記録から）
    avg_resolution_hours: float | None = None  # 平均解決時間（未解決なら None）
    top_responder_share: float = 0.0  # 上位1名集中率（負荷分散）
    processing_latency: ProcessingLatency = Field(default_factory=ProcessingLatency)  # p50/p95 (ms)
    latest_eval: EvalSnapshot | None = None  # 推薦精度（未計測なら None）
    answers_per_responder: list[ResponderLoad] = Field(default_factory=list)
    topic_distribution: list[TopicCount] = Field(default_factory=list)
    feedback_by_stage: FeedbackByStage = Field(default_factory=FeedbackByStage)  # #237 段別ズレ件数
    # #294: 蓄積メトリクス（形式知化された量と、取次ぎからの回収率）
    knowledge_accumulation: KnowledgeAccumulation = Field(default_factory=KnowledgeAccumulation)


# --------------------------------------------------------------------------- #
# auth (#241)
# --------------------------------------------------------------------------- #
class LoginRequest(BaseModel):
    """Credentials posted to ``POST /auth/login`` (email + password)."""

    email: str = Field(min_length=1, max_length=320)
    password: str = Field(min_length=1, max_length=1024)


class PrincipalResponse(BaseModel):
    """The authenticated identity. ``id`` is ``None`` for the admin (not an
    employee); otherwise the external ``"E###"`` form."""

    id: str | None
    name: str
    dept: str | None
    is_admin: bool


class LoginResponse(BaseModel):
    """A successful login: the bearer token plus the resolved principal."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    principal: PrincipalResponse


# --------------------------------------------------------------------------- #
# Slack integration (chat -> Slack DM notification)
# --------------------------------------------------------------------------- #
class SlackAuthorizeUrlResponse(BaseModel):
    """The "Sign in with Slack" URL for the frontend to navigate the browser to."""

    url: str


class SlackLinkCompleteRequest(BaseModel):
    """The short-lived token the OAuth callback handed to the frontend (#494)."""

    pending_token: str = Field(min_length=1, max_length=4096)


class SlackStatusResponse(BaseModel):
    """Whether the acting employee currently has a linked Slack account."""

    linked: bool


class SlackUnlinkResponse(BaseModel):
    """Ack for ``POST /slack/unlink``."""

    ok: bool = True


class SlackUserSyncResponse(BaseModel):
    """What one run of the Slack directory sync did (``POST /slack/sync-users``).

    ``skipped`` is deliberately part of the response rather than log-only: most
    runs of a healthy sync change nothing, so "0 linked" is the normal answer and
    carries no information on its own. The reasons are what tell an operator
    whether nobody needed linking, or whether `users:read.email` is missing and
    every single member arrived without an address.
    """

    created: int = 0
    linked: int
    unlinked: int
    skipped: dict[str, int] = Field(default_factory=dict)

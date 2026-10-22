"""HTTP routes: /ask, /answer, /events (SSE), /dashboard.

``/ask`` starts a NEW question; ``/answer`` RESUMES a paused run (a clarification
reply or a responder outcome). Both enqueue the next input and return an ack; the
run streams over ``/events/{session_id}``. Resume-vs-new-question and the pending
interrupt kind are validated against the durable checkpointer state (409/422), so
a stray /ask cannot overwrite a paused run and an outcome cannot be mis-delivered
to a clarification.

All endpoints now require authentication (#241): reads and writes go through the
``require_principal`` / ``require_admin`` seam in ``tekijin.auth.deps`` (the concrete
realization of the ``require_reader`` seam ADR-0005 anticipated). ``/dashboard`` and
``/employees`` are admin-only; ``/ask``, ``/questions`` and ``/inbox`` additionally
bind identity so a non-admin may only act as themselves (``require_can_act_as``).
Unexpected errors are uniformly masked as a generic 500 via ``_generic_500`` (#146).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import logging
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy.exc import IntegrityError
from sse_starlette import EventSourceResponse

from tekijin.api import schemas
from tekijin.api.service import (
    AgentService,
    AskerNotFound,
    HandoffNotFound,
    ServiceBusy,
    SessionConflict,
    SessionInvalid,
)
from tekijin.auth.deps import (
    require_admin,
    require_can_act_as,
    require_principal,
    require_principal_sse,
    require_session_participant,
)
from tekijin.auth.principal import Principal
from tekijin.channels.notifier import get_notifier
from tekijin.config import get_settings
from tekijin.data.consult import (
    accepted_responder_id,
    has_retrospective,
    retrospective_context,
)
from tekijin.data.dashboard import dashboard_summary
from tekijin.data.db import session_scope
from tekijin.data.documents import get_document
from tekijin.data.feedback import record_feedback
from tekijin.data.handoff import question_consult_method
from tekijin.data.history import question_asker_id, recent_questions_for_asker
from tekijin.data.inbox import pending_handoffs_for_responder
from tekijin.data.knowledge_library import get_qa_detail, list_knowledge
from tekijin.data.messages import (
    create_message,
    messages_for_thread,
    thread_parties,
    threads_for_employee,
)
from tekijin.data.notifications import (
    notifications_for_asker,
    pending_request_notifications_for_responder,
)
from tekijin.data.repository import Repository
from tekijin.data.slack_channel_links import get_channel_link
from tekijin.data.writes import (
    ack_accepted_notifications,
    ack_decline_notifications,
    ack_request_notifications,
    delete_question,
    mark_self_resolved,
    record_offline_consult,
)
from tekijin.scorer.topics import TOPIC_VOCABULARY

logger = logging.getLogger(__name__)

router = APIRouter()


def _service(request: Request) -> AgentService:
    return request.app.state.agent_service


@contextlib.contextmanager
def _generic_500(route: str) -> Iterator[None]:
    """Log-and-mask an unexpected error as a generic 500 (no internal detail leaked).

    A deliberate :class:`HTTPException` (404/422/…) raised inside passes through
    unchanged; any OTHER exception is logged with a stack trace and reduced to a
    generic 500. This unifies the read endpoints (dashboard/employees/inbox/
    questions/documents) with the inline handling already on /ask, /answer and
    /handoff, so no read endpoint can surface an internal message (#146).
    """

    try:
        yield
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("%s failed", route)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc


@router.post("/ask", response_model=schemas.AckResponse)
def ask(
    req: schemas.AskRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.AckResponse:
    """Start a new question for ``session_id``; stream flows over /events.

    A non-admin may only ask as themselves; admin may ask as any employee (demo
    impersonation) — enforced by ``require_can_act_as`` on ``asker_id``.
    """

    require_can_act_as(principal, req.asker_id)
    try:
        _service(request).start_question(req.session_id, req.asker_id, req.question)
    except ServiceBusy as exc:
        # Backpressure (#180): the LLM run pool is full — shed with a graceful 503 +
        # Retry-After so the client can back off instead of hammering the GPU.
        logger.info("shedding /ask for session %s (busy): %s", req.session_id, exc)
        raise HTTPException(
            status_code=503,
            detail="現在混雑しています。少し待ってから、もう一度お試しください。",
            headers={"Retry-After": "5"},
        ) from exc
    except SessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except AskerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # unexpected: log detail, return a generic 500
        logger.exception("POST /ask failed for session %s", req.session_id)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc
    return schemas.AckResponse(session_id=req.session_id, status="accepted")


@router.post("/answer", response_model=schemas.AckResponse)
def answer(
    req: schemas.ResumeRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.AckResponse:
    """Resume a paused run: a responder outcome or a clarification reply.

    Object-level auth (#241): only the session's asker/responder (or admin) may
    resume it — a non-participant is 403 even with a valid token and session id.

    Answer capture is further restricted (#274): an ``answer_body`` is attributed to
    the responder and re-surfaced to others as their answer, so ONLY the responder
    (or admin) may supply one. The asker is a valid participant (they own the
    clarification reply), but must not be able to forge an answer in the responder's
    name — a mismatch is 403.

    The OUTCOME itself carries the same rule, for the same reason. Accepting or
    declining is the responder's act, and ``recommendations.outcome`` is the only
    durable record that it happened: the dashboard's acceptance rate counts it, the
    responder inbox is filtered by it (``outcome IS NULL``), and #247 uses it to
    decide whom a retrospective may credit with expertise. Participant-level auth
    alone let the ASKER post ``outcome="accepted"`` for their own question — a
    hand-off "taken" by someone who never saw it. A ``reply`` (the clarification the
    asker owns) is unaffected.
    """

    asker_id, responder_id = _service(request).session_participants(req.session_id)
    require_session_participant(principal, asker_id, responder_id)
    if req.outcome is not None and not principal.may_act_as(responder_id or -1):
        raise HTTPException(
            status_code=403,
            detail="承諾・辞退を登録できるのは取次ぎ先の担当者本人のみです。",
        )
    if req.clean_answer_body is not None and not principal.may_act_as(responder_id or -1):
        raise HTTPException(
            status_code=403,
            detail="回答本文を登録できるのは取次ぎ先の担当者本人のみです。",
        )
    try:
        _service(request).submit_resume(
            req.session_id,
            outcome=req.outcome,
            reply=req.reply,
            recommendation_id=req.recommendation_id,
            answer_body=req.clean_answer_body,
        )
    except SessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # unexpected: log detail, return a generic 500
        logger.exception("POST /answer failed for session %s", req.session_id)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc
    return schemas.AckResponse(session_id=req.session_id, status="resumed")


@router.get("/events/{session_id}")
def events(
    session_id: str,
    request: Request,
    principal: Principal = Depends(require_principal_sse),
) -> EventSourceResponse:
    """SSE stream of the queued run's node updates (model-definition §4).

    Returns 404 only when there is neither a queued run nor a paused run for the
    session; a paused session reconnects (re-emits its pending interrupt).

    A browser ``EventSource`` cannot send an ``Authorization`` header, so this is
    the ONE endpoint that accepts the token as a ``?token=`` query parameter
    (``require_principal_sse``). Object-level auth (#241): only the session's
    asker/responder (or admin) may open the stream.
    """

    service = _service(request)
    asker_id, responder_id = service.session_participants(session_id)
    require_session_participant(principal, asker_id, responder_id)
    if not service.is_streamable(session_id):
        raise HTTPException(status_code=404, detail="no active run for this session")
    return EventSourceResponse(service.stream_events(session_id))


@router.get("/handoff/{session_id}", response_model=schemas.HandoffResponse)
def handoff(
    session_id: str,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.HandoffResponse:
    """Responder-facing handoff payload for a session paused at ``send``.

    Read-only view of the durable state (product-spec 画面4): the question, the
    asker, the filled-in slots, why this responder was chosen, the draft, and the
    responder's past-answer reuse. 404 when no handoff is pending (unknown /
    finished); 409 when the session is awaiting a clarification instead.

    Object-level auth (#241): only the session's asker/responder (or admin) may
    read it — a non-participant is 403 even with a valid session id.
    """

    service = _service(request)
    asker_id, responder_id = service.session_participants(session_id)
    require_session_participant(principal, asker_id, responder_id)
    try:
        return service.get_handoff(session_id)
    except HandoffNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # unexpected: log detail, return a generic 500
        logger.exception("GET /handoff failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc


@router.post("/handoff/document-fallback", response_model=schemas.AckResponse)
def document_fallback(
    req: schemas.DocumentFallbackRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.AckResponse:
    """Let the asker turn a completed document result into a person hand-off."""

    service = _service(request)
    asker_id, _ = service.session_participants(req.session_id)
    if asker_id is not None:
        # This transition belongs to the asker; a suggested responder must not be
        # able to volunteer the asker into a hand-off. Admin impersonation remains.
        require_can_act_as(principal, asker_id)
    try:
        service.request_document_fallback(req.session_id)
    except HandoffNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /handoff/document-fallback failed for %s", req.session_id)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc
    return schemas.AckResponse(session_id=req.session_id, status="handoff_queued")


@router.post("/handoff/draft", response_model=schemas.AckResponse)
def handoff_draft(
    req: schemas.HandoffDraftRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.AckResponse:
    """Persist the asker's edited hand-off draft (画面3) so the responder (画面4)
    reads the edited text (#174).

    Draft-only: never touches the recommendation/outcome state. 404 when no
    hand-off is pending (unknown / finished / already answered); 409 when the
    session is awaiting a clarification instead. A blank draft is rejected by the
    request schema (422) before it reaches the service. Object-level auth (#241):
    only the session's asker/responder (or admin) may edit the draft.
    """

    asker_id, responder_id = _service(request).session_participants(req.session_id)
    require_session_participant(principal, asker_id, responder_id)
    try:
        _service(request).save_handoff_draft(
            req.session_id, req.draft, req.consult_method, actor_id=principal.employee_id
        )
        if req.consult_method == "chat":
            meta = _service(request).pending_handoff_metadata(req.session_id)
            # Bound to locals rather than checked with `all(meta.get(k) is not None ...)`:
            # the values are `int | str | None`, and a comprehension-based guard narrows
            # nothing — neither for a type checker nor for a reader, who still has to
            # match the key strings in the guard against the ones used below (#441).
            # `meta_*` rather than `asker_id`/`responder_id`: those names are already
            # bound above from `session_participants` and are what the authorization
            # check used. Reusing them here would silently rebind the audited values.
            rec_id = meta.get("recommendation_id")
            meta_asker = meta.get("asker_id")
            meta_responder = meta.get("responder_id")
            if rec_id is not None and meta_asker is not None and meta_responder is not None:
                get_notifier().announce_handoff_request(
                    _service(request).session_factory,
                    session_id=req.session_id,
                    recommendation_id=int(rec_id),
                    thread_id=int(rec_id),
                    parties={
                        "asker_id": int(meta_asker),
                        "responder_id": int(meta_responder),
                    },
                    draft=req.draft,
                )
    except HandoffNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # unexpected: log detail, return a generic 500
        logger.exception("POST /handoff/draft failed for session %s", req.session_id)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc
    return schemas.AckResponse(session_id=req.session_id, status="draft_saved")


@router.post("/handoff/structure", response_model=schemas.QuestionStructureResponse)
def handoff_structure(
    req: schemas.QuestionStructureRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.QuestionStructureResponse:
    """Re-draft the asker's raw question into the four hand-off fields on demand (#475).

    The asker taps "AIに質問を整理してもらう" on the result screen; the model reshapes
    the already-stored question into 起きていること / 環境 / 試したこと / 詰まっている点,
    which the asker then edits before sending. Read-only and OUTSIDE the graph — it
    never advances or persists state, and never runs on the C1 critical path
    ([[tekijin-latency-and-streaming]]). 404 when the session has no question yet
    (unknown / not started). Object-level auth (#241): only the session's
    asker/responder (or admin) may structure it — the same rule as ``/handoff/draft``.
    """

    asker_id, responder_id = _service(request).session_participants(req.session_id)
    require_session_participant(principal, asker_id, responder_id)
    try:
        return _service(request).structure_question(req.session_id)
    except HandoffNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # unexpected: log detail, return a generic 500
        logger.exception("POST /handoff/structure failed for session %s", req.session_id)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc


@router.post("/handoff/select", response_model=schemas.HandoffSelectResponse)
def handoff_select(
    req: schemas.HandoffSelectRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.HandoffSelectResponse:
    """Asker picks a different (of the currently shown) candidate as the hand-off
    target; the draft is regenerated for them (#200).

    404 when no hand-off is pending (unknown / finished / already answered); 409
    when the session is awaiting a clarification instead; 422 when ``person_id``
    is not among the currently shown recommendations. Object-level auth (#241):
    only the session's asker/responder (or admin) may change the target — the
    same rule as ``/handoff/draft``, since this rewrites the durable hand-off.
    """

    asker_id, responder_id = _service(request).session_participants(req.session_id)
    require_session_participant(principal, asker_id, responder_id)
    try:
        return _service(request).select_handoff_candidate(req.session_id, req.person_id)
    except HandoffNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # unexpected: log detail, return a generic 500
        logger.exception("POST /handoff/select failed for session %s", req.session_id)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc


@router.post("/handoff/exclude", response_model=schemas.AckResponse)
def handoff_exclude(
    req: schemas.HandoffExcludeRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.AckResponse:
    """Asker excludes the current send target ("この人には聞かない"), rerouting to a
    freshly-scored next candidate (#260).

    Queues the same reroute a responder decline drives; the new candidate + draft
    arrive over the open ``/events`` stream, so this only acks. 404 when no
    hand-off is pending (unknown / finished / already answered); 409 when the
    session is awaiting a clarification instead; 422 when ``person_id`` is not the
    current hand-off target. Object-level auth (#241): only the session's
    asker/responder (or admin) may exclude, the same rule as ``/handoff/select``.
    The exclusion is recorded as a ``c6`` feedback signal (#237).
    """

    asker_id, responder_id = _service(request).session_participants(req.session_id)
    require_session_participant(principal, asker_id, responder_id)
    try:
        _service(request).exclude_handoff_target(
            req.session_id, req.person_id, actor_id=principal.employee_id
        )
    except HandoffNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # unexpected: log detail, return a generic 500
        logger.exception("POST /handoff/exclude failed for session %s", req.session_id)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc
    return schemas.AckResponse(session_id=req.session_id, status="reroute_queued")


@router.post("/handoff/refer", response_model=schemas.AckResponse)
def handoff_refer(
    req: schemas.HandoffReferRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.AckResponse:
    """Responder refers the hand-off to a person THEY name ("他に適任者がいる → X", #518).

    Records ``outcome="referred"`` on the current primary and reroutes with the named
    person seated at rank 1; the new hand-off + draft arrive over the open ``/events``
    stream, so this only acks. 404 when no hand-off is pending (unknown / finished /
    already answered); 409 when awaiting a clarification instead; 422 when the named
    person is unknown, the asker, or the current responder.

    Referring is the RESPONDER's act — like accept/decline (#274), only the session's
    responder (or admin) may do it, a stricter rule than the asker-side exclude/select:
    a 403 for any other participant even though they pass the participant check.
    """

    asker_id, responder_id = _service(request).session_participants(req.session_id)
    require_session_participant(principal, asker_id, responder_id)
    if not principal.may_act_as(responder_id or -1):
        raise HTTPException(
            status_code=403,
            detail="取り次ぎ先を推薦できるのは担当者本人のみです。",
        )
    try:
        _service(request).refer_handoff_target(
            req.session_id, req.person_id, actor_id=principal.employee_id
        )
    except HandoffNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # unexpected: log detail, return a generic 500
        logger.exception("POST /handoff/refer failed for session %s", req.session_id)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc
    return schemas.AckResponse(session_id=req.session_id, status="reroute_queued")


@router.post("/handoff/redraft", response_model=schemas.AckResponse)
def handoff_redraft(
    req: schemas.HandoffRedraftRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.AckResponse:
    """Asker asks the AI to regenerate the hand-off draft ("下書きの作り直し", #260).

    Regenerates the draft for the current send target (discarding any saved edit);
    the new draft arrives over the open ``/events`` stream, so this only acks. 404
    when no hand-off is pending (unknown / finished / already answered); 409 when
    the session is awaiting a clarification instead. Object-level auth (#241): only
    the session's asker/responder (or admin), the same rule as ``/handoff/draft``.
    The regeneration is recorded as a ``c7`` feedback signal (#237).
    """

    asker_id, responder_id = _service(request).session_participants(req.session_id)
    require_session_participant(principal, asker_id, responder_id)
    try:
        _service(request).regenerate_handoff_draft(req.session_id, actor_id=principal.employee_id)
    except HandoffNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # unexpected: log detail, return a generic 500
        logger.exception("POST /handoff/redraft failed for session %s", req.session_id)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc
    return schemas.AckResponse(session_id=req.session_id, status="redraft_queued")


@router.post("/handoff/correct", response_model=schemas.AckResponse)
def handoff_correct(
    req: schemas.HandoffCorrectRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.AckResponse:
    """Asker corrects the AI's interpretation of their question ("解釈の訂正", #260).

    Folds the ``supplement`` into the question and re-runs the pipeline from C1;
    the re-run streams over the open ``/events`` connection, so this only acks. 404
    when no hand-off is pending (unknown / finished / already answered); 409 when
    the session is awaiting a clarification instead; 422 when there is no question
    to correct. Object-level auth (#241): only the session's asker/responder (or
    admin), the same rule as ``/handoff/draft``. The correction is recorded as a
    ``c1`` feedback signal (#237).
    """

    asker_id, responder_id = _service(request).session_participants(req.session_id)
    require_session_participant(principal, asker_id, responder_id)
    try:
        _service(request).correct_interpretation(
            req.session_id, req.supplement, actor_id=principal.employee_id
        )
    except HandoffNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SessionConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except SessionInvalid as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # unexpected: log detail, return a generic 500
        logger.exception("POST /handoff/correct failed for session %s", req.session_id)
        raise HTTPException(status_code=500, detail="内部エラーが発生しました") from exc
    return schemas.AckResponse(session_id=req.session_id, status="reinterpret_queued")


@router.get(
    "/dashboard",
    response_model=schemas.DashboardResponse,
    dependencies=[Depends(require_admin)],
)
def dashboard(
    request: Request,
    top_responders: int = Query(default=5, ge=1, le=50),
) -> schemas.DashboardResponse:
    """Aggregate load / topic mix / recent activity for the dashboard.

    Admin-only (#241): the dashboard aggregates everyone's activity, so it is
    gated behind ``require_admin``.

    ``top_responders`` sizes the load-distribution list (#76). Bounded at 50: this
    endpoint is aggregate-only by design (product-spec §241-251), and an unbounded
    limit would turn the load list into a full per-employee roster — an audit view
    the dashboard deliberately is not. It also keeps one query from scanning the
    whole answers table on a large corpus.
    """

    with _generic_500("GET /dashboard"):
        with _service(request).session_factory() as session:
            data = dashboard_summary(
                session,
                top_responders=top_responders,
                # Injected so the month boundary follows the service clock the
                # rest of the run uses, not the process clock (#294).
                now=_service(request).now(),
            )
        return schemas.DashboardResponse(**data)


@router.get(
    "/employees",
    response_model=schemas.EmployeeListResponse,
    dependencies=[Depends(require_admin)],
)
def employees(request: Request) -> schemas.EmployeeListResponse:
    """List employees for the ADMIN's demo impersonation switcher (id / name / dept).

    Admin-only (#241): this directory is what lets the admin act as any employee,
    so it is gated behind ``require_admin``. Regular users never call it. ids are
    the external ``"E###"`` form to match the rest of the contract.
    """

    with _generic_500("GET /employees"):
        with _service(request).session_factory() as session:
            rows = Repository(session).list_employees()
        return schemas.EmployeeListResponse(
            employees=[
                schemas.EmployeeSummary(
                    id=schemas.format_employee_id(row.id),
                    name=row.name,
                    dept=row.department,
                )
                for row in rows
            ]
        )


@router.get("/inbox", response_model=schemas.InboxResponse)
def inbox(
    request: Request,
    responder_id: str = Query(min_length=1),
    principal: Principal = Depends(require_principal),
) -> schemas.InboxResponse:
    """Questions currently awaiting ``responder_id`` (the responder inbox, #123).

    ``responder_id`` accepts an int or the ``"E###"`` form (422 otherwise). A
    non-admin may only read their own inbox; admin may read anyone's (demo).
    Each item deep-links to ``/answer/{session_id}``; seeded history (no session)
    is excluded — there is no live handoff to open.
    """

    with _generic_500("GET /inbox"):
        try:
            rid = schemas.coerce_employee_id(responder_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="responder_id must be an int or 'E###'"
            ) from exc
        require_can_act_as(principal, rid)

        with _service(request).session_factory() as session:
            rows = pending_handoffs_for_responder(session, rid)
        return schemas.InboxResponse(
            items=[
                schemas.InboxItem(
                    session_id=row["session_id"],
                    question_id=row["question_id"],
                    question=row["question"],
                    topics=row["topics"],
                    asker=schemas.HandoffAsker(
                        id=schemas.format_employee_id(row["asker_id"]),
                        name=row["asker_name"],
                        dept=row["asker_dept"],
                    ),
                    consult_method=schemas.normalize_consult_method(row["consult_method"]),
                    created_at=row["created_at"],
                )
                for row in rows
            ]
        )


@router.get("/questions", response_model=schemas.RecentQuestionsResponse)
def questions(
    request: Request,
    asker_id: str = Query(min_length=1),
    limit: int = Query(5, ge=1, le=200),
    principal: Principal = Depends(require_principal),
) -> schemas.RecentQuestionsResponse:
    """The asker's own recent questions with resolution state (画面1 の一覧, #125).

    ``asker_id`` accepts an int or the ``"E###"`` form (422 otherwise). A non-admin
    may only read their own questions; admin may read anyone's (demo). ``limit``
    (default 5, the question screen's small recap; the history screen #208 requests
    up to 200) caps how many newest-first questions are returned.
    """

    with _generic_500("GET /questions"):
        try:
            aid = schemas.coerce_employee_id(asker_id)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail="asker_id must be an int or 'E###'"
            ) from exc
        require_can_act_as(principal, aid)

        with _service(request).session_factory() as session:
            rows = recent_questions_for_asker(session, aid, limit=limit)
        return schemas.RecentQuestionsResponse(
            items=[schemas.RecentQuestionItem(**row) for row in rows]
        )


@router.get("/knowledge", response_model=schemas.KnowledgeListResponse)
def knowledge(
    request: Request,
    q: str | None = Query(default=None, min_length=1),
    department: str | None = Query(default=None, min_length=1),
    topic: str | None = Query(default=None, min_length=1),
    since: dt.date | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(8, ge=1, le=200),
    principal: Principal = Depends(require_principal),
) -> schemas.KnowledgeListResponse:
    """Company-wide list of accumulated knowledge — answers AND documents.

    Open to every authenticated user — unlike ``/dashboard``, this is NOT
    admin-only, since the whole point (#293, #301) is that someone else's past
    answer (or an existing internal document) is discoverable ("これに近い話、
    前にも誰かが聞いてたはず"). Each item carries its own text (``summary``),
    not just who answered — and every item's ``source_id``/``kind`` match a
    self-answer's citation (#291), so a chat citation and this list point at
    the same entity. ``summary`` (the response field) reuses the dashboard's
    self-resolution rate rather than adding new aggregation logic. ``offset``/
    ``limit`` page a search's results (the frontend keeps the unsearched view
    to one unpaginated page of the latest ``limit`` items).
    """

    with _generic_500("GET /knowledge"):
        with _service(request).session_factory() as session:
            rows, total_matching, summary = list_knowledge(
                session,
                q=q,
                department=department,
                topic=topic,
                since=since,
                offset=offset,
                limit=limit,
            )
        return schemas.KnowledgeListResponse(
            total_matching=total_matching,
            items=[schemas.KnowledgeItem(**row) for row in rows],
            summary=schemas.KnowledgeSummary(**summary),
        )


@router.get("/knowledge/{source_id}", response_model=schemas.KnowledgeItem)
def knowledge_detail(
    source_id: str,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.KnowledgeItem:
    """Full detail of one past-Q&A knowledge item, keyed by ``Answer.id``.

    The ``"document"`` counterpart already has its own stable detail viewer at
    ``GET /documents/{doc_id}`` (#143) — this fills the gap #321's chat
    citation chip was left non-linked for (``kind="qa"`` had nowhere to go).
    404 when ``source_id`` is unknown.
    """

    with _generic_500("GET /knowledge/{source_id}"):
        with _service(request).session_factory() as session:
            row = get_qa_detail(session, source_id)
        if row is None:
            raise HTTPException(status_code=404, detail="knowledge item not found")
        return schemas.KnowledgeItem(**row)


@router.delete("/questions/{question_id}", response_model=schemas.DeleteQuestionResponse)
def delete_question_endpoint(
    request: Request,
    question_id: str,
    principal: Principal = Depends(require_principal),
) -> schemas.DeleteQuestionResponse:
    """Delete one of the asker's own past questions and its history (#207).

    Only the owning asker — or an admin (demo) — may delete. A missing question is
    a 404; a question owned by someone else is a 403 (``require_can_act_as``). The
    question and its FK children (answers / recommendations / events) are removed
    in one transaction; the lookup and the delete share it so ownership cannot
    change between the check and the write.
    """

    with _generic_500("DELETE /questions"):
        with session_scope(_service(request).session_factory) as session:
            owner = question_asker_id(session, question_id)
            if owner is None:
                raise HTTPException(status_code=404, detail="question not found")
            require_can_act_as(principal, owner)
            delete_question(session, question_id)
        # Existence was just confirmed above and the delete ran in the same
        # transaction, so the question is gone.
        return schemas.DeleteQuestionResponse(question_id=question_id, deleted=True)


@router.post("/questions/{question_id}/resolve", response_model=schemas.ResolveQuestionResponse)
def resolve_question_endpoint(
    request: Request,
    question_id: str,
    principal: Principal = Depends(require_principal),
) -> schemas.ResolveQuestionResponse:
    """Mark one of the asker's own questions self-resolved (#159).

    The "人を介さず解決した" UX signal: the asker got what they needed from a
    document / past answer and did not ask a person. Only the owning asker — or an
    admin — may mark it; a missing question is a 404, someone else's is a 403
    (``require_can_act_as``). Idempotent (first-wins): re-marking, or marking a
    question a responder already resolved, is a no-op that still acks. Feeds the
    dashboard self-resolution rate.
    """

    with _generic_500("POST /questions/{id}/resolve"):
        with session_scope(_service(request).session_factory) as session:
            owner = question_asker_id(session, question_id)
            if owner is None:
                raise HTTPException(status_code=404, detail="question not found")
            require_can_act_as(principal, owner)
            mark_self_resolved(session, question_id, _service(request).now())
        return schemas.ResolveQuestionResponse(question_id=question_id, resolved=True)


@router.post("/feedback", response_model=schemas.FeedbackAck)
def feedback(
    req: schemas.FeedbackRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.FeedbackAck:
    """Record the asking side's correction of an AI output (#237 Phase 1, hardened #263).

    The signal the runtime used to discard: "the interpretation / recommendation /
    draft is wrong". ``actor_id`` is taken from the authenticated principal (never
    the body), so feedback cannot be attributed to another user.

    Object-level authorization (#263): feedback is append-only, but a link to
    someone else's ``question_id`` / ``session_id`` would let a user pollute that
    target's learning signal / metrics. So a caller may only tag a target they own:
    a ``session_id`` requires being that session's asker/responder (or admin) while
    it is still live; a ``question_id`` requires being that question's asker (or
    admin). An UNKNOWN target link is silently dropped (recorded without it) rather
    than 403'd — so a known-not-owned target is a 403 while an unknown one is a 200,
    the SAME 403-confirms-ownership shape ``DELETE /questions`` and the other
    session-scoped endpoints already carry (no NEW enumeration oracle), and never
    the FK ``IntegrityError`` → 500 oracle the pre-#263 code called out.

    Rate limited per actor (#263) so the append-only signal cannot be flooded.
    """

    limiter = request.app.state.feedback_rate_limiter
    if not limiter.allow(f"feedback:{principal.employee_id}"):
        raise HTTPException(status_code=429, detail="フィードバックの送信が多すぎます。")

    # Session-scoped auth: only a live-session participant (or admin) may tag it.
    if req.session_id is not None:
        asker_id, responder_id = _service(request).session_participants(req.session_id)
        require_session_participant(principal, asker_id, responder_id)

    with _generic_500("POST /feedback"):
        with session_scope(_service(request).session_factory) as session:
            # Question-scoped auth: a KNOWN question may be tagged only by its owner
            # (or admin); an UNKNOWN id is dropped (no FK IntegrityError → 500, and
            # no existence oracle), recording the signal without the link.
            question_id = req.question_id
            if question_id is not None:
                owner = question_asker_id(session, question_id)
                if owner is None:
                    question_id = None
                else:
                    require_can_act_as(principal, owner)
            record_feedback(
                session,
                stage=req.stage,
                kind=req.kind,
                question_id=question_id,
                session_id=req.session_id,
                target=req.target,
                payload=req.payload,
                actor_id=principal.employee_id,
            )
        return schemas.FeedbackAck(status="recorded")


@router.get("/topics", response_model=schemas.TopicVocabularyResponse)
def topics(principal: Principal = Depends(require_principal)) -> schemas.TopicVocabularyResponse:
    """The closed topic vocabulary the scorer joins on (#247).

    Static — it comes from ``scorer/topics.py``, not the database. Served so the
    retrospective form can offer exactly the topics the scorer understands instead
    of shipping a copy that drifts (#116). Behind auth like every other endpoint;
    there is nothing sensitive here, but an unauthenticated hole is a hole.
    """

    return schemas.TopicVocabularyResponse(topics=list(TOPIC_VOCABULARY))


@router.get(
    "/consult-retrospective/{session_id}",
    response_model=schemas.ConsultRetrospectiveContext,
)
def consult_retrospective_context(
    session_id: str,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.ConsultRetrospectiveContext:
    """What the 直接相談 write-up form is built from (#247).

    Read from SQL rather than reusing ``GET /handoff``: that endpoint serves the
    PENDING hand-off out of the checkpoint and 404s the moment the responder
    records an outcome — which is precisely when a face-to-face consultation can
    finally have happened. Building the form on it made the feature reachable only
    in the window before the thing it documents took place.

    404 for a session with no question; object-level auth mirrors the write side —
    only the question's own asker (or an admin) may read it, since it names who
    they consulted.
    """

    with (
        _generic_500("GET /consult-retrospective"),
        session_scope(_service(request).session_factory) as session,
    ):
        ctx = retrospective_context(session, session_id)
        if ctx is None:
            raise HTTPException(status_code=404, detail="session not found")
        require_can_act_as(principal, ctx["asker_id"])
        responder = (
            schemas.ConsultResponder(
                person_id=schemas.format_employee_id(ctx["responder_id"]),
                name=ctx["responder_name"] or "",
            )
            if ctx["responder_id"] is not None
            else None
        )
        return schemas.ConsultRetrospectiveContext(
            session_id=session_id,
            question_id=ctx["question_id"],
            question=ctx["question"],
            consult_method=schemas.normalize_consult_method(ctx["consult_method"]),
            responder=responder,
            already_recorded=ctx["already_recorded"],
        )


@router.post("/consult-retrospective", response_model=schemas.ConsultRetrospectiveAck)
def consult_retrospective(
    req: schemas.ConsultRetrospectiveRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.ConsultRetrospectiveAck:
    """Record the asker's write-up of a face-to-face 直接相談 (#247).

    A "direct" consultation (#245) produces no chat transcript, so the knowledge
    it carried is lost and F-10 (回答を索引に追加し専門性の推定を更新) has nothing
    to index. This endpoint is that missing record — and, unlike ``/feedback``, it
    becomes EXPERTISE EVIDENCE for ``responder_id``, which is why the checks below
    are stricter than "append-only, rate limited":

    * ``asker_id`` is taken from the principal, never the body.
    * Only the QUESTION'S OWN ASKER (or an admin) may write one — that settles WHO
      may write.
    * ``responder_id`` must equal the employee who ACCEPTED this question's
      hand-off — that settles WHOM they may write about, which is the half that
      actually guards the score. Authorising the author alone still let anyone
      create a question and then credit any colleague (or themselves) with up to
      ``OFFLINE_CONSULT_EVIDENCE_CAP`` consultations' worth of evidence on any
      topic. Checking against the ACCEPTED row (not merely a shown recommendation)
      also means the id is a real employee, so a bogus one is a 422 here instead of
      an FK ``IntegrityError`` surfacing as a 500 (the shape #263 removed from
      ``/feedback``). ``POST /handoff/select`` sets the precedent: a person id from
      the client is checked against what the system actually offered.
    * ``responder_id`` may not be the asker themselves (422). Unreachable while the
      scorer excludes the asker from its own candidate pool, but stated locally so
      the guarantee does not depend on another module's invariant.
    * At most ONE write-up per question (409 on a second): exactly one hand-off is
      ever accepted, so "the consultation" is singular, and repeating it would let
      one real conversation fill the whole evidence cap.
    * The question must have been handed off as ``direct``. A chat consultation
      already has a transcript; a remembered summary layered on top of it is a
      second, weaker copy of the same exchange.
    * An unknown ``question_id`` is a 404 rather than a silently-dropped link (the
      way ``/feedback`` treats it): a retrospective with no question is not a
      weaker signal, it is an unverifiable one, so there is nothing to record.
    * Topics are constrained to ``TOPIC_VOCABULARY`` by the request model — the
      scorer joins on these strings, and free text would match nothing (#116).

    Rate limited per actor with the feedback limiter's budget (60/60s), under a
    SEPARATE key: both are asker-driven append-only writes at human speed, and one
    write per finished consultation is far below that ceiling.
    """

    if not get_settings().consult_retrospective_enabled:
        raise HTTPException(status_code=503, detail="ふりかえりの記録は現在無効です。")

    limiter = request.app.state.feedback_rate_limiter
    if not limiter.allow(f"consult-retrospective:{principal.employee_id}"):
        raise HTTPException(status_code=429, detail="ふりかえりの送信が多すぎます。")

    responder_id = _coerce_employee_id_or_422(req.responder_id, field="responder_id")

    with _generic_500("POST /consult-retrospective"):
        with session_scope(_service(request).session_factory) as session:
            owner = question_asker_id(session, req.question_id)
            if owner is None:
                raise HTTPException(status_code=404, detail="question not found")
            require_can_act_as(principal, owner)
            if (
                schemas.normalize_consult_method(question_consult_method(session, req.question_id))
                != "direct"
            ):
                raise HTTPException(
                    status_code=422,
                    detail="この依頼は直接相談ではないため、ふりかえりは記録できません。",
                )
            accepted = accepted_responder_id(session, req.question_id)
            if accepted is None:
                raise HTTPException(
                    status_code=422,
                    detail="この依頼はまだ受諾されていないため、ふりかえりは記録できません。",
                )
            if has_retrospective(session, req.question_id):
                # One accepted hand-off per question means one consultation; without
                # this, the same real conversation could be written up
                # ``OFFLINE_CONSULT_EVIDENCE_CAP`` times to reach the full cap.
                raise HTTPException(
                    status_code=409,
                    detail="この依頼のふりかえりは、すでに記録されています。",
                )
            if responder_id == owner:
                # Belt and braces. The check below would refuse this too — the
                # scorer never puts the asker in its own candidate pool, so they
                # can never be the accepting responder — but that is an invariant
                # of a DIFFERENT module. Stated here so "you cannot credit
                # yourself" is a local, readable rule rather than a consequence.
                raise HTTPException(
                    status_code=422,
                    detail="自分自身を相談相手として記録することはできません。",
                )
            if responder_id != accepted:
                raise HTTPException(
                    status_code=422,
                    detail="ふりかえりは、実際に相談を受けた相手についてのみ記録できます。",
                )
            try:
                consult_id = record_offline_consult(
                    session,
                    question_id=req.question_id,
                    responder_id=responder_id,
                    asker_id=principal.employee_id,
                    topics=req.topics,
                    asked=req.asked,
                    answer_body=req.answer_body,
                    resolution=req.resolution,
                )
            except IntegrityError as exc:
                # Two concurrent posts both clear the check above in separate
                # transactions; the loser hits uq_offline_consults_question. Answer
                # it the way the check does rather than letting a constraint
                # violation surface as a 500 (the shape #263 removed from
                # /feedback).
                raise HTTPException(
                    status_code=409,
                    detail="この依頼のふりかえりは、すでに記録されています。",
                ) from exc
        return schemas.ConsultRetrospectiveAck(status="recorded", consult_id=consult_id)


def _normalized_notification(row: dict[str, Any]) -> dict[str, Any]:
    """Apply the shared "NULL/anything but 'direct' means chat" rule (#509).

    Only "accepted" rows carry a raw ``consult_method`` (see
    ``data.notifications.pending_accepted_notifications_for_asker``); the
    normaliser is idempotent on the missing/``None`` case the other kinds pass.
    """

    if "consult_method" in row:
        row = {**row, "consult_method": schemas.normalize_consult_method(row["consult_method"])}
    return row


@router.get("/notifications", response_model=schemas.NotificationsResponse)
def notifications(
    request: Request,
    asker_id: str | None = Query(default=None, min_length=1),
    employee_id: str | None = Query(default=None, min_length=1),
    principal: Principal = Depends(require_principal),
) -> schemas.NotificationsResponse:
    """Notifications the caller hasn't seen yet, newest first.

    Exactly one of ``asker_id`` (declined + accepted outcomes on their own
    questions, #225/#509) or ``employee_id`` (incoming requests still
    awaiting their decision, #509) must be given — the two audiences never
    share a query. Declined is paired with the automatic reroute (#206): the
    system has already moved on to the next candidate by the time it fires,
    so it is informational, not a request to act. Both params accept an int
    or the ``"E###"`` form (422 otherwise). A non-admin may only read their
    own notifications; admin may read anyone's (#241, the same rule as
    ``/questions``/``/inbox``).
    """

    if (asker_id is None) == (employee_id is None):
        raise HTTPException(
            status_code=422, detail="specify exactly one of 'asker_id' or 'employee_id'"
        )

    with _generic_500("GET /notifications"):
        if asker_id is not None:
            aid = _coerce_employee_id_or_422(asker_id, field="asker_id")
            require_can_act_as(principal, aid)
            with _service(request).session_factory() as session:
                rows = notifications_for_asker(session, aid)
        elif employee_id is not None:
            eid = _coerce_employee_id_or_422(employee_id, field="employee_id")
            require_can_act_as(principal, eid)
            with _service(request).session_factory() as session:
                rows = pending_request_notifications_for_responder(session, eid)
        else:
            # Unreachable: the exactly-one check above guarantees one is set.
            raise HTTPException(
                status_code=422, detail="specify exactly one of 'asker_id' or 'employee_id'"
            )
        return schemas.NotificationsResponse(
            items=[schemas.Notification(**_normalized_notification(row)) for row in rows]
        )


@router.post("/notifications/ack", response_model=schemas.NotificationAckResponse)
def ack_notifications(
    req: schemas.NotificationAckRequest,
    request: Request,
    principal: Principal = Depends(require_principal),
) -> schemas.NotificationAckResponse:
    """Mark notifications as seen (#225 declined, #509 accepted/request_received).

    Acking is a write on the caller's own rows, so it carries the same
    act-as rule as the read above (#241) — without it any logged-in user could
    silently clear someone else's unread notifications. ``req``'s own
    validator already enforces that the id matching ``req.kind`` is present.
    """

    with _generic_500("POST /notifications/ack"):
        if req.kind == "declined" and req.asker_id is not None:
            require_can_act_as(principal, req.asker_id)
            with session_scope(_service(request).session_factory) as session:
                count = ack_decline_notifications(session, req.asker_id, req.ids)
        elif req.kind == "accepted" and req.asker_id is not None:
            require_can_act_as(principal, req.asker_id)
            with session_scope(_service(request).session_factory) as session:
                count = ack_accepted_notifications(session, req.asker_id, req.ids)
        elif req.kind == "request_received" and req.employee_id is not None:
            require_can_act_as(principal, req.employee_id)
            with session_scope(_service(request).session_factory) as session:
                count = ack_request_notifications(session, req.employee_id, req.ids)
        else:
            # Unreachable: NotificationAckRequest's validator enforces the
            # matching id is present for each kind.
            raise HTTPException(status_code=500, detail="invalid notification ack request")
        return schemas.NotificationAckResponse(acknowledged=count)


def _coerce_employee_id_or_422(value: str, *, field: str) -> int:
    try:
        return schemas.coerce_employee_id(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{field} must be an int or 'E###'") from exc


def _counterpart(parties: dict, employee_id: int) -> schemas.HandoffAsker:
    is_asker = employee_id == parties["asker_id"]
    return schemas.HandoffAsker(
        id=schemas.format_employee_id(parties["responder_id"] if is_asker else parties["asker_id"]),
        name=parties["responder_name"] if is_asker else parties["asker_name"],
        dept=parties["responder_dept"] if is_asker else parties["asker_dept"],
    )


@router.get("/messages/threads", response_model=schemas.MessageThreadListResponse)
def message_threads(
    request: Request,
    employee_id: str = Query(min_length=1),
    principal: Principal = Depends(require_principal),
) -> schemas.MessageThreadListResponse:
    """Accepted chat threads where ``employee_id`` is a party, newest activity first (#224).

    A non-admin may only list their own threads (#241, the same act-as rule as
    ``/questions`` and ``/inbox``). Without it the party check below would still
    pass for any id the caller cares to name — employee ids are the enumerable
    ``E###`` form, so that would expose everyone's conversations.
    """

    with _generic_500("GET /messages/threads"):
        eid = _coerce_employee_id_or_422(employee_id, field="employee_id")
        require_can_act_as(principal, eid)
        with _service(request).session_factory() as session:
            rows = threads_for_employee(session, eid)
        return schemas.MessageThreadListResponse(
            items=[
                schemas.MessageThreadSummary(
                    thread_id=row["thread_id"],
                    question_id=row["question_id"],
                    question_title=row["question_title"],
                    counterpart=schemas.HandoffAsker(
                        id=schemas.format_employee_id(row["counterpart_id"]),
                        name=row["counterpart_name"],
                        dept=row["counterpart_dept"],
                    ),
                    last_message=row["last_message"],
                    last_message_at=row["last_message_at"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]
        )


@router.get("/messages/threads/{thread_id}", response_model=schemas.MessageThreadDetail)
def message_thread_detail(
    thread_id: int,
    request: Request,
    employee_id: str = Query(min_length=1),
    principal: Principal = Depends(require_principal),
) -> schemas.MessageThreadDetail:
    """One thread's full history (#224). 404 if unaccepted or ``employee_id`` isn't a party.

    Non-parties get the same 404 as an unknown thread — never a 403 — so the
    response cannot be used to confirm a thread exists to someone uninvolved.
    Acting as someone else is refused first (403, #241): the 404-for-non-parties
    rule protects third parties, it is not a substitute for authenticating the
    caller.
    """

    with _generic_500("GET /messages/threads/{thread_id}"):
        eid = _coerce_employee_id_or_422(employee_id, field="employee_id")
        require_can_act_as(principal, eid)
        with _service(request).session_factory() as session:
            parties = thread_parties(session, thread_id)
            if parties is None or eid not in (parties["asker_id"], parties["responder_id"]):
                raise HTTPException(status_code=404, detail="thread not found")
            rows = messages_for_thread(session, thread_id)
            channel_link = get_channel_link(session, parties["asker_id"], parties["responder_id"])
        slack_channel_url = (
            f"https://slack.com/app_redirect"
            f"?channel={channel_link.slack_channel_id}&team={channel_link.slack_team_id}"
            if channel_link is not None
            else None
        )
        return schemas.MessageThreadDetail(
            thread_id=thread_id,
            question_id=parties["question_id"],
            question_title=parties["question_title"],
            counterpart=_counterpart(parties, eid),
            messages=[
                schemas.MessageItem(
                    id=row["id"],
                    thread_id=thread_id,
                    sender_id=schemas.format_employee_id(row["sender_id"]),
                    body=row["body"],
                    created_at=row["created_at"],
                )
                for row in rows
            ],
            slack_channel_url=slack_channel_url,
        )


@router.post("/messages", response_model=schemas.MessageItem)
def send_message(
    req: schemas.MessageSendRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    principal: Principal = Depends(require_principal),
) -> schemas.MessageItem:
    """Send one chat message on an accepted thread (#224). 404 if unaccepted or non-party.

    ``sender_id`` is bound to the authenticated principal (#241): without this a
    party could post as the OTHER party, and anyone could post as anyone.

    If this pair has a shared Slack channel (#hand-off-chat), the message is
    relayed into it as a background task after the response — Slack being
    slow or unreachable must never delay or fail sending the chat message
    itself (``ChannelNotifier.relay_message``; the Slack side is
    ``tekijin.slack.notify.relay_to_channel``, shared with the Slack-reply path, #388).
    """

    with _generic_500("POST /messages"):
        require_can_act_as(principal, req.sender_id)
        with session_scope(_service(request).session_factory) as session:
            parties = thread_parties(session, req.thread_id)
            if parties is None or req.sender_id not in (
                parties["asker_id"],
                parties["responder_id"],
            ):
                raise HTTPException(status_code=404, detail="thread not found")
            now = dt.datetime.now()  # noqa: DTZ005 - naive is intentional, matches created_at
            row = create_message(session, req.thread_id, req.sender_id, req.body, now)
            is_asker = req.sender_id == parties["asker_id"]
            sender_name = parties["asker_name"] if is_asker else parties["responder_name"]
            background_tasks.add_task(
                get_notifier().relay_message,
                _service(request).session_factory,
                employee_a=parties["asker_id"],
                employee_b=parties["responder_id"],
                sender_name=sender_name,
                body=req.body,
            )
        return schemas.MessageItem(
            id=row["id"],
            thread_id=row["thread_id"],
            sender_id=schemas.format_employee_id(row["sender_id"]),
            body=row["body"],
            created_at=row["created_at"],
        )


@router.get(
    "/documents/{doc_id}",
    response_model=schemas.DocumentDetail,
    dependencies=[Depends(require_principal)],
)
def document_detail(doc_id: str, request: Request) -> schemas.DocumentDetail:
    """Full content of one internal document, for the document viewer (#143).

    Read-only: viewing the cited document (from the ``document`` route's terminal
    message) never advances a run. 404 when ``doc_id`` is unknown.
    """

    with _generic_500("GET /documents"):
        with _service(request).session_factory() as session:
            row = get_document(session, doc_id)
        if row is None:
            raise HTTPException(status_code=404, detail="document not found")
        return schemas.DocumentDetail(**row)

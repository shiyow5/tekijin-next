"""Write-side persistence for the API flow (questions, recommendations, outcomes).

Kept apart from the read-only :mod:`tekijin.data.repository`. Each function takes
an active session and mutates it; the caller owns the transaction (``session_scope``).
The API persists so ``load`` (recent recommendations) and the dashboard reflect
real usage — the "使うほど育つ" loop. (The C8 person_topic_edges write stays a
separate issue.)
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.orm import Session, aliased

from tekijin.models.tables import (
    Answer,
    Employee,
    EvalRun,
    Event,
    Feedback,
    Message,
    OfflineConsult,
    Question,
    Recommendation,
)

# One recorded run stage: (stage name, started_at, ended_at, meta).
EventRow = tuple[str, dt.datetime, dt.datetime, dict[str, Any] | None]


def employee_exists(session: Session, employee_id: int) -> bool:
    """True if ``employee_id`` is a real employee (asker FK pre-check).

    Called before inserting a Question so a bad ``asker_id`` becomes a clean 404
    at the boundary instead of an ``IntegrityError`` (FK violation) mid-flush.
    """

    return (
        session.execute(select(Employee.id).where(Employee.id == employee_id)).first() is not None
    )


def persist_question(
    session: Session,
    question_id: str,
    asker_id: int,
    body: str,
    now: dt.datetime,
    *,
    session_id: str | None = None,
) -> None:
    """Insert the asked question (``created_at`` from the run's injected ``now``).

    ``session_id`` is the graph thread_id; stored so the responder inbox (#123)
    can deep-link a pending handoff back to ``/answer/{session_id}``.
    """

    session.add(
        Question(
            id=question_id,
            asker_id=asker_id,
            body=body,
            topics=[],
            status="open",
            created_at=now,
            session_id=session_id,
        )
    )


def delete_question(session: Session, question_id: str) -> None:
    """Delete a question and everything that references it, children first (#207, #286).

    ``answers`` / ``recommendations`` / ``events`` / ``feedback`` all FK
    ``questions.id`` with no ``ON DELETE CASCADE``, so a bare question delete would
    raise ``IntegrityError`` mid-flush. ``messages`` is one FK hop further out — it
    keys off ``recommendation_id``, not ``question_id`` (see :class:`Message`) — so
    its rows are found via this question's recommendation ids before those
    recommendations are deleted. Delete grandchildren, then children, then the
    question.

    ``offline_consults`` (#247) is the same shape and was added later still: the
    asker's write-up of a 直接相談 keys off the question, so a question with one
    could not be deleted at all. It goes with the question for the same reason as
    ``feedback`` — the row holds the asker's own account of the conversation.

    ``feedback`` was missed when this function was written (#207) and only surfaces
    once the asker has corrected something: a draft edit, an excluded person or a
    re-run all write a row keyed to the question, and the delete then failed with a
    500. Those rows are deleted rather than orphaned by nulling ``question_id``,
    because their ``payload`` holds the asker's own text (the draft they sent, the
    supplement they typed) — deleting the question has to take that with it.

    Scoped to one question by id. The caller confirms the question exists (via
    ``question_asker_id``) and authorizes the actor (only the asker or an admin)
    before this runs, so this just performs the removal; a non-existent id simply
    deletes nothing (idempotent).
    """

    recommendation_ids = (
        session.execute(select(Recommendation.id).where(Recommendation.question_id == question_id))
        .scalars()
        .all()
    )
    if recommendation_ids:
        session.execute(delete(Message).where(Message.recommendation_id.in_(recommendation_ids)))
    session.execute(delete(Feedback).where(Feedback.question_id == question_id))
    session.execute(delete(OfflineConsult).where(OfflineConsult.question_id == question_id))
    session.execute(delete(Answer).where(Answer.question_id == question_id))
    session.execute(delete(Recommendation).where(Recommendation.question_id == question_id))
    session.execute(delete(Event).where(Event.question_id == question_id))
    session.execute(delete(Question).where(Question.id == question_id))


def update_question_topics(session: Session, question_id: str, topics: list[str]) -> None:
    """Backfill C1's extracted topics onto the question (for the dashboard mix)."""

    session.execute(update(Question).where(Question.id == question_id).values(topics=list(topics)))


def update_question_route(session: Session, question_id: str, route: str) -> None:
    """Record the C5 route on the question (drives the dashboard 自己解決率)."""

    session.execute(update(Question).where(Question.id == question_id).values(route=route))


def reopen_question_for_handoff(session: Session, question_id: str) -> None:
    """Reopen a document-resolved question before its explicit person hand-off.

    The same Question row/session is retained; only the resolution markers and
    route change, so history never gains a duplicate question (#351).
    """

    session.execute(
        update(Question)
        .where(Question.id == question_id)
        .values(route="person", status="open", resolved_at=None, resolution_kind=None)
    )


def update_question_body(session: Session, question_id: str, body: str) -> None:
    """Persist an enriched question body (#268).

    A clarification reply (``ask`` node) or an interpretation correction (#260)
    folds extra context into the in-graph question. Without this, ``Question.body``
    stayed the ORIGINAL text while ``GET /handoff`` showed the enriched one — so the
    responder's ``/inbox`` preview and the asker's ``/history`` displayed a
    different question than was actually processed. Writing the enriched body keeps
    those SQL projections consistent with the run.
    """

    session.execute(update(Question).where(Question.id == question_id).values(body=body))


def detach_question_session(session: Session, session_id: str) -> None:
    """Unlink the question(s) of a discarded run from its graph thread (#60).

    ``/history`` links a question to its result page by ``session_id``; once the
    checkpoint is deleted that page would 404, so the row becomes history-only
    (``NULL``, like a seeded question).
    """

    session.execute(
        update(Question).where(Question.session_id == session_id).values(session_id=None)
    )


def update_question_consult_method(session: Session, question_id: str, consult_method: str) -> None:
    """Record the asker's chosen consultation method ("direct" | "chat").

    Lives on the question (not the recommendation) so it survives a decline+
    reroute, which creates a new rank-1 recommendation for the same question.
    """

    session.execute(
        update(Question).where(Question.id == question_id).values(consult_method=consult_method)
    )


def record_events(session: Session, question_id: str, rows: list[EventRow]) -> None:
    """Persist per-stage run timing (technical-spec §7) — the latency KPI source.

    One ``events`` row per graph stage with its ``started_at``/``ended_at``; the
    dashboard sums these per question for p50/p95 processing latency (#177). Batch
    ``add_all`` in the caller's transaction; a no-op on an empty list. Human-wait
    gaps (followup/send interrupts) are excluded because each run *segment* records
    only its own stages, so the sum is compute time, not wall-clock.
    """

    if not rows:
        return
    session.add_all(
        Event(
            question_id=question_id,
            stage=stage,
            started_at=started_at,
            ended_at=ended_at,
            meta=meta,
        )
        for stage, started_at, ended_at, meta in rows
    )


def mark_question_resolved(session: Session, question_id: str, resolved_at: dt.datetime) -> None:
    """Stamp the runtime resolution time on the question, first-wins (#97).

    Only sets ``resolved_at`` when it is still NULL, so a decline→reroute→accept
    or a duplicate/replayed terminal cannot move an already-recorded resolution
    time. Drives the dashboard's average resolution time from live traffic.
    """

    session.execute(
        update(Question)
        .where(Question.id == question_id, Question.resolved_at.is_(None))
        .values(resolved_at=resolved_at)
    )


def mark_self_resolved(session: Session, question_id: str, now: dt.datetime) -> None:
    """Record that the asker solved the question WITHOUT a live hand-off (#159).

    Sets ``resolution_kind="self"`` and stamps ``resolved_at``, guarded on BOTH
    ``resolution_kind IS NULL`` AND ``resolved_at IS NULL`` so it is genuinely
    first-wins across both resolution paths: a question a responder already resolved
    (``mark_question_resolved`` stamps ``resolved_at`` on an accept, but not
    ``resolution_kind``) is NOT re-labelled self. Idempotent. Drives the dashboard's
    self-resolution rate as a genuine "人を介さず満足" signal, not a route-only proxy.
    """

    session.execute(
        update(Question)
        .where(
            Question.id == question_id,
            Question.resolution_kind.is_(None),
            Question.resolved_at.is_(None),
        )
        .values(resolution_kind="self", resolved_at=now)
    )


def insert_eval_run(session: Session, metrics: dict[str, Any]) -> int:
    """Persist an offline-evaluation snapshot; returns its row id.

    Stores only the aggregate metrics the dashboard shows — never per-query data.
    """

    row = EvalRun(
        top1_accuracy=metrics.get("top1_accuracy"),
        recall_at_3=metrics.get("recall_at_3"),
        mrr=metrics.get("mrr"),
        route_accuracy=metrics.get("route_accuracy"),
        n_ranked=metrics.get("n_ranked"),
        n_routed=metrics.get("n_routed"),
    )
    session.add(row)
    session.flush()
    session.refresh(row)
    return row.id


def insert_shown_recommendations(
    session: Session,
    question_id: str,
    recommendations: list[dict[str, Any]],
) -> list[int]:
    """Persist EVERY shown recommendation (rank 1..N) and return their ids in order.

    The list is already ranked (top first). ``rank`` is ``1``-based by position.
    The returned ids preserve that order, so ``ids[0]`` is the primary (top /
    handed-off) recommendation whose outcome the responder later records; the rest
    are stored as ``outcome=NULL`` "shown" rows for the dashboard / audit trail.

    ``created_at`` is intentionally NOT set here: the column's ``server_default``
    (``now()``) stamps the actual INSERT time, which is the recommendation's real
    generation moment. This is deliberately separate from the injected ``now`` the
    agent uses for scoring — a reroute inserts a fresh row whose ``created_at``
    reflects when it was produced, keeping the scorer's 7-day ``load`` window
    accurate (codex#4).
    """

    ids: list[int] = []
    for position, rec in enumerate(recommendations, start=1):
        row = Recommendation(
            question_id=question_id,
            employee_id=rec["person_id"],
            rank=position,
            score=rec.get("score"),
            reasons={"reasons": rec.get("reasons") or []},  # JSONB column typed as dict
            outcome=None,
        )
        session.add(row)
        session.flush()
        session.refresh(row)
        ids.append(row.id)
    return ids


def reorder_recommendation_ranks(session: Session, ordered_ids: list[int]) -> None:
    """Set ``rank`` (1-based) on these ``Recommendation`` rows to match ``ordered_ids``.

    Used when the asker reselects a different shown candidate as the hand-off
    target (#200/#A1): the durable checkpoint's ``recommendations`` order is the
    source of truth for the *graph*, but ``GET /inbox`` reads the DB ``rank``
    column directly (SQL-only, never consults the checkpoint) — so a reselect
    must also sync this projection, or the newly-selected responder's inbox will
    not show the pending handoff. ``ordered_ids`` are the specific row ids from
    the current checkpoint's ``recommendation_ids`` (not a question-wide query),
    so a decline+reroute's earlier rank==1 rows for the same question are never
    touched by this.
    """

    for position, rec_id in enumerate(ordered_ids, start=1):
        session.execute(
            update(Recommendation).where(Recommendation.id == rec_id).values(rank=position)
        )


def latest_primary_recommendation(session: Session, question_id: str) -> int | None:
    """The id of the most recent rank-1 recommendation for a question, if any.

    Durable fallback for outcome recording when the primary id was not written
    back into the checkpoint (e.g. a disconnect before ``update_state`` ran). On a
    decline+reroute there are several rank-1 rows; the newest (highest id) is the
    one currently being answered.
    """

    row = session.execute(
        select(Recommendation.id)
        .where(Recommendation.question_id == question_id, Recommendation.rank == 1)
        .order_by(Recommendation.id.desc())
        .limit(1)
    ).first()
    return row[0] if row else None


def latest_recommendation_for_person(
    session: Session, question_id: str, employee_id: int
) -> int | None:
    """The newest recommendation row for a (question, employee) pair, if any.

    Used to re-bind a hand-off whose checkpointed ``primary_recommendation_id``
    went stale: a decline+reroute inserts a fresh row set, and a client disconnect
    between that INSERT and the checkpoint write-back leaves the id pointing at the
    DECLINED row while the graph already shows the next candidate (#94-3). The
    person currently shown is the authority; the newest row for them is the one to
    record the outcome on (a reroute can only add rows, never reorder existing ones).
    """

    row = session.execute(
        select(Recommendation.id)
        .where(
            Recommendation.question_id == question_id,
            Recommendation.employee_id == employee_id,
        )
        .order_by(Recommendation.id.desc())
        .limit(1)
    ).first()
    return row[0] if row else None


def recommendation_employee_id(session: Session, recommendation_id: int) -> int | None:
    """The employee a recommendation points at (``None`` if the id is unknown).

    The authoritative responder for an accepted hand-off: read from the DB row
    rather than the graph state so the ``answers`` row (#274) attributes the
    answer to a real employee id regardless of the external ``E###`` wire form.
    """

    return session.execute(
        select(Recommendation.employee_id).where(Recommendation.id == recommendation_id)
    ).scalar_one_or_none()


def create_answer(
    session: Session,
    *,
    question_id: str,
    responder_id: int,
    body: str,
    topic: str | None,
    embedding: Sequence[float] | None,
    created_at: dt.datetime,
) -> str:
    """Persist a runtime answer and return its id (#274, accumulation loop).

    The responder's answer text becomes an ``answers`` row so it (a) fuels reuse
    — the retriever searches ``answers`` by dense embedding and BM25, so a future
    self-answer / prior_answer path can surface it — (b) lights the asker's "回答が
    届きました" history (``recent_questions_for_asker`` joins ``answers``), and (c)
    feeds the accumulation dashboard. ``embedding`` may be ``None`` (the embedder
    was unavailable): the row is still BM25-reusable and history/metrics still see
    it; only the dense channel skips it until it is embedded.
    """

    answer_id = f"ans_{uuid.uuid4().hex}"
    session.add(
        Answer(
            id=answer_id,
            question_id=question_id,
            responder_id=responder_id,
            body=body,
            topic=topic,
            embedding=list(embedding) if embedding is not None else None,
            created_at=created_at,
            reuse_count=0,
        )
    )
    return answer_id


def shown_recommendation_ids(session: Session, question_id: str) -> list[int]:
    """Existing shown recommendation ids for a question, in display order.

    Used by the document-fallback continuation to recover from a disconnect after
    SQL persistence but before the ids reached the graph checkpoint (#351).
    """

    return list(
        session.execute(
            select(Recommendation.id)
            .where(Recommendation.question_id == question_id)
            .order_by(Recommendation.rank, Recommendation.id)
        )
        .scalars()
        .all()
    )


def recommendation_outcome(session: Session, recommendation_id: int) -> str | None:
    """The currently recorded outcome for a recommendation (``None`` if unset)."""

    return session.execute(
        select(Recommendation.outcome).where(Recommendation.id == recommendation_id)
    ).scalar_one_or_none()


def set_recommendation_outcome(
    session: Session,
    recommendation_id: int,
    outcome: str,
    *,
    referred_to: int | None = None,
) -> None:
    """Record the responder's accept/decline on a recommendation, once.

    The update is guarded on ``outcome IS NULL`` so a duplicate submission — e.g. a
    lost acknowledgement retried after a process restart cleared the in-memory
    dedup guard — cannot overwrite an already-recorded outcome (idempotent write).

    ``referred_to`` (#518): on a ``'referred'`` outcome, the employee the responder
    named as more suitable, written in the SAME guarded update so the outcome and its
    target land atomically (or not at all on a duplicate).
    """

    values: dict[str, object] = {"outcome": outcome}
    if referred_to is not None:
        values["referred_to_employee_id"] = referred_to
    session.execute(
        update(Recommendation)
        .where(Recommendation.id == recommendation_id, Recommendation.outcome.is_(None))
        .values(**values)
    )


def ack_decline_notifications(session: Session, asker_id: int, ids: list[int]) -> int:
    """Mark these decline notifications seen, scoped to ``asker_id``'s own questions (#E7).

    Scoping by ``asker_id`` (via a subquery on the owning question) means one
    asker can never acknowledge — and so hide — another asker's notification by
    guessing an id. Returns the number of rows actually updated.
    """

    if not ids:
        return 0
    result = session.execute(
        update(Recommendation)
        .where(
            Recommendation.id.in_(ids),
            Recommendation.rank == 1,
            Recommendation.outcome == "declined",
            Recommendation.declined_seen_at.is_(None),
            Recommendation.question_id.in_(
                select(Question.id).where(Question.asker_id == asker_id)
            ),
        )
        .values(declined_seen_at=func.now())
    )
    # ``Session.execute`` is typed as returning ``Result``; a DML statement always
    # yields a ``CursorResult``, which is where ``rowcount`` lives.
    return cast("CursorResult[Any]", result).rowcount or 0


def ack_accepted_notifications(session: Session, asker_id: int, ids: list[int]) -> int:
    """Mark these accepted notifications seen, scoped to ``asker_id``'s own questions (#509).

    Mirrors ``ack_decline_notifications``; see its docstring for the scoping
    rationale. Returns the number of rows actually updated.
    """

    if not ids:
        return 0
    result = session.execute(
        update(Recommendation)
        .where(
            Recommendation.id.in_(ids),
            Recommendation.outcome == "accepted",
            Recommendation.accepted_seen_at.is_(None),
            Recommendation.question_id.in_(
                select(Question.id).where(Question.asker_id == asker_id)
            ),
        )
        .values(accepted_seen_at=func.now())
    )
    return cast("CursorResult[Any]", result).rowcount or 0


def ack_request_notifications(session: Session, responder_id: int, ids: list[int]) -> int:
    """Mark these incoming-request notifications seen, scoped to ``responder_id`` (#509).

    The read side deduplicates anomalous multiple pending rank-1 rows for one
    question. Resolve the selected ids to their eligible question ids first,
    then acknowledge every matching pending row so an older duplicate cannot
    reappear on the next poll. Returns the number of rows actually updated
    (which can exceed ``len(ids)`` only for that duplicate-row anomaly).
    """

    if not ids:
        return 0
    selected = aliased(Recommendation)
    selected_question_ids = select(selected.question_id).where(
        selected.id.in_(ids),
        selected.rank == 1,
        selected.outcome.is_(None),
        selected.employee_id == responder_id,
        selected.question_id.in_(select(Question.id).where(Question.session_id.is_not(None))),
    )
    result = session.execute(
        update(Recommendation)
        .where(
            Recommendation.question_id.in_(selected_question_ids),
            Recommendation.rank == 1,
            Recommendation.outcome.is_(None),
            Recommendation.request_seen_at.is_(None),
            Recommendation.employee_id == responder_id,
        )
        .values(request_seen_at=func.now())
    )
    return cast("CursorResult[Any]", result).rowcount or 0


def record_offline_consult(
    session: Session,
    *,
    question_id: str,
    responder_id: int,
    asker_id: int | None,
    topics: Sequence[str],
    asked: str | None,
    answer_body: str,
    resolution: str,
) -> int:
    """Insert one 直接相談 retrospective and return its id (#247).

    Written by the ASKER about the responder, so ``asker_id`` must come from the
    authenticated principal (the route enforces that) — this row becomes expertise
    evidence for ``responder_id``, and an unattributable one would be a way to
    fabricate someone's standing.
    """

    row = OfflineConsult(
        question_id=question_id,
        responder_id=responder_id,
        asker_id=asker_id,
        topics=list(topics),
        asked=asked,
        answer_body=answer_body,
        resolution=resolution,
    )
    session.add(row)
    session.flush()  # so the caller can ack with a real id
    return row.id

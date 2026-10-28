"""Integration tests for the API against live PostgreSQL + pgvector (pgserver/CI).

The agent runs on the deterministic stub LLM nodes and (usually) injected fake C4/C6
so the SSE flow is reproducible; C6/dashboard read the real seeded DB. No network,
no model download. ``now`` is injected for determinism.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import threading
import time
from inspect import signature

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from tekijin.agent.stubs import (
    KeywordIntentModel,
    RuleSufficiencyModel,
    TemplateDraftModel,
    TemplateQuestionStructurer,
)
from tekijin.api.routes import knowledge as knowledge_route
from tekijin.api.service import (
    SESSION_TTL_SECONDS,
    AgentService,
    HandoffNotFound,
    SessionConflict,
    SessionInvalid,
    _SessionCtx,
)
from tekijin.app import create_app
from tekijin.auth.principal import Principal
from tekijin.auth.tokens import create_access_token
from tekijin.config import get_settings
from tekijin.data.dashboard import dashboard_summary
from tekijin.data.db import get_sessionmaker, session_scope
from tekijin.data.slack_channel_links import get_channel_link
from tekijin.data.slack_links import upsert_slack_link
from tekijin.models.tables import Answer, Event, Feedback, Message, Question, Recommendation

NOW = dt.datetime(2026, 9, 15, 12, 0, 0)
GOOD_Q = "現行のVPN機器で3拠点の拠点間接続について相談したいです"
# Vague, no extractable topic -> C1 confidence stays sub-threshold -> C2 still asks
# a clarification. Under the #113 safety valve a topic-only question routes straight
# through, so this is what now exercises the ask/followup interrupt path.
VAGUE_Q = "相談したいことがあります"


class _FakeRetriever:
    def __init__(self, *, people=(), **conf) -> None:
        self._payload = {
            "past_answers": list(conf.get("past_answers", [])),
            "documents": list(conf.get("documents", [])),
            "candidate_people": list(people),
            "answer_confidence": conf.get("answer_confidence", 0.0),
            "document_confidence": conf.get("document_confidence", 0.0),
            "people_confidence": conf.get("people_confidence", 0.0),
        }

    def search(self, query: str, *, query_vector=None) -> dict:
        return self._payload


class _FakeScorer:
    def __init__(self, recs: list[dict]) -> None:
        self._recs = recs

    def rank(self, topics, candidates, asker_id, now, *, top_k=3, question_similarity=None) -> dict:
        # Honour the decline loop: only recommend candidates still in the pool.
        recs = [r for r in self._recs if r["person_id"] in candidates]
        return {"recommendations": recs[:top_k]}


def _recs(*ids: int) -> list[dict]:
    return [
        {
            "person_id": i,
            "name": f"社員{i}",
            "dept": "営業部",
            "score": 0.9 - 0.01 * i,
            "confidence": "中",
            "reasons": [{"type": "self", "detail": "自己申告スキル"}],
        }
        for i in ids
    ]


# Every endpoint now requires auth (#241). These tests act as ADMIN by default so
# they can drive any asker_id/responder_id (admin may impersonate anyone) — the
# same freedom they had before auth. Non-admin restriction + login are covered by
# dedicated tests below and in test_auth_integration.py.
def _admin_headers() -> dict[str, str]:
    settings = get_settings()
    token = create_access_token(
        Principal(employee_id=None, name="管理者", dept=None, is_admin=True),
        secret=settings.auth_secret,
        ttl_hours=1,
    )
    return {"Authorization": f"Bearer {token}"}


def _user_headers(
    employee_id: int, *, name: str = "利用者", dept: str | None = None
) -> dict[str, str]:
    settings = get_settings()
    token = create_access_token(
        Principal(employee_id=employee_id, name=name, dept=dept, is_admin=False),
        secret=settings.auth_secret,
        ttl_hours=1,
    )
    return {"Authorization": f"Bearer {token}"}


def _app_client(service: AgentService) -> TestClient:
    """A TestClient over ``create_app`` pre-authenticated as admin (default)."""

    client = TestClient(create_app(agent_service=service))
    client.headers.update(_admin_headers())
    return client


def _client(
    engine,
    embedder,
    *,
    retriever=None,
    scorer=None,
    checkpointer=None,
    answerability_model=None,
    answerability_threshold=40,
    self_answer_model=None,
    knowledge_answer_min_similarity=None,
    question_structurer=None,
) -> TestClient:
    return _app_client(
        _service(
            engine,
            embedder,
            retriever=retriever,
            scorer=scorer,
            checkpointer=checkpointer,
            answerability_model=answerability_model,
            answerability_threshold=answerability_threshold,
            self_answer_model=self_answer_model,
            knowledge_answer_min_similarity=knowledge_answer_min_similarity,
            question_structurer=question_structurer,
        )
    )


def _service(
    engine,
    embedder,
    *,
    retriever=None,
    scorer=None,
    checkpointer=None,
    answerability_model=None,
    answerability_threshold=40,
    self_answer_model=None,
    knowledge_answer_min_similarity=None,
    question_structurer=None,
) -> AgentService:
    """Same wiring as :func:`_client`, but hands back the service itself.

    Tests that need to poke at durable state (rather than only the HTTP surface)
    build the service here and wrap it with :func:`_app_client` themselves.
    """

    return AgentService(
        session_factory=get_sessionmaker(engine),
        checkpointer=checkpointer or MemorySaver(),
        embedder=embedder,
        intent_model=KeywordIntentModel(),
        sufficiency_model=RuleSufficiencyModel(),
        draft_model=TemplateDraftModel(),
        # #475: always wire the on-demand question structurer (stub) so the
        # /handoff/structure endpoint is exercised; a test can inject its own.
        question_structurer=question_structurer or TemplateQuestionStructurer(),
        answerability_model=answerability_model,
        answerability_threshold=answerability_threshold,
        self_answer_model=self_answer_model,
        knowledge_answer_min_similarity=knowledge_answer_min_similarity,
        retriever=retriever,
        scorer=scorer,
        now_factory=lambda: NOW,
    )


class _FixedSelfAnswer:
    """#291 self-answer composer stand-in for the SSE test."""

    def __init__(self, *, grounded: bool, answer: str = "", cites=()) -> None:
        self._grounded = grounded
        self._answer = answer
        self._cites = list(cites)

    def compose(self, question, evidence):
        from tekijin.agent.protocols import SelfAnswerResult

        if not self._grounded:
            return SelfAnswerResult(answer="", cited_source_ids=[], grounded=False)
        return SelfAnswerResult(answer=self._answer, cited_source_ids=self._cites, grounded=True)


class _FixedAnswerability:
    """#70 critic stand-in with a fixed confidence, for the SSE/persist tests."""

    def __init__(self, confidence: int) -> None:
        self._confidence = confidence

    def assess(self, question, candidate_evidence):
        from tekijin.agent.protocols import AnswerabilityResult

        return AnswerabilityResult(confidence=self._confidence, reason="判定理由")


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    event: str | None = None
    data: list[str] = []
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].strip())
        elif line == "" and event is not None:
            out.append((event, json.loads("".join(data)) if data else {}))
            event, data = None, []
    if event is not None:
        out.append((event, json.loads("".join(data)) if data else {}))
    return out


def _events(client: TestClient, session_id: str) -> list[tuple[str, dict]]:
    # sse-starlette keeps a module-global graceful-shutdown Event bound to the
    # loop of the first request; TestClient uses a fresh loop per request, so
    # reset it before each SSE call (a test-only artifact; real uvicorn has one
    # loop). It is lazily recreated on the current loop.
    from sse_starlette.sse import AppStatus

    AppStatus.should_exit_event = None
    resp = client.get(f"/events/{session_id}")
    assert resp.status_code == 200
    return _parse_sse(resp.text)


# Slack's request handlers finish the slow part AFTER responding, in daemon
# threads (its interactivity budget is ~3s, a graph advance is not). Those
# threads write to the DB, so a test that returns while one is still running
# races the cleanup below: it deletes events, then questions, and the thread
# inserts a fresh event in between -> events_question_id_fkey (#460).
_ASYNC_SLACK_THREADS = (
    "slack-interactivity-advance",
    "slack-pending-handoff-setup",
    "slack-handoff-channel-setup",
)


def _join_async_slack_work(timeout: float = 15.0) -> None:
    """Block until no Slack background thread is still touching the DB.

    Waiting on ``ctx.pending is None`` is NOT enough: ``_dispatch_stream`` clears
    that when it STARTS the queued Command, while the event rows are written at
    the end of the run.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [
            t for t in threading.enumerate() if t.name in _ASYNC_SLACK_THREADS and t.is_alive()
        ]
        if not alive:
            return
        time.sleep(0.02)
    raise AssertionError(
        f"Slack background threads still running after {timeout}s: "
        f"{[t.name for t in threading.enumerate() if t.name in _ASYNC_SLACK_THREADS]}"
    )


def test_join_async_slack_work_waits_for_a_running_thread() -> None:
    """The guard for #460: if this join is ever dropped, the FK error comes back
    as an intermittent teardown failure that reruns hide. Test the mechanism
    directly rather than by racing the graph, so it is fast and deterministic.
    """

    released = threading.Event()
    finished: list[float] = []

    def _work() -> None:
        released.wait(timeout=5.0)
        finished.append(time.monotonic())

    worker = threading.Thread(target=_work, name="slack-interactivity-advance")
    worker.start()
    try:
        threading.Timer(0.2, released.set).start()
        _join_async_slack_work(timeout=5.0)
        returned = time.monotonic()
        assert finished, "join returned while the thread was still running"
        assert returned >= finished[0]
    finally:
        released.set()
        worker.join(timeout=5.0)


def test_join_async_slack_work_raises_rather_than_deleting_under_a_live_writer() -> None:
    # Timing out must FAIL loudly. Returning quietly would drop straight into the
    # deletes with a writer still live — the exact condition being guarded against.
    released = threading.Event()
    worker = threading.Thread(
        target=lambda: released.wait(timeout=5.0), name="slack-pending-handoff-setup"
    )
    worker.start()
    try:
        with pytest.raises(AssertionError, match="still running"):
            _join_async_slack_work(timeout=0.3)
    finally:
        released.set()
        worker.join(timeout=5.0)


@pytest.fixture(autouse=True)
def _cleanup_api_rows(engine):
    # The API COMMITS questions/recommendations (id prefix "api_"). Remove them
    # after each test so the committed seed / other tests stay isolated.
    from sqlalchemy import text

    yield
    # BEFORE any delete: see _join_async_slack_work. Ordering the deletes by FK
    # is not enough on its own when a writer is still running.
    _join_async_slack_work()
    session = get_sessionmaker(engine)()
    try:
        # messages FK-reference recommendations (no ON DELETE CASCADE), so chat
        # rows from #224 tests must go first.
        session.execute(
            text(
                r"DELETE FROM messages WHERE recommendation_id IN "
                r"(SELECT id FROM recommendations WHERE question_id LIKE 'api\_%' ESCAPE '\')"
            )
        )
        # feedback (#237) is runtime-only (seed writes none) and FKs questions, so
        # clear it first — before the questions it may reference are deleted.
        session.execute(text("DELETE FROM feedback"))
        # offline_consults (#247) is likewise runtime-only and FKs questions +
        # employees, so it must be cleared before the questions it references.
        session.execute(text("DELETE FROM offline_consults"))
        session.execute(
            text(r"DELETE FROM recommendations WHERE question_id LIKE 'api\_%' ESCAPE '\'")
        )
        # events reference questions (FK) and are now written at runtime (#177), so
        # they must be removed before the questions they point at.
        session.execute(text(r"DELETE FROM events WHERE question_id LIKE 'api\_%' ESCAPE '\'"))
        # answers also FK questions (the #207 delete tests insert api_ answers).
        session.execute(text(r"DELETE FROM answers WHERE question_id LIKE 'api\_%' ESCAPE '\'"))
        session.execute(text(r"DELETE FROM questions WHERE id LIKE 'api\_%' ESCAPE '\'"))
        session.commit()
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# happy path: understood -> route -> recommend -> draft -> (resume) -> done
# --------------------------------------------------------------------------- #
def test_happy_path_ask_events_answer(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    ack = client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "s1"})
    assert ack.status_code == 200 and ack.json() == {"session_id": "s1", "status": "accepted"}

    evs = _events(client, "s1")
    names = [e for e, _ in evs]
    assert names == ["understood", "route", "recommend", "draft"]
    assert evs[0][1]["topics"] == ["ネットワーク・VPN"]
    assert evs[1][1]["route"] == "person"
    # person_id crosses the boundary in the external "E###" form (codex#7).
    assert [r["person_id"] for r in evs[2][1]["recommendations"]] == ["E001", "E002", "E003"]
    assert "社員1さん" in evs[3][1]["draft"]

    # Resume with the responder's acceptance -> done.
    ans = client.post("/answer", json={"session_id": "s1", "outcome": "accepted"})
    assert ans.status_code == 200 and ans.json()["status"] == "resumed"
    done = _events(client, "s1")
    assert len(done) == 1 and done[0][0] == "done"
    assert done[0][1]["status"] == "sent"
    assert "取り次ぎました" in done[0][1]["answer"]
    # #177: the done event carries the segment's processing latency (int ms).
    assert isinstance(done[0][1]["latency_ms"], int)


# --------------------------------------------------------------------------- #
# #177: per-stage run events (latency KPI source) + dashboard percentiles
# --------------------------------------------------------------------------- #
def _events_for(engine, question_id: str) -> list[Event]:
    check = get_sessionmaker(engine)()
    try:
        return check.query(Event).filter(Event.question_id == question_id).all()
    finally:
        check.close()


def test_run_records_stage_events(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "ev1"})
    _events(client, "ev1")  # run to the send interrupt

    qid = _latest_question(engine).id
    rows = _events_for(engine, qid)
    stages = {r.stage for r in rows}
    # Every compute stage of the person-route segment is recorded (not just the
    # ones that surface an SSE event), so the latency sum is real processing time.
    assert {"c1_intent", "c3_embed", "c4_retrieve", "c5_route", "c6_score", "c7_draft"} <= stages
    # No interrupt pseudo-node is recorded, and timing columns are populated.
    assert "__interrupt__" not in stages
    for r in rows:
        assert r.started_at is not None and r.ended_at is not None


def test_dashboard_exposes_processing_latency_from_events(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(engine, fake_embedder)
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "lat1"})
    _events(client, "lat1")

    body = client.get("/dashboard").json()
    lat = body["processing_latency"]
    assert lat["sample_size"] >= 1
    # p50/p95 are present and non-negative once at least one run is recorded.
    assert lat["p50_ms"] is not None and lat["p50_ms"] >= 0
    assert lat["p95_ms"] is not None and lat["p95_ms"] >= 0


def test_dashboard_top_responders_is_a_bounded_query_param(
    seed_counts, engine, fake_embedder
) -> None:
    # #76: the load list size was hardcoded at 5. It is now a query param, bounded
    # at 1..50 — the dashboard is aggregate-only by design (product-spec §241-251),
    # so an unbounded limit would turn it into a per-employee roster.
    client = _client(engine, fake_embedder)

    default = client.get("/dashboard").json()["answers_per_responder"]
    assert len(default) == 5  # unchanged default (the seed has 40 distinct responders)

    wider = client.get("/dashboard", params={"top_responders": 12}).json()
    assert len(wider["answers_per_responder"]) > len(default)  # the seed has >5 responders
    assert len(wider["answers_per_responder"]) <= 12
    # The list stays ordered by load, and widening only appends.
    assert [r["employee_id"] for r in wider["answers_per_responder"]][: len(default)] == [
        r["employee_id"] for r in default
    ]

    assert client.get("/dashboard", params={"top_responders": 0}).status_code == 422
    assert client.get("/dashboard", params={"top_responders": 51}).status_code == 422
    assert client.get("/dashboard", params={"top_responders": "all"}).status_code == 422

    # The load KPI must NOT move with the limit: its denominator is the total answer
    # count, not the sum of the truncated list. (If it were the latter, widening the
    # list would silently change a headline number.)
    assert wider["top_responder_share"] == pytest.approx(
        client.get("/dashboard").json()["top_responder_share"]
    )


def test_run_records_nonzero_monotonic_durations(seed_counts, engine, fake_embedder) -> None:
    # With an advancing clock (not the fixed test clock), each stage has a real,
    # positive, non-overlapping duration and the terminal reports a non-zero
    # latency_ms — exercising the actual prev-threading in _run (#177 review).
    import datetime as dt
    import itertools

    base = dt.datetime(2026, 1, 1, 0, 0, 0)
    ticks = itertools.count()

    svc = _svc(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer([]),  # empty -> reaches the no_candidate terminal (a done-ish end)
        now_factory=lambda: base + dt.timedelta(seconds=next(ticks)),
    )
    svc.start_question("dur1", 10, GOOD_Q)
    out = list(svc.stream_events("dur1"))

    # The terminal message carries a strictly-positive processing latency.
    msg = next(e for e in out if e.event == "message")
    assert json.loads(msg.data)["latency_ms"] > 0

    qid = _latest_question(engine).id
    rows = sorted(_events_for(engine, qid), key=lambda r: r.started_at)
    assert rows
    for r in rows:
        assert r.ended_at > r.started_at  # positive duration
    for a, b in zip(rows, rows[1:], strict=False):
        assert b.started_at >= a.ended_at  # non-overlapping, monotonic


# --------------------------------------------------------------------------- #
# clarification: followup -> reply resume -> continues
# --------------------------------------------------------------------------- #
def test_followup_then_reply_resumes(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    # Topic-only question -> missing slots -> C2 asks a followup.
    client.post("/ask", json={"asker_id": 10, "question": VAGUE_Q, "session_id": "s2"})
    first = _events(client, "s2")
    assert [e for e, _ in first] == ["understood", "followup"]
    assert "教えてください" in first[1][1]["question"]

    # Reply with the missing info -> resumes and runs through to the draft.
    client.post("/answer", json={"session_id": "s2", "reply": "現行はVPN機器で3拠点です"})
    cont = [e for e, _ in _events(client, "s2")]
    assert cont == ["understood", "route", "recommend", "draft"]


# --------------------------------------------------------------------------- #
# decline -> reroute to next candidate -> accept -> done
# --------------------------------------------------------------------------- #
def test_decline_reroutes_then_accept(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "s3"})
    first = _events(client, "s3")
    assert first[2][1]["recommendations"][0]["person_id"] == "E001"  # drafted for 1

    # Decline -> reroute -> re-scored excluding 1 -> next candidate 2.
    client.post("/answer", json={"session_id": "s3", "outcome": "declined"})
    second = _events(client, "s3")
    assert [e for e, _ in second] == ["recommend", "draft"]
    assert second[0][1]["recommendations"][0]["person_id"] == "E002"

    client.post("/answer", json={"session_id": "s3", "outcome": "accepted"})
    done = _events(client, "s3")
    assert done[0][0] == "done"

    # The decline never stamped resolved_at; the final accept does, once (#97).
    check = get_sessionmaker(engine)()
    try:
        q = check.query(Question).filter(Question.session_id == "s3").first()
        assert q is not None and q.resolved_at is not None
    finally:
        check.close()


# --------------------------------------------------------------------------- #
# POST /handoff/refer : responder names a specific person to refer to (#518)
# --------------------------------------------------------------------------- #
def test_refer_seats_the_named_person_and_records_the_referral(
    seed_counts, engine, fake_embedder
) -> None:
    # Referring to employee 5 (a real employee NOT among the shown [1,2,3]) seats
    # them at rank 1, records outcome=referred + referred_to on the referrer's row,
    # and puts 5 in 5's inbox.
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3, 5)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "ref1"})
    first = _events(client, "ref1")
    assert first[2][1]["recommendations"][0]["person_id"] == "E001"  # drafted for 1

    resp = client.post("/handoff/refer", json={"session_id": "ref1", "person_id": "E005"})
    assert resp.status_code == 200
    second = _events(client, "ref1")
    assert [e for e, _ in second] == ["recommend", "draft"]
    assert second[0][1]["recommendations"][0]["person_id"] == "E005"  # named person is rank 1

    # The referrer's row records the referral (distinct from a plain decline).
    qid = _question_id_for_session(engine, "ref1")
    check = get_sessionmaker(engine)()
    try:
        from sqlalchemy import select as _select

        row = check.execute(
            _select(Recommendation.outcome, Recommendation.referred_to_employee_id).where(
                Recommendation.question_id == qid, Recommendation.employee_id == 1
            )
        ).first()
        assert row is not None and row[0] == "referred" and row[1] == 5
    finally:
        check.close()

    # 5 can now see the hand-off in their inbox.
    inbox = client.get("/inbox", params={"responder_id": "E005"}).json()
    assert any(item["question_id"] == qid for item in inbox["items"])


def test_refer_rejects_asker_and_current_responder(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "ref2"})
    _events(client, "ref2")
    # The asker (10) cannot be referred to (can't answer their own question).
    assert (
        client.post("/handoff/refer", json={"session_id": "ref2", "person_id": "E010"}).status_code
        == 422
    )
    # The current responder (1) is a no-op self-referral.
    assert (
        client.post("/handoff/refer", json={"session_id": "ref2", "person_id": "E001"}).status_code
        == 422
    )


def test_refer_rejects_an_already_declined_person(seed_counts, engine, fake_embedder) -> None:
    # #518 review (MEDIUM): the c6 pin would silently drop an already-declined person,
    # so referring to them is rejected up front rather than recorded as a no-op.
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "ref5"})
    _events(client, "ref5")
    # Decline 1 -> reroute to 2; now 1 is in declined_ids.
    client.post("/answer", json={"session_id": "ref5", "outcome": "declined"})
    _events(client, "ref5")
    resp = client.post("/handoff/refer", json={"session_id": "ref5", "person_id": "E001"})
    assert resp.status_code == 422


def test_refer_is_responder_only(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "ref3"})
    _events(client, "ref3")
    # The asker (10) is a participant but NOT the responder -> 403.
    resp = client.post(
        "/handoff/refer",
        json={"session_id": "ref3", "person_id": "E002"},
        headers=_user_headers(10),
    )
    assert resp.status_code == 403


def test_refer_404_when_no_handoff(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    resp = client.post("/handoff/refer", json={"session_id": "nope", "person_id": "E002"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# dispatch guards: 409 on busy/paused, 422 on wrong resume kind
# --------------------------------------------------------------------------- #
def test_second_ask_while_queued_conflicts(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    assert (
        client.post(
            "/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "c1"}
        ).status_code
        == 200
    )
    # A run is already queued (not yet streamed) -> 409, does not overwrite.
    assert (
        client.post(
            "/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "c1"}
        ).status_code
        == 409
    )


def test_ask_while_paused_conflicts(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "c2"})
    _events(client, "c2")  # runs to the send interrupt (paused)
    # Session is awaiting a resume -> a new /ask is rejected.
    assert (
        client.post(
            "/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "c2"}
        ).status_code
        == 409
    )


def test_answer_wrong_kind_is_422(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    # Paused at send (expects an outcome) — sending a 'reply' is the wrong kind.
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "c3"})
    _events(client, "c3")
    assert client.post("/answer", json={"session_id": "c3", "reply": "x"}).status_code == 422

    # Paused at ask (expects a reply) — sending an 'outcome' is the wrong kind.
    client.post("/ask", json={"asker_id": 10, "question": VAGUE_Q, "session_id": "c4"})
    _events(client, "c4")
    assert (
        client.post("/answer", json={"session_id": "c4", "outcome": "accepted"}).status_code == 422
    )


def test_answer_without_interrupt_conflicts(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    # Never asked / not paused -> 409.
    assert (
        client.post("/answer", json={"session_id": "never", "outcome": "accepted"}).status_code
        == 409
    )


# --------------------------------------------------------------------------- #
# reconnect: /events again on a paused session re-emits the pending interrupt
# --------------------------------------------------------------------------- #
def test_reconnect_resends_pending_interrupt(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    # send interrupt -> reconnect re-sends the candidates AND the draft, so a
    # client that reconnects (or one that reads after another consumer drained the
    # live segment) can fully reconstruct the hand-off (#38 re-review) — now
    # preceded by the progress the run already produced (#512).
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "r1"})
    _events(client, "r1")
    again = _events(client, "r1")  # no /answer: reconnect
    assert [e for e, _ in again] == ["understood", "route", "recommend", "draft"]
    recommend = next(data for name, data in again if name == "recommend")
    assert recommend["recommendations"][0]["person_id"] == "E001"

    # ask interrupt -> reconnect re-sends the followup, after the one step that
    # HAS run (c1). c5 has not run yet, so there is no route to claim.
    client.post("/ask", json={"asker_id": 10, "question": VAGUE_Q, "session_id": "r2"})
    _events(client, "r2")
    again2 = _events(client, "r2")
    assert [e for e, _ in again2] == ["understood", "followup"]


def test_reconnect_replays_the_progress_a_client_already_missed(
    seed_counts, engine, fake_embedder
) -> None:
    """#512: revisiting a session must show the same thinking steps as the live run.

    The steps the UI renders are built from ``understood`` / ``route`` /
    ``recommend`` / ``draft``. Before this, a reconnect re-sent only the pending
    interrupt (or only the stored terminal), so returning to a session — from the
    history list, from the result screen, or by reloading — produced a screen with
    no progress on it at all, which reads as a different screen entirely.

    The replayed payloads must equal the live ones: a reconstruction that merely
    resembles the original is how the two screens drift apart again.
    """
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "prog"})
    live = _events(client, "prog")
    assert [e for e, _ in live] == ["understood", "route", "recommend", "draft"]

    client.post("/answer", json={"session_id": "prog", "outcome": "accepted"})
    _events(client, "prog")  # drains the live `done`

    again = _events(client, "prog")  # reconnect AFTER completion
    assert [e for e, _ in again] == ["understood", "route", "recommend", "draft", "done"]
    replayed = {name: data for name, data in again}
    for name, data in live:
        assert replayed[name] == data, name

    # Read-only: replaying progress must not re-run the graph or re-insert rows.
    assert len(_recs_for(engine, _latest_question(engine).id)) == 1


def test_progress_replay_does_not_invent_steps_the_run_never_reached(
    seed_counts, engine, fake_embedder
) -> None:
    """A terminal reached before C5/C6/C7 replays only the steps that ran.

    ``reset`` seeds `route`/`recommendations`/`draft` with defaults BEFORE C1, so a
    naive "emit whatever is in the state" replay would announce 「経路を判断しました:
    人に取り次ぐ」 on a question that was never routed.
    """
    # The default keyword intent stub marks 天気 out of scope -> off_topic terminal.
    client = _client(engine, fake_embedder)
    client.post("/ask", json={"asker_id": 10, "question": "今日の天気は？", "session_id": "ot"})
    live = [e for e, _ in _events(client, "ot")]
    assert live[-1] == "message"

    again = [e for e, _ in _events(client, "ot")]
    assert again == live  # same shape as the live run, nothing invented
    assert "route" not in again
    assert "recommend" not in again
    assert "draft" not in again


# --------------------------------------------------------------------------- #
# persistence: /ask writes a Question, C6 a Recommendation, /answer the outcome
# --------------------------------------------------------------------------- #
def test_flow_persists_question_and_recommendation(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2]),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 7, "question": GOOD_Q, "session_id": "p1"})
    _events(client, "p1")
    client.post("/answer", json={"session_id": "p1", "outcome": "accepted"})
    _events(client, "p1")

    check = get_sessionmaker(engine)()
    try:
        q = (
            check.query(Question)
            .filter(Question.asker_id == 7, Question.body == GOOD_Q)
            .order_by(Question.created_at.desc())
            .first()
        )
        assert q is not None and q.id.startswith("api_")  # uuid-based, collision-free
        assert q.topics == ["ネットワーク・VPN"]  # C1 topics backfilled
        assert q.route == "person"  # C5 route persisted (dashboard 自己解決率 source)
        recs = (
            check.query(Recommendation)
            .filter(Recommendation.question_id == q.id)
            .order_by(Recommendation.rank)
            .all()
        )
        # Every shown recommendation (rank 1..2) is persisted (codex#4).
        assert [r.rank for r in recs] == [1, 2]
        assert [r.employee_id for r in recs] == [1, 2]
        # Only the primary (rank 1, handed off) carries the outcome; the rest are
        # "shown" rows with outcome=NULL.
        assert recs[0].outcome == "accepted"
        assert recs[1].outcome is None
        # An accepted hand-off stamps the runtime resolution time (#97).
        assert q.resolved_at is not None
    finally:
        check.close()


def test_accept_with_answer_body_creates_embedded_answer_row(
    seed_counts, engine, fake_embedder
) -> None:
    # #274: when the responder accepts WITH an answer body, the flow persists an
    # ``answers`` row — attributed to the responder, tagged with the question topic,
    # and embedded — so the accumulation loop closes (reuse + history + dashboard).
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2]),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 7, "question": GOOD_Q, "session_id": "ab1"})
    _events(client, "ab1")
    resp = client.post(
        "/answer",
        json={
            "session_id": "ab1",
            "outcome": "accepted",
            "answer_body": "拠点間はIPsec VPNで設定し、各ルータのSAを揃えてください。",
        },
    )
    assert resp.status_code == 200
    _events(client, "ab1")

    check = get_sessionmaker(engine)()
    try:
        q = (
            check.query(Question)
            .filter(Question.asker_id == 7, Question.body == GOOD_Q)
            .order_by(Question.created_at.desc())
            .first()
        )
        assert q is not None
        answers = check.query(Answer).filter(Answer.question_id == q.id).all()
        assert len(answers) == 1
        ans = answers[0]
        assert ans.id.startswith("ans_")  # uuid-based, collision-free
        assert ans.responder_id == 1  # the primary (rank 1) responder
        assert ans.body == "拠点間はIPsec VPNで設定し、各ルータのSAを揃えてください。"
        assert ans.topic == "ネットワーク・VPN"  # C1 topic, drives answers_by_topic reuse
        assert ans.reuse_count == 0
        # Embedded (FakeEmbedder sizes to the configured dim) -> dense-reusable.
        assert ans.embedding is not None
        assert len(ans.embedding) == get_settings().embedding_dim
    finally:
        check.close()


def test_accept_without_answer_body_creates_no_answer_row(
    seed_counts, engine, fake_embedder
) -> None:
    # Accepting WITHOUT a body (older client / "direct" method / blank text) leaves
    # no ``answers`` row — capture is opt-in and never fabricated.
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2]),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 7, "question": GOOD_Q, "session_id": "ab2"})
    _events(client, "ab2")
    # A blank body is treated as no answer (stripped -> None), not an empty row.
    client.post("/answer", json={"session_id": "ab2", "outcome": "accepted", "answer_body": "  "})
    _events(client, "ab2")

    check = get_sessionmaker(engine)()
    try:
        q = (
            check.query(Question)
            .filter(Question.asker_id == 7, Question.body == GOOD_Q)
            .order_by(Question.created_at.desc())
            .first()
        )
        assert q is not None
        assert check.query(Answer).filter(Answer.question_id == q.id).count() == 0
    finally:
        check.close()


def test_answer_body_rejected_on_non_accept_outcome_422(seed_counts, engine, fake_embedder) -> None:
    # An answer body only belongs on an accepted hand-off; pairing it with a
    # decline (or a clarification reply) is a 422 boundary error, not silently
    # dropped (#274).
    client = _client(engine, fake_embedder)
    bad = client.post(
        "/answer",
        json={"session_id": "x1", "outcome": "declined", "answer_body": "本文"},
    )
    assert bad.status_code == 422


def test_answer_body_from_non_responder_forbidden(seed_counts, engine, fake_embedder) -> None:
    # #274 security: the asker is a valid participant (they own the clarification
    # reply) but must NOT be able to forge an answer attributed to the responder.
    # Supplying answer_body as anyone but the responder (or admin) is 403, and no
    # answers row is written.
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2]),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post(
        "/ask",
        json={"asker_id": 7, "question": GOOD_Q, "session_id": "ab3"},
        headers=_user_headers(7),
    )
    _events(client, "ab3")
    # asker (7) tries to attach an answer -> 403 (responder is employee 1).
    forbidden = client.post(
        "/answer",
        json={"session_id": "ab3", "outcome": "accepted", "answer_body": "偽の回答"},
        headers=_user_headers(7),
    )
    assert forbidden.status_code == 403

    check = get_sessionmaker(engine)()
    try:
        q = (
            check.query(Question)
            .filter(Question.asker_id == 7, Question.body == GOOD_Q)
            .order_by(Question.created_at.desc())
            .first()
        )
        assert q is not None
        assert check.query(Answer).filter(Answer.question_id == q.id).count() == 0
    finally:
        check.close()


def test_answer_capture_failure_does_not_block_accept(
    seed_counts, engine, fake_embedder, monkeypatch
) -> None:
    # #274 best-effort: if persisting the answers row fails, the accept still
    # succeeds (outcome recorded, question resolved, graph resumed) — the capture
    # runs in its own post-commit transaction and swallows failures.
    import tekijin.api.service as service_mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated answers insert failure")

    monkeypatch.setattr(service_mod, "create_answer", _boom)

    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2]),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 7, "question": GOOD_Q, "session_id": "ab4"})
    _events(client, "ab4")
    resp = client.post(
        "/answer",
        json={"session_id": "ab4", "outcome": "accepted", "answer_body": "落ちる回答"},
    )
    assert resp.status_code == 200  # the accept is not blocked by the capture failure
    done = _events(client, "ab4")
    assert done and done[-1][0] == "done"

    check = get_sessionmaker(engine)()
    try:
        q = (
            check.query(Question)
            .filter(Question.asker_id == 7, Question.body == GOOD_Q)
            .order_by(Question.created_at.desc())
            .first()
        )
        assert q is not None
        assert q.resolved_at is not None  # outcome transaction committed
        # The accepted outcome is durably recorded despite the capture failure.
        recs = check.query(Recommendation).filter(Recommendation.question_id == q.id).all()
        assert any(r.outcome == "accepted" for r in recs)
        # No answers row (the capture failed and was swallowed).
        assert check.query(Answer).filter(Answer.question_id == q.id).count() == 0
    finally:
        check.close()


def test_document_route_stamps_resolved_at(seed_counts, engine, fake_embedder) -> None:
    # A self-resolving document route records resolved_at even though no responder
    # ever accepts — so it counts toward the dashboard's avg resolution time (#97).
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(
            documents=[{"doc_id": "doc_001", "score": 0.05}],
            document_confidence=0.8,  # strongly on-topic document
            people_confidence=0.2,  # weak person signal -> document wins
            people=[1, 2, 3],
        ),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 8, "question": GOOD_Q, "session_id": "docres"})
    events = _events(client, "docres")
    names = [e for e, _ in events]
    assert "message" in names  # terminal document message, no hand-off
    # #279/#281: C6 runs on the document route to add an inline person fallback to
    # the answer, but it is NOT a live hand-off — so no `recommend` event fires and
    # NO Recommendation rows are persisted (they could never resolve on a terminal
    # document route and would inflate the pending-handoff KPI / pollute inboxes).
    assert "recommend" not in names
    message = next(data for e, data in events if e == "message")
    assert "社員1さん" in message.get("message", "")  # the inline person fallback
    assert message["fallback_responder"]["person_id"] == "E001"

    check = get_sessionmaker(engine)()
    try:
        q = check.query(Question).filter(Question.session_id == "docres").first()
        assert q is not None and q.route == "document"
        assert q.resolved_at is not None
        # No phantom recommendation rows for the self-resolving document route.
        assert check.query(Recommendation).filter(Recommendation.question_id == q.id).count() == 0
    finally:
        check.close()


def test_document_fallback_reuses_question_and_enters_existing_handoff_flow(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(
            documents=[{"doc_id": "doc_001", "score": 0.05}],
            document_confidence=0.8,
            people_confidence=0.2,
            people=[1, 2],
        ),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 8, "question": GOOD_Q, "session_id": "docfallback"})
    _events(client, "docfallback")

    ack = client.post("/handoff/document-fallback", json={"session_id": "docfallback"})
    assert ack.status_code == 200
    assert ack.json() == {"session_id": "docfallback", "status": "handoff_queued"}

    resumed = _events(client, "docfallback")
    assert [name for name, _ in resumed] == ["recommend", "draft"]
    assert resumed[0][1]["recommendations"][0]["person_id"] == "E001"
    assert "社員1さん" in resumed[1][1]["draft"]

    check = get_sessionmaker(engine)()
    try:
        questions = check.query(Question).filter(Question.session_id == "docfallback").all()
        assert len(questions) == 1  # the original question is reused, never duplicated
        question = questions[0]
        assert question.route == "person"
        assert question.resolved_at is None
        recs = (
            check.query(Recommendation)
            .filter(Recommendation.question_id == question.id)
            .order_by(Recommendation.rank)
            .all()
        )
        assert [rec.employee_id for rec in recs] == [1, 2]
    finally:
        check.close()

    handoff = client.get("/handoff/docfallback")
    assert handoff.status_code == 200
    assert handoff.json()["responder"]["person_id"] == "E001"

    duplicate = client.post("/handoff/document-fallback", json={"session_id": "docfallback"})
    assert duplicate.status_code == 409


# --------------------------------------------------------------------------- #
# self-answer (#291): a grounded answer terminates with cited source links
# --------------------------------------------------------------------------- #
def test_self_answer_message_carries_citations(seed_counts, engine, fake_embedder) -> None:
    # A strong document route + a grounded composer -> the run answers directly
    # (status=self_answered) with the cited sources, and does NOT hand off (no
    # recommend/draft events).
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(
            documents=[{"doc_id": "doc_001", "score": 0.05}],
            document_confidence=0.8,
            people_confidence=0.2,
            people=[1, 2],
        ),
        scorer=_FakeScorer(_recs(1, 2)),
        self_answer_model=_FixedSelfAnswer(
            grounded=True, answer="保守時間内に更新します。", cites=["doc_001"]
        ),
    )
    client.post("/ask", json={"asker_id": 8, "question": GOOD_Q, "session_id": "sa"})
    events = _events(client, "sa")
    names = [e for e, _ in events]
    assert "recommend" not in names and "draft" not in names  # not a hand-off
    message = next(data for e, data in events if e == "message")
    assert message["status"] == "self_answered"
    assert message["message"] == "保守時間内に更新します。"
    assert message["citations"] == [{"source_id": "doc_001", "kind": "document"}]


def test_knowledge_answer_persists_route_and_self_resolution(
    seed_counts, engine, fake_embedder
) -> None:
    # #357 slice 4c: a grounded knowledge answer terminates before C5, so the run
    # must still (a) emit a self_answered message and (b) persist a synthetic
    # "knowledge" route + mark the question self-resolved (dashboards segment by route).
    from tekijin.data.knowledge import (
        get_knowledge_unit_by_source,
        set_review_status,
        upsert_knowledge_unit,
    )
    from tekijin.knowledge.index import embed_knowledge_units

    setup = get_sessionmaker(engine)()
    try:
        upsert_knowledge_unit(
            setup,
            kind="case",
            problem=GOOD_Q,
            action="SFA/CRM を提案",
            result="受注",
            topics=["CRM・営業支援"],
            source_type="daily_report",
            source_id="kb_api_1",
            confidence=0.9,
        )
        setup.flush()
        dto = get_knowledge_unit_by_source(setup, "daily_report", "kb_api_1")
        set_review_status(setup, dto.id, "approved")
        setup.flush()
        embed_knowledge_units(setup, fake_embedder)
        setup.commit()
    finally:
        setup.close()

    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2]),
        knowledge_answer_min_similarity=0.1,
    )
    client.post("/ask", json={"asker_id": 8, "question": GOOD_Q, "session_id": "skb"})
    events = _events(client, "skb")
    assert next(d for e, d in events if e == "message")["status"] == "self_answered"

    check = get_sessionmaker(engine)()
    try:
        q = check.query(Question).filter(Question.session_id == "skb").first()
        assert q is not None and q.route == "knowledge"  # synthetic route persisted
        assert q.resolved_at is not None and q.resolution_kind == "self"
    finally:
        check.close()


def test_self_answer_on_prior_answer_route_marks_self_resolved(
    seed_counts, engine, fake_embedder
) -> None:
    # #291 review (HIGH): a grounded self-answer on the prior_answer route (which is
    # NOT stamped at c5 like document) must still count as self-resolved — else the
    # KPI this feature exists to move undercounts. resolution_kind="self" + resolved_at.
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(
            people=[1],
            past_answers=[{"qa_id": "a", "score": 0.05, "responder_id": 1}],
            answer_confidence=0.9,  # near-duplicate past QA -> prior_answer route
        ),
        self_answer_model=_FixedSelfAnswer(grounded=True, answer="過去回答より。", cites=[]),
    )
    client.post("/ask", json={"asker_id": 8, "question": GOOD_Q, "session_id": "sapa"})
    events = _events(client, "sapa")
    assert next(d for e, d in events if e == "message")["status"] == "self_answered"

    check = get_sessionmaker(engine)()
    try:
        q = check.query(Question).filter(Question.session_id == "sapa").first()
        assert q is not None and q.route == "prior_answer"
        assert q.resolved_at is not None  # stamped despite not being the document route
        assert q.resolution_kind == "self"  # counts toward the self-resolution rate
    finally:
        check.close()


# --------------------------------------------------------------------------- #
# answerability critic (#70): SSE + persistence gated on the critic's verdict
# --------------------------------------------------------------------------- #
def test_answerability_accept_surfaces_recommend_and_persists(
    seed_counts, engine, fake_embedder
) -> None:
    # A confident critic behaves exactly like the pre-#70 happy path: recommend
    # event fires and the shown rows are persisted for the outcome record.
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
        answerability_model=_FixedAnswerability(confidence=85),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "ans_ok"})
    names = [e for e, _ in _events(client, "ans_ok")]
    assert names == ["understood", "route", "recommend", "draft"]  # unchanged flow

    check = get_sessionmaker(engine)()
    try:
        q = check.query(Question).filter(Question.session_id == "ans_ok").first()
        assert q is not None
        assert check.query(Recommendation).filter(Recommendation.question_id == q.id).count() == 3
    finally:
        check.close()


def test_answerability_reject_suppresses_recommend_and_persists_nothing(
    seed_counts, engine, fake_embedder
) -> None:
    # A rejecting critic: NO recommend event (the held one is dropped), NO draft,
    # a `no_expert` terminal message, and — critically — NO Recommendation rows
    # (a rejected set must never become a phantom pending row / inbox dead-link,
    # the same integrity guard as the document route, #281).
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
        answerability_model=_FixedAnswerability(confidence=15),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "ans_ng"})
    events = _events(client, "ans_ng")
    names = [e for e, _ in events]
    assert "recommend" not in names and "draft" not in names
    assert names[-1] == "message"
    message = next(data for e, data in events if e == "message")
    assert message["status"] == "no_expert"
    assert "詳しいメンバーの記録が、組織の中に見つかりませんでした" in message["message"]

    check = get_sessionmaker(engine)()
    try:
        q = check.query(Question).filter(Question.session_id == "ans_ng").first()
        assert q is not None and q.route == "person"
        # No phantom recommendation rows for the rejected hand-off.
        assert check.query(Recommendation).filter(Recommendation.question_id == q.id).count() == 0
    finally:
        check.close()


# --------------------------------------------------------------------------- #
# validation / errors
# --------------------------------------------------------------------------- #
def test_ask_validation_422(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    assert (
        client.post("/ask", json={"asker_id": 1, "question": "", "session_id": "s"}).status_code
        == 422
    )
    assert (
        client.post("/ask", json={"question": "q", "session_id": "s"}).status_code == 422
    )  # missing asker_id


def test_answer_invalid_outcome_422(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    assert client.post("/answer", json={"session_id": "s", "outcome": "maybe"}).status_code == 422
    assert client.post("/answer", json={"session_id": "s"}).status_code == 422  # neither field


def test_events_unknown_session_404(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    assert client.get("/events/nonexistent").status_code == 404


def test_cors_headers_present(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_lifespan_startup_and_shutdown(seed_counts, engine, fake_embedder) -> None:
    # ``with TestClient`` triggers the FastAPI lifespan: startup log, then shutdown
    # calls service.close() (engine.dispose is safe — the engine re-pools on demand).
    service = AgentService(
        session_factory=get_sessionmaker(engine),
        checkpointer=MemorySaver(),
        embedder=fake_embedder,
        intent_model=KeywordIntentModel(),
        sufficiency_model=RuleSufficiencyModel(),
        draft_model=TemplateDraftModel(),
        retriever=_FakeRetriever(),
        scorer=_FakeScorer([]),
        now_factory=lambda: NOW,
    )
    with _app_client(service) as client:
        assert client.get("/health").status_code == 200


# --------------------------------------------------------------------------- #
# dashboard
# --------------------------------------------------------------------------- #
def test_dashboard_route_shape_and_seed_values(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    body = client.get("/dashboard").json()
    assert body["total_employees"] == 40
    assert body["total_questions"] == 150
    assert body["total_answers"] == 150
    assert body["recommendation_count"] == 0  # none persisted yet
    assert body["answers_per_responder"]  # load distribution from seed
    assert all("answer_count" in r for r in body["answers_per_responder"])
    assert body["topic_distribution"]  # topic mix from seed questions
    # Aggregate-only: no per-record listing (codex#5, product-spec §241-251).
    assert "recent_recommendations" not in body
    assert body["recommendation_outcomes"] == {"accepted": 0, "declined": 0, "pending": 0}
    assert body["acceptance_rate"] == 0.0
    # product-spec 画面5 metrics are present.
    assert body["self_resolution_rate"] == 0.0  # no routed questions in the seed
    assert body["avg_resolution_hours"] is not None and body["avg_resolution_hours"] > 0
    assert 0.0 <= body["top_responder_share"] <= 1.0
    assert body["latest_eval"] is None  # no eval snapshot stored yet


def test_employees_route_lists_directory_for_switcher(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    body = client.get("/employees").json()
    people = body["employees"]
    assert len(people) == 40  # the seeded roster
    # ids are the external "E###" form (round-trips as asker_id / responder id),
    # ordered by employee id ascending.
    ids = [p["id"] for p in people]
    assert ids[0] == "E001"
    assert ids == sorted(ids)
    assert all(p["name"] for p in people)  # every row has a display name
    assert all(set(p) == {"id", "name", "dept"} for p in people)


def test_inbox_lists_pending_handoff_then_clears_after_answer(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "inbox-s1"})
    _events(client, "inbox-s1")  # run to the send interrupt: persists recs + session_id

    # The primary (rank 1) responder E001 sees the pending handoff, deep-linkable
    # by its session id; the question and asker are previewed.
    body = client.get("/inbox", params={"responder_id": "E001"}).json()
    item = next(i for i in body["items"] if i["session_id"] == "inbox-s1")
    assert item["asker"]["id"] == "E010"
    assert item["question"] == GOOD_Q
    assert item["topics"]  # C1 topics persisted onto the question
    # Never-chosen defaults to "chat"; the responder must see this BEFORE
    # accepting, since "direct" means no chat thread is ever opened (#245).
    assert item["consult_method"] == "chat"

    # A non-primary candidate (E002) was not handed off -> nothing pending.
    assert client.get("/inbox", params={"responder_id": "E002"}).json()["items"] == []

    # Choosing 直接相談 at send time surfaces on the inbox item (#245).
    client.post(
        "/handoff/draft",
        json={"session_id": "inbox-s1", "draft": "直接うかがいます", "consult_method": "direct"},
    )
    direct = client.get("/inbox", params={"responder_id": "E001"}).json()
    picked = next(i for i in direct["items"] if i["session_id"] == "inbox-s1")
    assert picked["consult_method"] == "direct"

    # Once the responder accepts, the handoff clears from the inbox.
    client.post("/answer", json={"session_id": "inbox-s1", "outcome": "accepted"})
    _events(client, "inbox-s1")  # drain so the outcome is recorded
    after = client.get("/inbox", params={"responder_id": "E001"}).json()
    assert all(i["session_id"] != "inbox-s1" for i in after["items"])


def test_inbox_rejects_a_bad_responder_id(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    assert client.get("/inbox", params={"responder_id": "not-an-id"}).status_code == 422


def test_questions_route_reflects_asker_history_and_resolution(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "rh-s1"})
    _events(client, "rh-s1")  # reach the send interrupt: question persisted, not yet accepted

    body = client.get("/questions", params={"asker_id": "E010"}).json()
    mine = next(i for i in body["items"] if i["title"] == GOOD_Q)
    assert mine["resolved"] is False and mine["responder_name"] is None

    # After acceptance the same question shows resolved with the responder's name.
    client.post("/answer", json={"session_id": "rh-s1", "outcome": "accepted"})
    _events(client, "rh-s1")
    after = client.get("/questions", params={"asker_id": "E010"}).json()
    mine_after = next(i for i in after["items"] if i["title"] == GOOD_Q)
    assert mine_after["resolved"] is True and mine_after["responder_name"]


def test_questions_route_rejects_a_bad_asker_id(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    assert client.get("/questions", params={"asker_id": "nope"}).status_code == 422


def test_document_route_returns_seeded_document(seed_counts, engine, fake_embedder) -> None:
    # The document viewer (#143) fetches a cited doc by id; doc_001 is seeded.
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    resp = client.get("/documents/doc_001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "doc_001"
    assert body["title"] and body["body"]  # full content present


def test_document_route_unknown_id_is_404(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    assert client.get("/documents/doc_nope").status_code == 404


# --------------------------------------------------------------------------- #
# GET/POST /messages : chat threads on accepted recommendations (#224)
# --------------------------------------------------------------------------- #
def test_message_thread_appears_for_both_parties_after_acceptance(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-s1"})
    _events(client, "msg-s1")

    # Before acceptance, neither party has a thread yet.
    assert client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"] == []
    assert client.get("/messages/threads", params={"employee_id": "E001"}).json()["items"] == []

    client.post("/answer", json={"session_id": "msg-s1", "outcome": "accepted"})
    _events(client, "msg-s1")

    asker_threads = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"]
    responder_threads = client.get("/messages/threads", params={"employee_id": "E001"}).json()[
        "items"
    ]
    assert len(asker_threads) == 1
    assert len(responder_threads) == 1
    thread_id = asker_threads[0]["thread_id"]
    assert responder_threads[0]["thread_id"] == thread_id
    assert asker_threads[0]["counterpart"]["id"] == "E001"
    assert responder_threads[0]["counterpart"]["id"] == "E010"
    assert asker_threads[0]["question_title"] == GOOD_Q


def test_accepted_thread_is_seeded_with_the_asker_draft_as_first_message(
    seed_counts, engine, fake_embedder
) -> None:
    # The asker's own request text (the hand-off draft) must be visible in the
    # chat, not just implied by the question preview — so the responder opens
    # the thread already seeing what they were asked.
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-seed1"})
    _events(client, "msg-seed1")
    draft = client.get("/handoff/msg-seed1").json()["draft"]
    assert draft

    client.post("/answer", json={"session_id": "msg-seed1", "outcome": "accepted"})
    _events(client, "msg-seed1")
    thread_id = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"][0][
        "thread_id"
    ]

    detail = client.get(f"/messages/threads/{thread_id}", params={"employee_id": "E001"}).json()
    assert detail["messages"][0]["body"] == draft
    assert detail["messages"][0]["sender_id"] == "E010"

    listing = client.get("/messages/threads", params={"employee_id": "E001"}).json()["items"][0]
    assert listing["last_message"] == draft


def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.02) -> bool:
    """Poll ``predicate`` until it's true or ``timeout`` elapses — used to wait
    for the fire-and-forget background thread `schedule_channel_setup_and_draft`
    spawns (accept-time Slack channel setup has no request/response cycle to
    hook a synchronous check into, unlike ``BackgroundTasks``-based sends)."""

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _reset_channel_link(engine, employee_a: int, employee_b: int) -> None:
    """Delete any pre-existing SlackChannelLink for this pair.

    Every hand-off test below asks as E010 and gets routed to responder E001
    (the shared `_recs`/`_FakeRetriever` boilerplate always ranks candidate 1
    first), so every one of them shares the SAME (1, 10) pair. `engine` is a
    session-scoped fixture — the DB persists across every test in the run —
    so without this, a channel a PRIOR test created would leak into the next
    one and be silently reused instead of freshly created.
    """

    with get_sessionmaker(engine)() as session:
        link = get_channel_link(session, employee_a, employee_b)
        if link is not None:
            session.delete(link)
            session.commit()


def test_accepting_a_chat_handoff_sets_up_the_pair_channel_and_posts_the_draft(
    monkeypatch, seed_counts, engine, fake_embedder
) -> None:
    """#hand-off-chat: accepting (with BOTH parties Slack-linked) creates their
    shared private channel, invites them, then posts the draft and acceptance
    progress into it — a background thread (see `schedule_channel_setup_and_draft`),
    so this waits briefly for both messages to land rather than asserting immediately."""

    from tekijin.slack.notify import HANDOFF_ACCEPTED_TEXT

    with get_sessionmaker(engine)() as session:
        upsert_slack_link(session, 1, slack_user_id="U_RESPONDER", slack_team_id="T1", now=NOW)
        upsert_slack_link(session, 10, slack_user_id="U_ASKER", slack_team_id="T1", now=NOW)
        session.commit()
    _reset_channel_link(engine, 1, 10)

    created: list[dict] = []
    invited: list[dict] = []
    posted: list[dict] = []
    monkeypatch.setattr(
        "tekijin.slack.notify.create_private_channel",
        lambda **kw: (created.append(kw), "C1")[1],
    )
    monkeypatch.setattr(
        "tekijin.slack.notify.invite_to_channel",
        lambda **kw: (invited.append(kw), True)[1],
    )
    monkeypatch.setattr("tekijin.slack.notify.post_message", lambda **kw: posted.append(kw))
    monkeypatch.setenv("TEKIJIN_SLACK_BOT_TOKEN", "xoxb-test")
    get_settings.cache_clear()
    try:
        client = _client(
            engine,
            fake_embedder,
            retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
            scorer=_FakeScorer(_recs(1, 2, 3)),
        )
        client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-seed2"})
        _events(client, "msg-seed2")
        draft = client.get("/handoff/msg-seed2").json()["draft"]
        assert draft

        client.post("/answer", json={"session_id": "msg-seed2", "outcome": "accepted"})
        _events(client, "msg-seed2")
        thread_id = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"][
            0
        ]["thread_id"]

        assert _wait_until(lambda: len(posted) == 2)
    finally:
        get_settings.cache_clear()

    assert created == [{"bot_token": "xoxb-test", "name": "tekijin-1-10"}]
    assert invited == [
        {"bot_token": "xoxb-test", "channel_id": "C1", "user_ids": ["U_ASKER", "U_RESPONDER"]}
    ]
    assert posted == [
        {"bot_token": "xoxb-test", "channel_id": "C1", "text": draft},
        {"bot_token": "xoxb-test", "channel_id": "C1", "text": HANDOFF_ACCEPTED_TEXT},
    ]

    with get_sessionmaker(engine)() as session:
        link = get_channel_link(session, 1, 10)
        assert link is not None
        assert link.slack_channel_id == "C1"
        assert link.current_thread_id == thread_id


def test_accepting_a_direct_handoff_never_sets_up_a_slack_channel(
    monkeypatch, seed_counts, engine, fake_embedder
) -> None:
    """ "直接相談" never gets a chat thread at all (existing behaviour) — the new
    accept-time Slack hook must not fire for it either."""

    with get_sessionmaker(engine)() as session:
        upsert_slack_link(session, 1, slack_user_id="U_RESPONDER", slack_team_id="T1", now=NOW)
        upsert_slack_link(session, 10, slack_user_id="U_ASKER", slack_team_id="T1", now=NOW)
        session.commit()
    _reset_channel_link(engine, 1, 10)

    created: list[dict] = []
    monkeypatch.setattr(
        "tekijin.slack.notify.create_private_channel",
        lambda **kw: (created.append(kw), "C1")[1],
    )
    monkeypatch.setenv("TEKIJIN_SLACK_BOT_TOKEN", "xoxb-test")
    get_settings.cache_clear()
    try:
        client = _client(
            engine,
            fake_embedder,
            retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
            scorer=_FakeScorer(_recs(1, 2, 3)),
        )
        client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-seed3"})
        _events(client, "msg-seed3")
        client.post(
            "/handoff/draft",
            json={
                "session_id": "msg-seed3",
                "draft": "直接会って話します",
                "consult_method": "direct",
            },
        )
        client.post("/answer", json={"session_id": "msg-seed3", "outcome": "accepted"})
        _events(client, "msg-seed3")
        time.sleep(0.2)  # give any (unexpected) background thread a chance to run
    finally:
        get_settings.cache_clear()

    assert created == []


def test_direct_consultation_gets_no_seeded_message(seed_counts, engine, fake_embedder) -> None:
    # No chat thread exists at all for "direct", so there is nothing to seed —
    # covered here as a guard against the seeding write itself erroring out.
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-seed2"})
    _events(client, "msg-seed2")
    client.post(
        "/handoff/draft",
        json={"session_id": "msg-seed2", "draft": "本文", "consult_method": "direct"},
    )
    answered = client.post("/answer", json={"session_id": "msg-seed2", "outcome": "accepted"})
    assert answered.status_code == 200
    _events(client, "msg-seed2")

    threads = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"]
    assert all(t["question_title"] != GOOD_Q for t in threads)


# --------------------------------------------------------------------------- #
# consultation method: 直接相談 / チャットで相談
# --------------------------------------------------------------------------- #
def test_consult_method_falls_back_to_chat_for_an_unknown_stored_value(
    seed_counts, engine, fake_embedder
) -> None:
    # `questions.consult_method` is a bare VARCHAR(32) with no CHECK constraint, so
    # anything can land in it (an older client, a manual fix-up, a future value rolled
    # back). The API schema types it as Literal["direct", "chat"], so an unknown value
    # reaching the response model is a 500 on two live endpoints — GET /handoff and
    # GET /inbox — for a row that is otherwise perfectly serviceable. Snap it at the
    # DB boundary instead: "not direct" behaves as "chat", which is what every
    # downstream branch already assumes (see data/messages.py's COALESCE ... != 'direct').
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "cm-unknown"})
    _events(client, "cm-unknown")

    with session_scope(get_sessionmaker(engine)) as session:
        row = session.query(Question).filter(Question.session_id == "cm-unknown").one()
        row.consult_method = "email"  # never a valid value; simulates stale/foreign data

    handoff = client.get("/handoff/cm-unknown")
    assert handoff.status_code == 200
    assert handoff.json()["consult_method"] == "chat"

    inbox = client.get("/inbox", params={"responder_id": "E001"})
    assert inbox.status_code == 200
    item = next(i for i in inbox.json()["items"] if i["session_id"] == "cm-unknown")
    assert item["consult_method"] == "chat"


def test_consult_method_defaults_to_chat_when_never_set(seed_counts, engine, fake_embedder) -> None:
    # Backward compatibility: an asker who never calls POST /handoff/draft (or a
    # client that predates this field) behaves exactly as before — "chat" everywhere.
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "cm-default"})
    _events(client, "cm-default")

    assert client.get("/handoff/cm-default").json()["consult_method"] == "chat"

    client.post("/answer", json={"session_id": "cm-default", "outcome": "accepted"})
    _events(client, "cm-default")
    threads = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"]
    assert any(t["question_title"] == GOOD_Q for t in threads)


def test_consult_method_direct_is_visible_before_acceptance(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "cm-direct1"})
    _events(client, "cm-direct1")

    saved = client.post(
        "/handoff/draft",
        json={"session_id": "cm-direct1", "draft": "本文", "consult_method": "direct"},
    )
    assert saved.status_code == 200

    assert client.get("/handoff/cm-direct1").json()["consult_method"] == "direct"


def test_consult_method_direct_never_gets_a_chat_thread(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "cm-direct2"})
    _events(client, "cm-direct2")
    client.post(
        "/handoff/draft",
        json={"session_id": "cm-direct2", "draft": "本文", "consult_method": "direct"},
    )
    # The accepted recommendation id doubles as the thread id (#224's scheme);
    # capture it before accepting — GET /handoff 404s once the outcome lands.
    thread_id = client.get("/handoff/cm-direct2").json()["recommendation_id"]
    assert thread_id is not None

    client.post("/answer", json={"session_id": "cm-direct2", "outcome": "accepted"})
    _events(client, "cm-direct2")

    asker_threads = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"]
    responder_threads = client.get("/messages/threads", params={"employee_id": "E001"}).json()[
        "items"
    ]
    assert all(t["question_title"] != GOOD_Q for t in asker_threads)
    assert all(t["question_title"] != GOOD_Q for t in responder_threads)

    # Even knowing the id, a "direct" consultation's thread is 404 for both parties.
    assert (
        client.get(f"/messages/threads/{thread_id}", params={"employee_id": "E010"}).status_code
        == 404
    )
    assert (
        client.post(
            "/messages", json={"thread_id": thread_id, "sender_id": "E010", "body": "hi"}
        ).status_code
        == 404
    )


def test_consult_method_survives_decline_and_reroute(seed_counts, engine, fake_embedder) -> None:
    # consult_method lives on the Question (not the Recommendation), so it must
    # carry over to the NEW rank-1 recommendation a decline+reroute creates.
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "cm-reroute"})
    _events(client, "cm-reroute")
    client.post(
        "/handoff/draft",
        json={"session_id": "cm-reroute", "draft": "本文", "consult_method": "direct"},
    )

    client.post("/answer", json={"session_id": "cm-reroute", "outcome": "declined"})
    _events(client, "cm-reroute")
    assert client.get("/handoff/cm-reroute").json()["consult_method"] == "direct"


def test_send_and_list_messages_round_trip(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-s2"})
    _events(client, "msg-s2")
    client.post("/answer", json={"session_id": "msg-s2", "outcome": "accepted"})
    _events(client, "msg-s2")
    thread_id = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"][0][
        "thread_id"
    ]

    send1 = client.post(
        "/messages",
        json={"thread_id": thread_id, "sender_id": "E010", "body": "よろしくお願いします"},
    )
    assert send1.status_code == 200
    assert send1.json()["sender_id"] == "E010"
    send2 = client.post(
        "/messages", json={"thread_id": thread_id, "sender_id": "E001", "body": "承知しました"}
    )
    assert send2.status_code == 200

    for employee_id in ("E010", "E001"):
        detail = client.get(
            f"/messages/threads/{thread_id}", params={"employee_id": employee_id}
        ).json()
        # The first message is auto-seeded from the asker's hand-off draft at
        # acceptance time, so the responder opens the thread already seeing what
        # they were asked, not an empty conversation.
        bodies = [m["body"] for m in detail["messages"]]
        assert bodies[1:] == ["よろしくお願いします", "承知しました"]
        assert bodies[0]  # non-empty draft text
        assert [m["sender_id"] for m in detail["messages"]] == ["E010", "E010", "E001"]

    listing = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"][0]
    assert listing["last_message"] == "承知しました"
    assert listing["last_message_at"] is not None


def test_send_message_relays_into_the_pairs_existing_slack_channel(
    monkeypatch, seed_counts, engine, fake_embedder
) -> None:
    """An ordinary chat send, once the pair's Slack channel exists, is relayed
    into it as a background task (unlike accept-time setup, `POST /messages`
    runs inside a normal request with `BackgroundTasks` — no polling needed;
    `TestClient` runs them before returning the response)."""

    with get_sessionmaker(engine)() as session:
        upsert_slack_link(session, 1, slack_user_id="U_RESPONDER", slack_team_id="T1", now=NOW)
        upsert_slack_link(session, 10, slack_user_id="U_ASKER", slack_team_id="T1", now=NOW)
        session.commit()
    _reset_channel_link(engine, 1, 10)

    monkeypatch.setattr("tekijin.slack.notify.create_private_channel", lambda **kw: "C1")
    monkeypatch.setattr("tekijin.slack.notify.invite_to_channel", lambda **kw: True)
    monkeypatch.setattr("tekijin.slack.notify.post_message", lambda **kw: None)
    monkeypatch.setenv("TEKIJIN_SLACK_BOT_TOKEN", "xoxb-test")
    get_settings.cache_clear()
    try:
        client = _client(
            engine,
            fake_embedder,
            retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
            scorer=_FakeScorer(_recs(1, 2, 3)),
        )
        client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-slack1"})
        _events(client, "msg-slack1")
        client.post("/answer", json={"session_id": "msg-slack1", "outcome": "accepted"})
        _events(client, "msg-slack1")
        thread_id = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"][
            0
        ]["thread_id"]

        # Accept-time channel setup is its own background thread — wait for it.
        def _channel_ready() -> bool:
            with get_sessionmaker(engine)() as session:
                return get_channel_link(session, 1, 10) is not None

        assert _wait_until(_channel_ready)

        relayed: list[dict] = []
        monkeypatch.setattr("tekijin.slack.notify.post_message", lambda **kw: relayed.append(kw))

        resp = client.post(
            "/messages",
            json={"thread_id": thread_id, "sender_id": "E010", "body": "よろしくお願いします"},
        )
        assert resp.status_code == 200
    finally:
        get_settings.cache_clear()

    assert len(relayed) == 1
    assert relayed[0]["bot_token"] == "xoxb-test"
    assert relayed[0]["channel_id"] == "C1"
    assert "よろしくお願いします" in relayed[0]["text"]


def test_send_message_skips_slack_when_not_configured(
    monkeypatch, seed_counts, engine, fake_embedder
) -> None:
    """No bot token configured (the default) -> no Slack channel setup or
    relay attempted at all, even for two linked employees."""

    with get_sessionmaker(engine)() as session:
        upsert_slack_link(session, 1, slack_user_id="U_RESPONDER", slack_team_id="T1", now=NOW)
        upsert_slack_link(session, 10, slack_user_id="U_ASKER", slack_team_id="T1", now=NOW)
        session.commit()
    _reset_channel_link(engine, 1, 10)

    created: list[dict] = []
    posted: list[dict] = []
    monkeypatch.setattr(
        "tekijin.slack.notify.create_private_channel",
        lambda **kw: (created.append(kw), "C1")[1],
    )
    monkeypatch.setattr("tekijin.slack.notify.post_message", lambda **kw: posted.append(kw))

    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-slack2"})
    _events(client, "msg-slack2")
    client.post("/answer", json={"session_id": "msg-slack2", "outcome": "accepted"})
    _events(client, "msg-slack2")
    thread_id = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"][0][
        "thread_id"
    ]
    time.sleep(0.2)  # give any (unexpected) background thread a chance to run

    resp = client.post(
        "/messages",
        json={"thread_id": thread_id, "sender_id": "E010", "body": "よろしくお願いします"},
    )
    assert resp.status_code == 200
    assert created == []
    assert posted == []


# --- POST /slack/events (#388: Slack -> TEKIJIN direction) ------------------ #
SLACK_SIGNING_SECRET = "test-signing-secret"  # noqa: S105 - test fixture, not a real secret


def _slack_event_headers(body: bytes, *, timestamp: str | None = None) -> dict[str, str]:
    # The endpoint verifies against the real clock (no test seam there, unlike
    # the `now`/`NOW` used elsewhere for `created_at`), so this must be fresh.
    ts = timestamp or str(int(time.time()))
    base = f"v0:{ts}:".encode() + body
    digest = hmac.new(SLACK_SIGNING_SECRET.encode(), base, hashlib.sha256).hexdigest()
    return {"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": f"v0={digest}"}


def _post_slack_event(client: TestClient, payload: dict) -> object:
    body = json.dumps(payload).encode()
    return client.post("/slack/events", content=body, headers=_slack_event_headers(body))


def _slack_configured(monkeypatch) -> None:
    monkeypatch.setenv("TEKIJIN_SLACK_SIGNING_SECRET", SLACK_SIGNING_SECRET)
    get_settings.cache_clear()


def test_slack_events_url_verification_echoes_the_challenge(
    monkeypatch, engine, fake_embedder
) -> None:
    _slack_configured(monkeypatch)
    try:
        client = _client(engine, fake_embedder)
        resp = _post_slack_event(client, {"type": "url_verification", "challenge": "abc123"})
        assert resp.status_code == 200
        assert resp.text == "abc123"
    finally:
        get_settings.cache_clear()


def test_slack_events_rejects_an_invalid_signature(engine, fake_embedder) -> None:
    # No TEKIJIN_SLACK_SIGNING_SECRET set -> verify_signature always fails closed.
    client = _client(engine, fake_embedder)
    body = json.dumps({"type": "url_verification", "challenge": "abc"}).encode()
    resp = client.post(
        "/slack/events",
        content=body,
        headers={"X-Slack-Request-Timestamp": "1700000000", "X-Slack-Signature": "v0=deadbeef"},
    )
    assert resp.status_code == 401


def _reaction_event_payload(*, event_id: str = "Ev_react_1") -> dict:
    return {
        "type": "event_callback",
        "event_id": event_id,
        "event": {
            "type": "reaction_added",
            "reaction": "white_check_mark",
            "user": "U_REACT",
            "item": {"type": "message", "channel": "C_REACT", "ts": "1.0"},
        },
    }


def test_slack_events_reaction_added_triggers_solve_capture(
    monkeypatch, engine, fake_embedder
) -> None:
    """#476: a ✅ reaction event_callback routes to the solve-capture scheduler
    (which the flag gates) — the daemon-thread extraction itself is unit-tested."""

    calls: list[dict] = []
    monkeypatch.setattr(
        "tekijin.api.slack_routes.schedule_solve_capture", lambda _sf, **kw: calls.append(kw)
    )
    monkeypatch.setenv("TEKIJIN_SLACK_SOLVE_CAPTURE_ENABLED", "true")
    _slack_configured(monkeypatch)
    try:
        client = _client(engine, fake_embedder)
        resp = _post_slack_event(client, _reaction_event_payload())
        assert resp.status_code == 200
        assert calls == [
            {"channel_id": "C_REACT", "message_ts": "1.0", "reactor_slack_user_id": "U_REACT"}
        ]
    finally:
        get_settings.cache_clear()


def test_slack_events_reaction_added_ignored_when_capture_disabled(
    monkeypatch, engine, fake_embedder
) -> None:
    # Flag OFF (default): the reaction is a no-op — the events path is unchanged.
    calls: list[dict] = []
    monkeypatch.setattr(
        "tekijin.api.slack_routes.schedule_solve_capture", lambda _sf, **kw: calls.append(kw)
    )
    _slack_configured(monkeypatch)  # signing secret only; capture flag stays OFF
    try:
        client = _client(engine, fake_embedder)
        resp = _post_slack_event(client, _reaction_event_payload(event_id="Ev_react_2"))
        assert resp.status_code == 200
        assert calls == []
    finally:
        get_settings.cache_clear()


def test_slack_events_message_lands_in_the_channels_current_thread(
    monkeypatch, seed_counts, engine, fake_embedder
) -> None:
    """A message posted in a hand-off's shared Slack channel is mirrored into
    the TEKIJIN thread its `current_thread_id` names. No relay back into
    Slack is needed here — both parties already see it there natively, since
    they're both members of the same channel."""

    with get_sessionmaker(engine)() as session:
        upsert_slack_link(session, 1, slack_user_id="U_RESPONDER", slack_team_id="T1", now=NOW)
        upsert_slack_link(session, 10, slack_user_id="U_ASKER", slack_team_id="T1", now=NOW)
        session.commit()
    _reset_channel_link(engine, 1, 10)

    monkeypatch.setattr("tekijin.slack.notify.create_private_channel", lambda **kw: "C1")
    monkeypatch.setattr("tekijin.slack.notify.invite_to_channel", lambda **kw: True)
    monkeypatch.setattr("tekijin.slack.notify.post_message", lambda **kw: None)
    monkeypatch.setenv("TEKIJIN_SLACK_BOT_TOKEN", "xoxb-test")
    _slack_configured(monkeypatch)
    try:
        client = _client(
            engine,
            fake_embedder,
            retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
            scorer=_FakeScorer(_recs(1, 2, 3)),
        )
        client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-slack3"})
        _events(client, "msg-slack3")
        client.post("/answer", json={"session_id": "msg-slack3", "outcome": "accepted"})
        _events(client, "msg-slack3")
        thread_id = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"][
            0
        ]["thread_id"]

        def _channel_ready() -> bool:
            with get_sessionmaker(engine)() as session:
                return get_channel_link(session, 1, 10) is not None

        assert _wait_until(_channel_ready)

        resp = _post_slack_event(
            client,
            {
                "type": "event_callback",
                "event_id": "Ev-lands-in-thread",
                "event": {
                    "type": "message",
                    "channel": "C1",
                    "user": "U_RESPONDER",
                    "text": "承知しました（Slackから返信）",
                },
            },
        )
        assert resp.status_code == 200

        detail = client.get(f"/messages/threads/{thread_id}", params={"employee_id": "E010"}).json()
        bodies = [m["body"] for m in detail["messages"]]
        assert "承知しました（Slackから返信）" in bodies
        last = next(m for m in detail["messages"] if m["body"] == "承知しました（Slackから返信）")
        assert last["sender_id"] == "E001"  # the responder who replied in Slack
    finally:
        get_settings.cache_clear()


def test_slack_events_ignores_the_bots_own_message(monkeypatch, engine, fake_embedder) -> None:
    """Without this, the bot's own post landing back in the channel it just
    sent to would loop back in as if a human had sent it."""

    _slack_configured(monkeypatch)
    try:
        client = _client(engine, fake_embedder)
        resp = _post_slack_event(
            client,
            {
                "type": "event_callback",
                "event_id": "Ev-bot-own-message",
                "event": {
                    "type": "message",
                    "channel": "C1",
                    "user": "U_RESPONDER",
                    "bot_id": "B_TEKIJIN",
                    "text": "通知メッセージ本文",
                },
            },
        )
        assert resp.status_code == 200  # ack'd; the bot_id short-circuit never touches the DB
    finally:
        get_settings.cache_clear()


def test_slack_events_ignores_a_message_for_an_unmanaged_channel(
    monkeypatch, engine, fake_embedder
) -> None:
    """A message in some other Slack channel the app happens to be able to see
    (not one TEKIJIN created) is ack'd and dropped — there's no SlackChannelLink
    to route it through."""

    _slack_configured(monkeypatch)
    try:
        client = _client(engine, fake_embedder)
        resp = _post_slack_event(
            client,
            {
                "type": "event_callback",
                "event_id": "Ev-unmanaged-channel",
                "event": {
                    "type": "message",
                    "channel": "C_UNMANAGED",
                    "user": "U_RESPONDER",
                    "text": "hello",
                },
            },
        )
        assert resp.status_code == 200
    finally:
        get_settings.cache_clear()


def test_slack_events_deduplicates_a_retried_delivery(
    monkeypatch, seed_counts, engine, fake_embedder
) -> None:
    """Slack retries a delivery whenever it didn't get a fast-enough ack — the
    SAME event_id arriving twice must only create one chat message."""

    with get_sessionmaker(engine)() as session:
        upsert_slack_link(session, 1, slack_user_id="U_RESPONDER", slack_team_id="T1", now=NOW)
        upsert_slack_link(session, 10, slack_user_id="U_ASKER", slack_team_id="T1", now=NOW)
        session.commit()
    _reset_channel_link(engine, 1, 10)

    monkeypatch.setattr("tekijin.slack.notify.create_private_channel", lambda **kw: "C1")
    monkeypatch.setattr("tekijin.slack.notify.invite_to_channel", lambda **kw: True)
    monkeypatch.setattr("tekijin.slack.notify.post_message", lambda **kw: None)
    monkeypatch.setenv("TEKIJIN_SLACK_BOT_TOKEN", "xoxb-test")
    _slack_configured(monkeypatch)
    try:
        client = _client(
            engine,
            fake_embedder,
            retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
            scorer=_FakeScorer(_recs(1, 2, 3)),
        )
        client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-slack4"})
        _events(client, "msg-slack4")
        client.post("/answer", json={"session_id": "msg-slack4", "outcome": "accepted"})
        _events(client, "msg-slack4")
        thread_id = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"][
            0
        ]["thread_id"]

        def _channel_ready() -> bool:
            with get_sessionmaker(engine)() as session:
                return get_channel_link(session, 1, 10) is not None

        assert _wait_until(_channel_ready)

        payload = {
            "type": "event_callback",
            "event_id": "Ev-retry-dedup",
            "event": {
                "type": "message",
                "channel": "C1",
                "user": "U_RESPONDER",
                "text": "重複チェック用メッセージ",
            },
        }
        assert _post_slack_event(client, payload).status_code == 200
        assert _post_slack_event(client, payload).status_code == 200  # Slack's retry

        detail = client.get(f"/messages/threads/{thread_id}", params={"employee_id": "E010"}).json()
        matching = [m for m in detail["messages"] if m["body"] == "重複チェック用メッセージ"]
        assert len(matching) == 1
    finally:
        get_settings.cache_clear()


# --- POST /slack/interactivity (#398/#399: 承諾/辞退/自分より適任がいる buttons) --- #
def _post_slack_interactivity(
    client: TestClient,
    action_id: str,
    *,
    session_id: str,
    recommendation_id: int,
    outcome: str,
    slack_user_id: str,
) -> object:
    """Build a real Slack Block Kit interactivity payload (form-encoded, a
    single ``payload`` field holding the JSON) and sign it the same way Slack
    does, so this exercises the exact wire format the real button posts."""

    from urllib.parse import quote

    action_payload = {
        "actions": [
            {
                "action_id": action_id,
                "value": json.dumps(
                    {
                        "session_id": session_id,
                        "recommendation_id": recommendation_id,
                        "outcome": outcome,
                    }
                ),
            }
        ],
        "user": {"id": slack_user_id},
    }
    body = f"payload={quote(json.dumps(action_payload))}".encode()
    headers = {**_slack_event_headers(body), "Content-Type": "application/x-www-form-urlencoded"}
    return client.post("/slack/interactivity", content=body, headers=headers)


def test_slack_interactivity_accept_behaves_like_pressing_the_app_button(
    monkeypatch, seed_counts, engine, fake_embedder
) -> None:
    """A Slack 承諾 click must both record the outcome AND advance the run —
    not just queue a ``Command`` and leave the hand-off parked, which is what
    made Slack show its "processing failed" warning on the message (the click
    never resolved anything an app user would see change)."""

    _slack_configured(monkeypatch)
    try:
        with get_sessionmaker(engine)() as session:
            upsert_slack_link(session, 1, slack_user_id="U_RESPONDER", slack_team_id="T1", now=NOW)
            session.commit()

        client = _client(
            engine,
            fake_embedder,
            retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
            scorer=_FakeScorer(_recs(1, 2, 3)),
        )
        sid = "slack-interactivity-accept"
        client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": sid})
        _events(client, sid)  # drains to the "send" pause, awaiting employee 1

        recommendation_id = client.get(f"/handoff/{sid}").json()["recommendation_id"]

        resp = _post_slack_interactivity(
            client,
            "tekijin_accept",
            session_id=sid,
            recommendation_id=recommendation_id,
            outcome="accepted",
            slack_user_id="U_RESPONDER",
        )
        assert resp.status_code == 200
        assert "承諾しました" in resp.json()["text"]

        # submit_resume() ALONE already records the outcome (and, for an accept,
        # seeds the chat thread) synchronously — so a 404 from GET /handoff is
        # NOT proof the run actually advanced, only that an outcome was queued.
        # The one thing that requires the graph to genuinely run the queued
        # Command is this in-memory registry entry: ``_dispatch_stream`` clears
        # ``ctx.pending`` only once it starts executing it (service.py). If the
        # fix regresses (interactivity goes back to calling only
        # ``submit_resume``), this stays non-None forever and the assertion
        # below times out — exactly the "hand-off parked until the asker's own
        # tab reconnects" bug this change fixes, which also has a durable-state
        # consequence: the queued Command lives ONLY in this in-memory
        # registry, so an app restart before anyone drains it would strand the
        # session at "send" forever with nothing left to redo it.
        svc = client.app.state.agent_service

        def _drained() -> bool:
            ctx = svc._registry.get(sid)
            return ctx is not None and ctx.pending is None

        assert _wait_until(_drained), "承諾クリック後もグラフが進んでいない(pendingが残ったまま)"

        with get_sessionmaker(engine)() as session:
            rec = session.get(Recommendation, recommendation_id)
            assert rec.outcome == "accepted"
    finally:
        get_settings.cache_clear()


def test_slack_interactivity_refer_declines_and_reroutes_to_the_next_candidate(
    monkeypatch, seed_counts, engine, fake_embedder
) -> None:
    """自分より適任がいる (refer) carries the same "declined" outcome the app's
    own 今は難しい/自分より適任がいる buttons both send today (#76: no dedicated
    referral outcome yet) — clicking it in Slack must reroute the hand-off to
    the next candidate exactly like declining in the app does."""

    _slack_configured(monkeypatch)
    try:
        with get_sessionmaker(engine)() as session:
            upsert_slack_link(session, 1, slack_user_id="U_RESPONDER", slack_team_id="T1", now=NOW)
            session.commit()

        client = _client(
            engine,
            fake_embedder,
            retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
            scorer=_FakeScorer(_recs(1, 2)),
        )
        sid = "slack-interactivity-refer"
        client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": sid})
        first = _events(client, sid)
        assert first[2][1]["recommendations"][0]["person_id"] == "E001"

        recommendation_id = client.get(f"/handoff/{sid}").json()["recommendation_id"]

        resp = _post_slack_interactivity(
            client,
            "tekijin_refer",
            session_id=sid,
            recommendation_id=recommendation_id,
            outcome="declined",
            slack_user_id="U_RESPONDER",
        )
        assert resp.status_code == 200
        assert "自分より適任" in resp.json()["text"]

        def _rerouted_to_next_candidate() -> bool:
            resp = client.get(f"/handoff/{sid}")
            return resp.status_code == 200 and resp.json()["recommendation_id"] != recommendation_id

        assert _wait_until(_rerouted_to_next_candidate), (
            "辞退クリック後も次候補へ振り分けられていない"
        )
        assert client.get(f"/handoff/{sid}").json()["responder"]["person_id"] == "E002"

        with get_sessionmaker(engine)() as session:
            rec = session.get(Recommendation, recommendation_id)
            assert rec.outcome == "declined"
    finally:
        get_settings.cache_clear()


def test_accepting_a_second_chat_handoff_between_the_same_pair_reuses_the_channel(
    monkeypatch, seed_counts, engine, fake_embedder
) -> None:
    """Consulting the same colleague again doesn't create a second channel —
    the existing one is reused, current_thread_id moves to the new thread, and each
    acceptance still gets a visible lifecycle message."""

    from tekijin.slack.notify import HANDOFF_ACCEPTED_TEXT

    with get_sessionmaker(engine)() as session:
        upsert_slack_link(session, 1, slack_user_id="U_RESPONDER", slack_team_id="T1", now=NOW)
        upsert_slack_link(session, 10, slack_user_id="U_ASKER", slack_team_id="T1", now=NOW)
        session.commit()
    _reset_channel_link(engine, 1, 10)

    created: list[dict] = []
    posted: list[dict] = []
    monkeypatch.setattr(
        "tekijin.slack.notify.create_private_channel",
        lambda **kw: (created.append(kw), "C1")[1],
    )
    monkeypatch.setattr("tekijin.slack.notify.invite_to_channel", lambda **kw: True)
    monkeypatch.setattr("tekijin.slack.notify.post_message", lambda **kw: posted.append(kw))
    monkeypatch.setenv("TEKIJIN_SLACK_BOT_TOKEN", "xoxb-test")
    get_settings.cache_clear()
    try:
        client = _client(
            engine,
            fake_embedder,
            retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
            scorer=_FakeScorer(_recs(1, 2, 3)),
        )

        def _accept(session_id: str, question: str = GOOD_Q) -> tuple[int, str]:
            client.post(
                "/ask", json={"asker_id": 10, "question": question, "session_id": session_id}
            )
            _events(client, session_id)
            draft = client.get(f"/handoff/{session_id}").json()["draft"]
            client.post("/answer", json={"session_id": session_id, "outcome": "accepted"})
            _events(client, session_id)
            thread_id = client.get("/messages/threads", params={"employee_id": "E010"}).json()[
                "items"
            ][0]["thread_id"]
            return thread_id, draft

        first_thread_id, first_draft = _accept("msg-reuse1")

        def _channel_ready() -> bool:
            with get_sessionmaker(engine)() as session:
                return get_channel_link(session, 1, 10) is not None

        assert _wait_until(_channel_ready)
        assert _wait_until(lambda: len(posted) == 2)

        second_thread_id, second_draft = _accept("msg-reuse2", f"{GOOD_Q} 前回とは別件です。")
        assert second_thread_id != first_thread_id

        def _current_thread_is_second() -> bool:
            with get_sessionmaker(engine)() as session:
                link = get_channel_link(session, 1, 10)
                return link is not None and link.current_thread_id == second_thread_id

        assert _wait_until(_current_thread_is_second)
        assert _wait_until(lambda: len(posted) == 3)
    finally:
        get_settings.cache_clear()

    # Only ONE channel ever created — the second accept reused it.
    assert created == [{"bot_token": "xoxb-test", "name": "tekijin-1-10"}]
    assert [post["text"] for post in posted] == [
        # A newly-created channel receives its draft before the acceptance marker.
        first_draft,
        HANDOFF_ACCEPTED_TEXT,
        # A reused channel does not duplicate the draft, only the new acceptance.
        HANDOFF_ACCEPTED_TEXT,
    ]
    assert second_draft not in [post["text"] for post in posted]
    assert all(post["channel_id"] == "C1" for post in posted)
    with get_sessionmaker(engine)() as session:
        link = get_channel_link(session, 1, 10)
        assert link.slack_channel_id == "C1"
        assert link.current_thread_id == second_thread_id


def test_message_thread_rejects_non_party(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-s3"})
    _events(client, "msg-s3")
    client.post("/answer", json={"session_id": "msg-s3", "outcome": "accepted"})
    _events(client, "msg-s3")
    thread_id = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"][0][
        "thread_id"
    ]

    assert (
        client.get(f"/messages/threads/{thread_id}", params={"employee_id": "E004"}).status_code
        == 404
    )
    assert (
        client.post(
            "/messages", json={"thread_id": thread_id, "sender_id": "E004", "body": "hi"}
        ).status_code
        == 404
    )


def test_message_thread_404_before_acceptance_and_for_declined(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-s4"})
    _events(client, "msg-s4")
    pending_rid = client.get("/handoff/msg-s4").json()["recommendation_id"]

    # Not yet accepted -> 404 for both parties.
    assert (
        client.get(f"/messages/threads/{pending_rid}", params={"employee_id": "E010"}).status_code
        == 404
    )

    # Decline -> reroute. The old (now declined) recommendation id stays 404
    # forever, even after a different recommendation on the same question is
    # accepted (#94-style reroute).
    client.post("/answer", json={"session_id": "msg-s4", "outcome": "declined"})
    _events(client, "msg-s4")
    new_rid = client.get("/handoff/msg-s4").json()["recommendation_id"]
    assert new_rid != pending_rid
    client.post("/answer", json={"session_id": "msg-s4", "outcome": "accepted"})
    _events(client, "msg-s4")

    assert (
        client.get(f"/messages/threads/{pending_rid}", params={"employee_id": "E010"}).status_code
        == 404
    )
    assert (
        client.get(f"/messages/threads/{new_rid}", params={"employee_id": "E010"}).status_code
        == 200
    )


def test_send_message_rejects_blank_body(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "msg-s5"})
    _events(client, "msg-s5")
    client.post("/answer", json={"session_id": "msg-s5", "outcome": "accepted"})
    _events(client, "msg-s5")
    thread_id = client.get("/messages/threads", params={"employee_id": "E010"}).json()["items"][0][
        "thread_id"
    ]

    resp = client.post(
        "/messages", json={"thread_id": thread_id, "sender_id": "E010", "body": "   "}
    )
    assert resp.status_code == 422


def test_message_threads_route_rejects_bad_employee_id(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    assert client.get("/messages/threads", params={"employee_id": "not-an-id"}).status_code == 422


def test_dashboard_self_resolution_and_latest_eval(seed_counts, session) -> None:
    from sqlalchemy import update

    from tekijin.data.writes import insert_eval_run
    from tekijin.models.tables import Question

    # Route three seeded questions. Only ``document`` counts as self-resolved:
    # ``prior_answer`` still hands off to the pinned responder in the current graph.
    routes = {"q_0001": "document", "q_0002": "prior_answer", "q_0003": "person"}
    for qid, route in routes.items():
        session.execute(update(Question).where(Question.id == qid).values(route=route))
    session.flush()

    summary = dashboard_summary(session)
    assert summary["self_resolution_rate"] == pytest.approx(1 / 3)  # only the document route

    # No eval snapshot -> None; after storing one, the latest is returned.
    assert summary["latest_eval"] is None
    insert_eval_run(
        session,
        {"top1_accuracy": 0.7, "recall_at_3": 0.6, "mrr": 0.72, "route_accuracy": 0.7},
    )
    session.flush()
    assert dashboard_summary(session)["latest_eval"]["top1_accuracy"] == pytest.approx(0.7)


def test_top_responder_share_zero_when_no_answers() -> None:
    from tekijin.data.dashboard import _top_responder_share

    assert _top_responder_share([], 0) == 0.0  # no answers -> no concentration
    assert _top_responder_share([{"answer_count": 4}], 10) == pytest.approx(0.4)


def test_dashboard_summary_aggregates_outcomes(seed_counts, session) -> None:
    # Direct data-layer test: flushed rows are visible within the same session.
    # 2 accepted, 1 declined, 1 pending -> acceptance_rate = 2/3.
    for eid, outcome in [(3, "accepted"), (4, "accepted"), (5, "declined"), (6, None)]:
        session.add(
            Recommendation(
                question_id="q_0001", employee_id=eid, rank=1, score=0.8, outcome=outcome
            )
        )
    session.flush()
    summary = dashboard_summary(session)
    assert summary["recommendation_count"] >= 4
    assert summary["recommendation_outcomes"] == {"accepted": 2, "declined": 1, "pending": 1}
    assert summary["acceptance_rate"] == 2 / 3
    assert "recent_recommendations" not in summary  # aggregate-only


# --------------------------------------------------------------------------- #
# #294: 蓄積メトリクス (knowledge accumulation on the admin dashboard)
# --------------------------------------------------------------------------- #
# Deliberately inside a month the FIXTURES populate: `answers.json` has 16 rows
# dated 2026-08. With a September `now` the "counts only runtime rows" test passes
# even for an implementation that counts the whole table, because the seed has
# nothing in September — the guard would be green for the wrong reason.
ACC_NOW = dt.datetime(2026, 8, 25, 12, 0, 0)


def _accepted_handoff(session, *, qid: str, responder_id: int, created: dt.datetime) -> None:
    """One accepted hand-off. Fixtures write NO recommendations, so a row here is
    unambiguously runtime — that is what makes the accumulation count meaningful."""

    session.add(
        Recommendation(
            question_id=qid,
            employee_id=responder_id,
            rank=1,
            score=0.9,
            outcome="accepted",
            created_at=created,
        )
    )
    session.flush()


def _captured_answer(session, *, aid: str, qid: str, responder_id: int, created: dt.datetime):
    from tekijin.models.tables import Answer

    session.add(
        Answer(
            id=aid,
            question_id=qid,
            responder_id=responder_id,
            body="教わった内容",
            topic="ネットワーク・VPN",
            created_at=created,
            reuse_count=0,
        )
    )
    session.flush()


def _consult(session, *, qid: str, responder_id: int, created: dt.datetime) -> None:
    from tekijin.models.tables import OfflineConsult

    session.add(
        OfflineConsult(
            question_id=qid,
            responder_id=responder_id,
            asker_id=33,
            topics=["ネットワーク・VPN"],
            answer_body="対面で聞いた内容",
            resolution="resolved",
            created_at=created,
        )
    )
    session.flush()


def test_accumulation_counts_only_what_the_runtime_produced(seed_counts, session) -> None:
    """The seed ships 150 answers, 16 of them dated in the current month.

    Counting every ``answers`` row would make "今月の形式化知識量" read 16 on a
    freshly seeded database, before anyone has used the product — a headline KPI
    that is wrong in the flattering direction. Runtime origin is knowable: a
    captured answer has an ACCEPTED recommendation behind it, and fixtures write
    no ``recommendations`` at all (``seed.py`` TRUNCATEs them, never inserts).
    """

    summary = dashboard_summary(session, now=ACC_NOW)
    assert summary["knowledge_accumulation"]["this_month"] == 0


def test_accumulation_counts_captured_answers_and_retrospectives(seed_counts, session) -> None:
    _accepted_handoff(session, qid="q_0001", responder_id=1, created=ACC_NOW)
    _captured_answer(session, aid="ans_acc1", qid="q_0001", responder_id=1, created=ACC_NOW)
    # A 直接相談 write-up (#247) is the other half of the loop: no chat transcript
    # exists, so this row IS the knowledge. Fixtures write none of these either.
    _consult(session, qid="q_0002", responder_id=2, created=ACC_NOW)

    acc = dashboard_summary(session, now=ACC_NOW)["knowledge_accumulation"]
    assert acc["captured_answers"] == 1
    assert acc["consult_retrospectives"] == 1
    assert acc["this_month"] == 2


def test_accumulation_separates_this_month_from_last(seed_counts, session) -> None:
    last_month = dt.datetime(2026, 7, 20, 9, 0, 0)
    _accepted_handoff(session, qid="q_0001", responder_id=1, created=last_month)
    _captured_answer(session, aid="ans_acc_old", qid="q_0001", responder_id=1, created=last_month)
    _accepted_handoff(session, qid="q_0002", responder_id=2, created=ACC_NOW)
    _captured_answer(session, aid="ans_acc_new", qid="q_0002", responder_id=2, created=ACC_NOW)

    acc = dashboard_summary(session, now=ACC_NOW)["knowledge_accumulation"]
    assert acc["this_month"] == 1
    assert acc["last_month"] == 1


def test_accumulation_reports_the_recovery_rate_of_handoffs(seed_counts, session) -> None:
    """暗黙知の回収率: of the hand-offs someone accepted, how many left knowledge behind.

    This is the number that says whether the loop is actually closing. Two accepted
    hand-offs, one of which produced an answer row -> 0.5.
    """

    _accepted_handoff(session, qid="q_0001", responder_id=1, created=ACC_NOW)
    _captured_answer(session, aid="ans_rec1", qid="q_0001", responder_id=1, created=ACC_NOW)
    _accepted_handoff(session, qid="q_0002", responder_id=2, created=ACC_NOW)

    acc = dashboard_summary(session, now=ACC_NOW)["knowledge_accumulation"]
    assert acc["accepted_handoffs"] == 2
    assert acc["capture_rate"] == pytest.approx(0.5)


def test_accumulation_capture_rate_cannot_exceed_one_across_a_month_boundary(
    seed_counts, session
) -> None:
    """A hand-off shown last month and answered this month must not break the rate.

    ``recommendations.created_at`` is when the hand-off was SHOWN — there is no
    ``accepted_at`` — so counting answers on their own timestamp against hand-offs
    on theirs puts such a pair in the numerator and not the denominator. Here that
    arithmetic gives 2/1 = 200% captured while the one hand-off actually shown this
    month captured nothing. Both halves come from one population instead.
    """

    last_month = dt.datetime(2026, 7, 28, 9, 0, 0)
    # Shown this month, nobody wrote anything down: the only row the rate is about.
    _accepted_handoff(session, qid="q_0001", responder_id=1, created=ACC_NOW)
    # Shown last month, answered this month — twice.
    for i, (qid, responder) in enumerate(((("q_0002"), 2), (("q_0003"), 3))):
        _accepted_handoff(session, qid=qid, responder_id=responder, created=last_month)
        _captured_answer(
            session, aid=f"ans_xmonth{i}", qid=qid, responder_id=responder, created=ACC_NOW
        )

    acc = dashboard_summary(session, now=ACC_NOW)["knowledge_accumulation"]
    # The knowledge itself WAS created this month — that count stays right.
    assert acc["captured_answers"] == 2
    assert acc["accepted_handoffs"] == 1
    assert acc["capture_rate"] == 0.0, "the hand-off shown this month captured nothing"
    assert acc["capture_rate"] <= 1.0


def test_a_direct_consultation_counts_as_captured_not_as_a_miss(seed_counts, session) -> None:
    """A 直接相談 leaves no ``answers`` row — that is what it IS (#247).

    Counting only ``answers`` scored every properly-written-up direct consult as an
    uncaptured hand-off: the same function called the retrospective knowledge in
    ``this_month`` and a failure in ``capture_rate``. The loop closes perfectly and
    the KPI read 0%.
    """

    _accepted_handoff(session, qid="q_0001", responder_id=1, created=ACC_NOW)
    _consult(session, qid="q_0001", responder_id=1, created=ACC_NOW)

    acc = dashboard_summary(session, now=ACC_NOW)["knowledge_accumulation"]
    assert acc["consult_retrospectives"] == 1
    assert acc["accepted_handoffs"] == 1
    assert acc["capture_rate"] == 1.0, "a written-up direct consult is captured knowledge"


def test_an_unresolved_retrospective_is_not_counted_as_formalized_knowledge(
    seed_counts, session
) -> None:
    """「聞いたが分からなかった」 is stored, but it is not knowledge.

    Everywhere else an ``unresolved`` consult is inert (断り≠非専門). Counting it
    here would inflate the headline in the flattering direction — the exact failure
    this metric is otherwise built to avoid.
    """

    from tekijin.models.tables import OfflineConsult

    _accepted_handoff(session, qid="q_0001", responder_id=1, created=ACC_NOW)
    session.add(
        OfflineConsult(
            question_id="q_0001",
            responder_id=1,
            asker_id=33,
            topics=["ネットワーク・VPN"],
            answer_body="聞いたが解決しなかった",
            resolution="unresolved",
            created_at=ACC_NOW,
        )
    )
    session.flush()

    acc = dashboard_summary(session, now=ACC_NOW)["knowledge_accumulation"]
    assert acc["consult_retrospectives"] == 0
    assert acc["this_month"] == 0


def test_accumulation_capture_rate_is_zero_not_one_when_nothing_was_handed_off(
    seed_counts, session
) -> None:
    """An empty numerator over an empty denominator must not read as "100% captured"."""

    acc = dashboard_summary(session, now=ACC_NOW)["knowledge_accumulation"]
    assert acc["accepted_handoffs"] == 0
    assert acc["capture_rate"] == 0.0


def test_accumulation_monthly_trend_is_dense_and_oldest_first(seed_counts, session) -> None:
    """A month with nothing accumulated must appear as 0, not be missing.

    A sparse series silently rescales the chart and turns a gap into a slope.
    """

    _accepted_handoff(session, qid="q_0001", responder_id=1, created=ACC_NOW)
    _captured_answer(session, aid="ans_tr", qid="q_0001", responder_id=1, created=ACC_NOW)

    monthly = dashboard_summary(session, now=ACC_NOW)["knowledge_accumulation"]["monthly"]
    assert [m["month"] for m in monthly] == [
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
        "2026-07",
        "2026-08",
    ]
    assert monthly[-1]["count"] == 1
    assert all(m["count"] == 0 for m in monthly[:-1])


# --------------------------------------------------------------------------- #
# GET /handoff : responder-facing payload for a session paused at ``send`` (#38)
# --------------------------------------------------------------------------- #
def test_handoff_returns_responder_payload(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "h1"})
    _events(client, "h1")  # run to the send interrupt (paused, draft ready)

    resp = client.get("/handoff/h1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "h1"
    assert body["question"] == GOOD_Q
    # asker enriched from the DB and exposed in the external "E###" form.
    assert body["asker"]["id"] == "E010"
    assert body["asker"]["name"]  # employee 10 exists in the seed
    # responder = the primary (handed-off) candidate, with its selection reasons.
    assert body["responder"]["person_id"] == "E001"
    assert body["responder"]["confidence"] == "中"
    assert body["responder"]["reasons"][0]["type"] == "self"
    # slots surfaced from the understood state; draft addressed to the responder.
    assert body["topics"] == ["ネットワーク・VPN"]
    assert "社員1さん" in body["draft"]
    # reuse aggregates are present and integral (exact totals covered in the unit).
    assert isinstance(body["reuse_count"], int)
    assert isinstance(body["helpful_answer_count"], int)
    # generation token for the stale-outcome guard (#94).
    assert isinstance(body["recommendation_id"], int)


def test_stale_outcome_recommendation_id_is_rejected(seed_counts, engine, fake_embedder) -> None:
    # After a decline→reroute the primary moves on; an outcome carrying the OLD
    # recommendation id (a competing tab / late submit) must 409, never bind the
    # accept to the new candidate. The current id still succeeds (#94).
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "stale1"})
    _events(client, "stale1")
    old_rid = client.get("/handoff/stale1").json()["recommendation_id"]
    assert old_rid is not None

    client.post("/answer", json={"session_id": "stale1", "outcome": "declined"})
    _events(client, "stale1")
    new_rid = client.get("/handoff/stale1").json()["recommendation_id"]
    assert new_rid is not None and new_rid != old_rid

    stale = client.post(
        "/answer",
        json={"session_id": "stale1", "outcome": "accepted", "recommendation_id": old_rid},
    )
    assert stale.status_code == 409

    ok = client.post(
        "/answer",
        json={"session_id": "stale1", "outcome": "accepted", "recommendation_id": new_rid},
    )
    assert ok.status_code == 200


def test_stale_checkpoint_primary_rebinds_to_the_shown_candidate(
    seed_counts, engine, fake_embedder
) -> None:
    # #94-3: `primary_recommendation_id` reaches the checkpoint only at the END of a
    # stream segment, while the graph checkpoints `recommendations` per node. A
    # disconnect in between (decline -> reroute -> rows INSERTed -> GeneratorExit)
    # leaves the id on the DECLINED row while the shown candidate has moved on.
    # Recording the next outcome there would attribute it to the wrong person AND
    # wedge the hand-off (that row already carries "declined", so every later submit
    # returns "already"). The SHOWN person wins.
    service = _service(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client = _app_client(service)
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "stale2"})
    _events(client, "stale2")
    old_rid = client.get("/handoff/stale2").json()["recommendation_id"]

    client.post("/answer", json={"session_id": "stale2", "outcome": "declined"})
    _events(client, "stale2")
    new_rid = client.get("/handoff/stale2").json()["recommendation_id"]
    assert new_rid != old_rid

    check = get_sessionmaker(engine)()
    try:
        question_id = check.get(Recommendation, new_rid).question_id
    finally:
        check.close()

    # The state the lost write-back leaves behind: shown person is candidate 2,
    # the stored id still points at candidate 1's declined row.
    stale_values = {
        "question_id": question_id,
        "primary_recommendation_id": old_rid,
        "recommendations": [{"person_id": 2}],
    }
    with session_scope(service._session_factory) as session:
        assert service._resolve_primary(session, stale_values) == new_rid
        # A consistent checkpoint is left alone (no rebind, no surprise).
        consistent = {
            "question_id": question_id,
            "primary_recommendation_id": old_rid,
            "recommendations": [{"person_id": 1}],
        }
        assert service._resolve_primary(session, consistent) == old_rid
        # A shown candidate with NO persisted row resolves to None rather than to
        # someone else's row: binding this person's answer to another employee's
        # recommendation would mis-attribute `answers.responder_id` (#274). The
        # caller degrades to "no_target" instead.
        orphan = {
            "question_id": question_id,
            "primary_recommendation_id": old_rid,
            "recommendations": [{"person_id": 39}],  # never recommended here
        }
        assert service._resolve_primary(session, orphan) is None


def test_stale_checkpoint_outcome_lands_on_the_shown_candidate(
    seed_counts, engine, fake_embedder
) -> None:
    # End-to-end consequence of the above: with a stale id in the checkpoint, the
    # responder's accept must still (a) be accepted rather than 409'd by its own
    # generation token, and (b) be recorded on the SHOWN candidate's row.
    service = _service(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client = _app_client(service)
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "stale3"})
    _events(client, "stale3")
    old_rid = client.get("/handoff/stale3").json()["recommendation_id"]
    client.post("/answer", json={"session_id": "stale3", "outcome": "declined"})
    _events(client, "stale3")
    new_rid = client.get("/handoff/stale3").json()["recommendation_id"]

    # Rewind ONLY the id, exactly as the lost `_persist_run_state` would.
    db = service._session_factory()
    try:
        service._graph(db).update_state(
            service._config("stale3"), {"primary_recommendation_id": old_rid}
        )
    finally:
        db.close()

    # /handoff must hand out the RESOLVED id, or the responder would echo the
    # superseded one straight back into the #94-2 guard and 409 their own submit.
    assert client.get("/handoff/stale3").json()["recommendation_id"] == new_rid

    ok = client.post(
        "/answer",
        json={"session_id": "stale3", "outcome": "accepted", "recommendation_id": new_rid},
    )
    assert ok.status_code == 200

    check = get_sessionmaker(engine)()
    try:
        assert check.get(Recommendation, new_rid).outcome == "accepted"
        assert check.get(Recommendation, old_rid).outcome == "declined"  # untouched
    finally:
        check.close()


def test_stale_checkpoint_answer_row_is_attributed_to_the_shown_responder(
    seed_counts, engine, fake_embedder
) -> None:
    # #94-3 x #274: the captured answer must be attributed to the person who
    # actually answered. Resolving through the stale id would file candidate 2's
    # answer under candidate 1 — polluting the reuse corpus and lighting the wrong
    # employee's "回答が届きました".
    service = _service(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client = _app_client(service)
    client.post("/ask", json={"asker_id": 7, "question": GOOD_Q, "session_id": "stale4"})
    _events(client, "stale4")
    old_rid = client.get("/handoff/stale4").json()["recommendation_id"]
    client.post("/answer", json={"session_id": "stale4", "outcome": "declined"})
    _events(client, "stale4")

    db = service._session_factory()
    try:
        service._graph(db).update_state(
            service._config("stale4"), {"primary_recommendation_id": old_rid}
        )
    finally:
        db.close()

    body = "拠点間はIPsec VPNで設定し、各ルータのSAを揃えてください。"
    assert (
        client.post(
            "/answer", json={"session_id": "stale4", "outcome": "accepted", "answer_body": body}
        ).status_code
        == 200
    )

    check = get_sessionmaker(engine)()
    try:
        q = (
            check.query(Question)
            .filter(Question.asker_id == 7, Question.body == GOOD_Q)
            .order_by(Question.created_at.desc())
            .first()
        )
        answers = check.query(Answer).filter(Answer.question_id == q.id).all()
        assert len(answers) == 1
        assert answers[0].responder_id == 2  # the SHOWN candidate, not the declined 1
    finally:
        check.close()


def test_handoff_conflicts_when_awaiting_clarification(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    # Topic-only question -> paused at ``ask`` (a followup is owed to the asker).
    client.post("/ask", json={"asker_id": 10, "question": VAGUE_Q, "session_id": "h2"})
    _events(client, "h2")
    assert client.get("/handoff/h2").status_code == 409  # not a responder handoff


def test_handoff_unknown_session_404(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    assert client.get("/handoff/nonexistent").status_code == 404


def test_handoff_finished_session_404(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "h4"})
    _events(client, "h4")
    client.post("/answer", json={"session_id": "h4", "outcome": "accepted"})
    _events(client, "h4")  # run to completion (done) — no longer awaiting an outcome
    assert client.get("/handoff/h4").status_code == 404


def test_handoff_gone_once_outcome_queued(seed_counts, engine, fake_embedder) -> None:
    # After an outcome is submitted (queued in the registry) but before an /events
    # reader consumes it, the durable snapshot still shows next==("send",). The
    # handoff must report itself as no-longer-offerable so a reload does not
    # re-render the form and invite a duplicate submission.
    svc = _svc(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    svc.start_question("hq", 10, GOOD_Q)
    list(svc.stream_events("hq"))  # pause at send
    svc.submit_resume("hq", outcome="accepted")  # queues the resume (not yet drained)
    with pytest.raises(HandoffNotFound):
        svc.get_handoff("hq")


# --------------------------------------------------------------------------- #
# POST /handoff/draft : persist the asker's edited hand-off draft (#174)
# --------------------------------------------------------------------------- #
def test_handoff_draft_persists_edited_text_for_the_responder(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "hd1"})
    _events(client, "hd1")  # pause at send (draft ready)
    before = client.get("/handoff/hd1").json()
    rec_id_before = before["recommendation_id"]

    edited = "社員1さん、お忙しいところ恐れ入ります。VPNの移行の件でご相談です。"
    saved = client.post("/handoff/draft", json={"session_id": "hd1", "draft": edited})
    assert saved.status_code == 200
    assert saved.json()["status"] == "draft_saved"

    # The responder now reads the edited draft — and the outcome token is untouched.
    after = client.get("/handoff/hd1").json()
    assert after["draft"] == edited
    assert after["recommendation_id"] == rec_id_before
    # The accept/decline path still works after a draft edit.
    assert (
        client.post("/answer", json={"session_id": "hd1", "outcome": "accepted"}).status_code == 200
    )


def test_handoff_draft_rejects_blank(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "hd2"})
    _events(client, "hd2")
    assert (
        client.post("/handoff/draft", json={"session_id": "hd2", "draft": "   "}).status_code == 422
    )


def test_handoff_draft_404_when_no_pending_handoff(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    resp = client.post("/handoff/draft", json={"session_id": "nope", "draft": "本文"})
    assert resp.status_code == 404


def test_handoff_draft_409_when_awaiting_clarification(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": VAGUE_Q, "session_id": "hd3"})
    _events(client, "hd3")  # paused at ``ask`` (a clarification is owed to the asker)
    resp = client.post("/handoff/draft", json={"session_id": "hd3", "draft": "本文"})
    assert resp.status_code == 409


def test_handoff_draft_404_after_answered(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "hd4"})
    _events(client, "hd4")
    client.post("/answer", json={"session_id": "hd4", "outcome": "accepted"})
    _events(client, "hd4")  # completed
    resp = client.post("/handoff/draft", json={"session_id": "hd4", "draft": "本文"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# POST /handoff/structure : on-demand question re-draft into the four fields (#475)
# --------------------------------------------------------------------------- #
class _RecordingStructurer:
    """Structurer spy: records the (question, situation, topics) it was called with."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, list[str]]] = []

    def structure(self, question, *, situation=None, topics=None):
        from tekijin.agent.protocols import QuestionStructureResult

        self.calls.append((question, situation, list(topics or [])))
        return QuestionStructureResult(
            summary="起きていること", environment="", tried="", blocker=question
        )


def test_handoff_structure_returns_the_four_fields(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "qs1"})
    _events(client, "qs1")  # pause at send (a person is offered)

    resp = client.post("/handoff/structure", json={"session_id": "qs1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "qs1"
    # The stub seeds blocker from the raw question and leaves env/tried empty — the
    # "don't invent a field the asker never gave" contract (the asker fills them in).
    assert body["blocker"] == GOOD_Q
    assert body["summary"]  # a one-line 起きていること
    assert body["environment"] == ""
    assert body["tried"] == ""


def test_handoff_structure_reuses_c1_understanding(seed_counts, engine, fake_embedder) -> None:
    # The endpoint must feed the STORED question + C1's topics/situation to the
    # structurer (not re-run C1), so the re-draft reflects what was already parsed.
    spy = _RecordingStructurer()
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1)),
        question_structurer=spy,
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "qs2"})
    _events(client, "qs2")

    resp = client.post("/handoff/structure", json={"session_id": "qs2"})
    assert resp.status_code == 200
    assert len(spy.calls) == 1
    question, _situation, topics = spy.calls[0]
    assert question == GOOD_Q
    assert topics == ["ネットワーク・VPN"]  # C1 (KeywordIntentModel) classification


def test_handoff_structure_404_when_no_question(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    resp = client.post("/handoff/structure", json={"session_id": "nope"})
    assert resp.status_code == 404


def test_handoff_structure_forbidden_for_non_participant(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "qs3"})
    _events(client, "qs3")
    # employee 11 is neither the asker (10) nor the responder (1) nor an admin -> 403.
    resp = client.post("/handoff/structure", json={"session_id": "qs3"}, headers=_user_headers(11))
    assert resp.status_code == 403


def test_save_handoff_draft_rejects_blank_at_the_service(
    seed_counts, engine, fake_embedder
) -> None:
    # Defense in depth: a direct caller (bypassing the request schema) still cannot
    # persist a blank draft.
    svc = _svc(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    svc.start_question("hd5", 10, GOOD_Q)
    list(svc.stream_events("hd5"))  # pause at send
    with pytest.raises(SessionInvalid):
        svc.save_handoff_draft("hd5", "   ", "chat")


def test_save_handoff_draft_gone_once_outcome_queued(seed_counts, engine, fake_embedder) -> None:
    # Mirrors get_handoff: an outcome queued (not yet drained) leaves next==("send",)
    # but the hand-off is no longer editable.
    svc = _svc(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    svc.start_question("hd6", 10, GOOD_Q)
    list(svc.stream_events("hd6"))  # pause at send
    svc.submit_resume("hd6", outcome="accepted")  # queue the resume (not drained)
    with pytest.raises(HandoffNotFound):
        svc.save_handoff_draft("hd6", "編集後の本文", "chat")


def test_handoff_draft_unexpected_error_is_generic_500(seed_counts, engine, fake_embedder) -> None:
    service = _svc(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))

    def _boom(*_a, **_k):
        raise RuntimeError("secret internal detail at 10.0.0.1")

    service.save_handoff_draft = _boom  # type: ignore[method-assign]
    client = _app_client(service)
    resp = client.post("/handoff/draft", json={"session_id": "anything", "draft": "本文"})
    assert resp.status_code == 500
    assert "内部エラー" in resp.text
    assert "secret internal" not in resp.text  # detail logged, never leaked


# --------------------------------------------------------------------------- #
# POST /handoff/select : asker reselects a different shown candidate (#200/#A1)
# --------------------------------------------------------------------------- #
def test_handoff_select_reorders_and_redrafts(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "hs1"})
    _events(client, "hs1")  # pause at send, drafted for E001
    before = client.get("/handoff/hs1").json()
    assert before["responder"]["person_id"] == "E001"

    resp = client.post("/handoff/select", json={"session_id": "hs1", "person_id": "E003"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["responder"]["person_id"] == "E003"
    assert "社員3" in body["draft"]
    assert body["recommendation_id"] != before["recommendation_id"]

    after = client.get("/handoff/hs1").json()
    assert after["responder"]["person_id"] == "E003"
    assert after["draft"] == body["draft"]
    assert after["recommendation_id"] == body["recommendation_id"]

    # /inbox reflects the DB rank sync: E003 now has the pending handoff, E001 no
    # longer does.
    assert client.get("/inbox", params={"responder_id": "E003"}).json()["items"]
    assert client.get("/inbox", params={"responder_id": "E001"}).json()["items"] == []

    assert (
        client.post("/answer", json={"session_id": "hs1", "outcome": "accepted"}).status_code == 200
    )


def test_handoff_select_rejects_unknown_person_id(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "hs2"})
    _events(client, "hs2")
    resp = client.post("/handoff/select", json={"session_id": "hs2", "person_id": "E099"})
    assert resp.status_code == 422


def test_handoff_select_404_when_no_pending_handoff(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    resp = client.post("/handoff/select", json={"session_id": "nope", "person_id": "E001"})
    assert resp.status_code == 404


def test_handoff_select_409_when_awaiting_clarification(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": VAGUE_Q, "session_id": "hs3"})
    _events(client, "hs3")  # paused at ``ask`` (a clarification is owed to the asker)
    resp = client.post("/handoff/select", json={"session_id": "hs3", "person_id": "E001"})
    assert resp.status_code == 409


def test_handoff_select_404_after_answered(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "hs4"})
    _events(client, "hs4")
    client.post("/answer", json={"session_id": "hs4", "outcome": "accepted"})
    _events(client, "hs4")  # completed
    resp = client.post("/handoff/select", json={"session_id": "hs4", "person_id": "E002"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# POST /handoff/exclude : asker excludes the send target ("この人には聞かない"),
# rerouting to a freshly-scored next candidate and recording c6 feedback (#260)
# --------------------------------------------------------------------------- #
def test_handoff_exclude_reroutes_to_next_candidate(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    # Act as the asker (employee 10), not admin, so the recorded actor_id is real
    # (admin's principal has no employee_id).
    client.headers.update(_user_headers(10))
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "hx1"})
    first = _events(client, "hx1")  # pause at send, drafted for E001
    assert first[2][1]["recommendations"][0]["person_id"] == "E001"

    # Asker excludes the current send target; the reroute is queued, not streamed
    # synchronously (the open /events stream picks it up), so this acks like /answer.
    resp = client.post("/handoff/exclude", json={"session_id": "hx1", "person_id": "E001"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "reroute_queued"

    # The next /events segment re-scores excluding E001 and re-drafts for E002 —
    # the same shape a responder decline produces, but asker-initiated.
    second = _events(client, "hx1")
    assert [e for e, _ in second] == ["recommend", "draft"]
    assert second[0][1]["recommendations"][0]["person_id"] == "E002"

    check = get_sessionmaker(engine)()
    try:
        # The exclusion is recorded as a c6 feedback signal (#237 Phase 1 table).
        fb = (
            check.query(Feedback).filter(Feedback.session_id == "hx1", Feedback.stage == "c6").all()
        )
        assert len(fb) == 1
        assert fb[0].kind == "person_excluded"
        assert fb[0].target == "E001"
        assert fb[0].actor_id == 10  # the asker (from the authenticated principal)

        # An asker exclusion is NOT a responder decline, but it still TERMINATES
        # E001's shown rank-1 row (outcome="excluded", not "declined" and not NULL)
        # so it stops showing as a pending hand-off — while never counting against
        # E001's acceptance rate the way "declined" would.
        q = check.query(Question).filter(Question.session_id == "hx1").first()
        e1_rows = (
            check.query(Recommendation)
            .filter(Recommendation.question_id == q.id, Recommendation.employee_id == 1)
            .all()
        )
        assert e1_rows and all(r.outcome == "excluded" for r in e1_rows)

        # Consequence: the excluded person no longer sees this session in their
        # /inbox; the rerouted-to candidate (E002) does (pending, outcome NULL).
        from tekijin.data.inbox import pending_handoffs_for_responder

        e1_inbox = pending_handoffs_for_responder(check, 1)
        e2_inbox = pending_handoffs_for_responder(check, 2)
        assert all(item["session_id"] != "hx1" for item in e1_inbox)
        assert any(item["session_id"] == "hx1" for item in e2_inbox)
    finally:
        check.close()

    # The rerouted hand-off still completes normally — accepted by E002, the person
    # it was rerouted TO. The client is authenticated as the asker for the exclusion
    # above, and an asker may not record an outcome in the responder's name.
    assert (
        client.post(
            "/answer",
            json={"session_id": "hx1", "outcome": "accepted"},
            headers=_user_headers(2),
        ).status_code
        == 200
    )


def test_handoff_exclude_rejects_non_primary_person(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "hx2"})
    _events(client, "hx2")  # primary is E001
    # Only the send target may be excluded via the reroute path; a shown-but-not-
    # target candidate is rejected (422) rather than silently declining E001.
    resp = client.post("/handoff/exclude", json={"session_id": "hx2", "person_id": "E003"})
    assert resp.status_code == 422


def test_handoff_exclude_404_when_no_pending_handoff(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    resp = client.post("/handoff/exclude", json={"session_id": "nope", "person_id": "E001"})
    assert resp.status_code == 404


def test_handoff_exclude_409_when_awaiting_clarification(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": VAGUE_Q, "session_id": "hx3"})
    _events(client, "hx3")  # paused at ``ask`` (a clarification is owed to the asker)
    resp = client.post("/handoff/exclude", json={"session_id": "hx3", "person_id": "E001"})
    assert resp.status_code == 409


def test_handoff_exclude_404_after_answered(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "hx4"})
    _events(client, "hx4")
    client.post("/answer", json={"session_id": "hx4", "outcome": "accepted"})
    _events(client, "hx4")  # completed
    resp = client.post("/handoff/exclude", json={"session_id": "hx4", "person_id": "E001"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# POST /handoff/redraft : asker regenerates the hand-off draft ("作り直し", #260)
# --------------------------------------------------------------------------- #
class _CountingDraft:
    """A draft model that tags each generation so a redraft is observably different
    (the deterministic TemplateDraftModel would return identical text)."""

    def __init__(self) -> None:
        self.n = 0

    def draft(self, question, top, asker, missing, *, situation, topics, known_values) -> str:
        self.n += 1
        return f"下書きv{self.n}（{top['name']}さんへ）"


def _redraft_client(engine, fake_embedder, draft_model) -> TestClient:
    service = AgentService(
        session_factory=get_sessionmaker(engine),
        checkpointer=MemorySaver(),
        embedder=fake_embedder,
        intent_model=KeywordIntentModel(),
        sufficiency_model=RuleSufficiencyModel(),
        draft_model=draft_model,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
        now_factory=lambda: NOW,
    )
    client = _app_client(service)
    client.headers.update(_user_headers(10))  # act as the asker for a real actor_id
    return client


def test_handoff_redraft_regenerates_draft(seed_counts, engine, fake_embedder) -> None:
    client = _redraft_client(engine, fake_embedder, _CountingDraft())
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "rd1"})
    first = _events(client, "rd1")
    first_drafts = [d["draft"] for e, d in first if e == "draft"]
    assert first_drafts and "下書きv1" in first_drafts[-1]

    resp = client.post("/handoff/redraft", json={"session_id": "rd1"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "redraft_queued"

    # The redraft segment re-runs C7 (not C6) and re-emits just a fresh draft, then
    # re-pauses at send — the same top candidate, a newly generated text.
    second = _events(client, "rd1")
    assert [e for e, _ in second] == ["draft"]
    assert "下書きv2" in second[0][1]["draft"]

    check = get_sessionmaker(engine)()
    try:
        # The regeneration is recorded as a c7 feedback signal with the discarded
        # draft in the payload.
        fb = (
            check.query(Feedback).filter(Feedback.session_id == "rd1", Feedback.stage == "c7").all()
        )
        assert len(fb) == 1
        assert fb[0].kind == "draft_regenerated"
        assert fb[0].actor_id == 10
        assert "下書きv1" in (fb[0].payload or {}).get("previous", "")
    finally:
        check.close()

    # The regenerated hand-off still completes normally — accepted by E001, the
    # person it is drafted for (the client is authenticated as the asker).
    assert (
        client.post(
            "/answer",
            json={"session_id": "rd1", "outcome": "accepted"},
            headers=_user_headers(1),
        ).status_code
        == 200
    )


def test_handoff_redraft_404_when_no_pending_handoff(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    resp = client.post("/handoff/redraft", json={"session_id": "nope"})
    assert resp.status_code == 404


def test_handoff_redraft_409_when_awaiting_clarification(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": VAGUE_Q, "session_id": "rd2"})
    _events(client, "rd2")  # paused at ``ask`` (a clarification is owed to the asker)
    resp = client.post("/handoff/redraft", json={"session_id": "rd2"})
    assert resp.status_code == 409


def test_handoff_redraft_404_after_answered(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "rd3"})
    _events(client, "rd3")
    client.post("/answer", json={"session_id": "rd3", "outcome": "accepted"})
    _events(client, "rd3")  # completed
    resp = client.post("/handoff/redraft", json={"session_id": "rd3"})
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# POST /handoff/correct : asker corrects the AI's interpretation ("解釈の訂正"),
# re-running the whole pipeline from C1 with an added supplement (#260)
# --------------------------------------------------------------------------- #
def test_handoff_correct_reruns_pipeline_from_c1(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.headers.update(_user_headers(10))  # act as the asker for a real actor_id
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "hc1"})
    first = _events(client, "hc1")
    assert first[2][1]["recommendations"][0]["person_id"] == "E001"

    resp = client.post(
        "/handoff/correct",
        json={"session_id": "hc1", "supplement": "実は対象は5拠点で機器はUTMです"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "reinterpret_queued"

    # The correction restarts the WHOLE pipeline (reset → C1 → … → send), so the
    # segment re-emits the full understood/route/recommend/draft sequence.
    second = [e for e, _ in _events(client, "hc1")]
    assert second == ["understood", "route", "recommend", "draft"]

    # The enriched question now drives the run (visible on the responder handoff).
    handoff = client.get("/handoff/hc1").json()
    assert "5拠点" in handoff["question"]

    check = get_sessionmaker(engine)()
    try:
        # The correction is recorded as a c1 feedback signal with the supplement.
        fb = (
            check.query(Feedback).filter(Feedback.session_id == "hc1", Feedback.stage == "c1").all()
        )
        assert len(fb) == 1
        assert fb[0].kind == "interpretation_corrected"
        assert fb[0].actor_id == 10
        assert "5拠点" in (fb[0].payload or {}).get("supplement", "")

        # The abandoned original hand-off row is terminated ("superseded") so it
        # stops showing as pending; the re-run's fresh row is the live one.
        q = check.query(Question).filter(Question.session_id == "hc1").first()
        e1_rows = (
            check.query(Recommendation)
            .filter(Recommendation.question_id == q.id, Recommendation.employee_id == 1)
            .order_by(Recommendation.id)
            .all()
        )
        assert any(r.outcome == "superseded" for r in e1_rows)
        assert e1_rows[-1].outcome is None  # the newest (re-run) row is live/pending
    finally:
        check.close()

    # The re-interpreted hand-off still completes normally — accepted by E001, the
    # person the re-run drafted for (the client is authenticated as the asker).
    assert (
        client.post(
            "/answer",
            json={"session_id": "hc1", "outcome": "accepted"},
            headers=_user_headers(1),
        ).status_code
        == 200
    )


def test_handoff_correct_rejects_blank_supplement(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "hc2"})
    _events(client, "hc2")
    resp = client.post("/handoff/correct", json={"session_id": "hc2", "supplement": "   "})
    assert resp.status_code == 422  # schema rejects a blank supplement


def test_handoff_correct_404_when_no_pending_handoff(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    resp = client.post("/handoff/correct", json={"session_id": "nope", "supplement": "補足"})
    assert resp.status_code == 404


def test_handoff_correct_409_when_awaiting_clarification(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": VAGUE_Q, "session_id": "hc3"})
    _events(client, "hc3")  # paused at ``ask`` (a clarification is owed to the asker)
    resp = client.post("/handoff/correct", json={"session_id": "hc3", "supplement": "補足"})
    assert resp.status_code == 409


# --------------------------------------------------------------------------- #
# #268: an enriched question (clarification reply / interpretation correction)
# is persisted to Question.body so /inbox and /history match the processed run
# --------------------------------------------------------------------------- #
def _question_body(engine, session_id: str) -> str:
    with get_sessionmaker(engine)() as s:
        q = s.query(Question).filter(Question.session_id == session_id).first()
        return q.body if q else ""


def test_clarification_reply_persists_enriched_question_body(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": VAGUE_Q, "session_id": "qb1"})
    _events(client, "qb1")  # paused at ask (a clarification is owed)
    assert VAGUE_Q in _question_body(engine, "qb1")  # still the original so far

    client.post("/answer", json={"session_id": "qb1", "reply": "現行はVPN機器で3拠点です"})
    _events(client, "qb1")  # resumes; the ask node folds the reply into the question
    body = _question_body(engine, "qb1")
    assert "現行はVPN機器で3拠点です" in body  # #268: enriched body persisted for /inbox / /history


def test_correction_persists_enriched_question_body(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "qb2"})
    _events(client, "qb2")  # pause at send
    client.post("/handoff/correct", json={"session_id": "qb2", "supplement": "対象は5拠点です"})
    # The correction persists the enriched body durably, before the re-run streams.
    assert "対象は5拠点です" in _question_body(engine, "qb2")


def test_record_feedback_bounds_oversized_payload_values() -> None:
    """Internal callers stash drafts in ``payload``; an oversized string is
    truncated so the JSONB row cannot grow unbounded (bypasses the public 16KB
    schema cap) (#260 review MEDIUM)."""
    from tekijin.data.feedback import (
        _MAX_PAYLOAD_VALUE_CHARS,
        _TRUNCATION_MARK,
        _bounded_payload,
    )

    assert _bounded_payload(None) is None
    huge = "あ" * (_MAX_PAYLOAD_VALUE_CHARS + 500)
    out = _bounded_payload({"previous": huge, "kept": "short", "n": 3})
    assert out is not None
    assert out["previous"] == huge[:_MAX_PAYLOAD_VALUE_CHARS] + _TRUNCATION_MARK
    assert out["kept"] == "short"  # under the cap, untouched
    assert out["n"] == 3  # non-string values pass through


def test_set_recommendation_outcome_is_idempotent(seed_counts, session) -> None:
    from tekijin.data.writes import set_recommendation_outcome

    rec = Recommendation(question_id="q_0001", employee_id=3, rank=1, score=0.5, outcome=None)
    session.add(rec)
    session.flush()
    set_recommendation_outcome(session, rec.id, "accepted")
    set_recommendation_outcome(session, rec.id, "declined")  # must NOT overwrite
    session.flush()
    session.refresh(rec)
    assert rec.outcome == "accepted"  # first write wins; second is a no-op


def test_resume_reconciles_to_stored_outcome_after_restart(
    seed_counts, engine, fake_embedder
) -> None:
    # Divergence + stuck guard (#38 re-review): if a restart loses the in-memory
    # pending guard after the outcome was recorded, a second /answer with a
    # DIFFERENT action must neither overwrite the DB nor leave the graph stuck. The
    # stored outcome wins and the graph is resumed with it (consistent).
    svc = _svc(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2]),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    svc.start_question("dv", 10, GOOD_Q)
    list(svc.stream_events("dv"))  # pause at send
    svc.submit_resume("dv", outcome="accepted")  # records accepted + queues resume
    svc._registry.clear()  # simulate restart: pending guard lost, outcome persisted
    svc.submit_resume("dv", outcome="declined")  # resubmit a DIFFERENT action
    recs = _recs_for(engine, _latest_question(engine).id)
    assert recs[0].outcome == "accepted"  # first write wins; never overwritten
    # The graph advances consistently (resumed with the stored 'accepted'), not
    # left permanently paused at send.
    done = [ev.event for ev in svc.stream_events("dv")]
    assert "done" in done
    assert _recs_for(engine, _latest_question(engine).id)[0].outcome == "accepted"


def test_handoff_unexpected_error_is_generic_500(seed_counts, engine, fake_embedder) -> None:
    service = _svc(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))

    def _boom(*_a, **_k):
        raise RuntimeError("secret internal detail at 10.0.0.1")

    service.get_handoff = _boom  # type: ignore[method-assign]
    client = _app_client(service)
    resp = client.get("/handoff/anything")
    assert resp.status_code == 500
    assert "内部エラー" in resp.text
    assert "secret internal" not in resp.text  # detail logged, never leaked


def test_responder_reuse_stats_counts_reuse_and_helpful(seed_counts, session) -> None:
    from tekijin.data.handoff import responder_reuse_stats

    before = responder_reuse_stats(session, 3)
    session.add_all(
        [
            Answer(
                id="h_ra_1", question_id="q_0001", responder_id=3, reuse_count=2, was_helpful=True
            ),
            Answer(
                id="h_ra_2", question_id="q_0001", responder_id=3, reuse_count=3, was_helpful=True
            ),
            Answer(
                id="h_ra_3",
                question_id="q_0001",
                responder_id=3,
                reuse_count=None,
                was_helpful=False,
            ),
        ]
    )
    session.flush()
    after = responder_reuse_stats(session, 3)
    # NULL reuse_count coalesces to 0; only was_helpful=True rows are counted.
    assert after["reuse_count"] - before["reuse_count"] == 5
    assert after["helpful_answer_count"] - before["helpful_answer_count"] == 2


def test_employee_brief_returns_name_and_dept(seed_counts, session) -> None:
    from tekijin.data.handoff import employee_brief

    name, _dept = employee_brief(session, 1)
    assert name  # employee 1 exists in the seed
    assert employee_brief(session, 999999) == (None, None)  # unknown -> both None


# --------------------------------------------------------------------------- #
# real HybridRetriever + real scorer smoke (no fakes) — flow completes
# --------------------------------------------------------------------------- #
def test_real_agent_smoke(seed_counts, engine, fake_embedder) -> None:
    # The default (real) retriever/scorer run over the committed seed. Embeddings
    # are NULL (dense returns nothing); BM25 over seed text still drives C4, so the
    # flow completes without a crash. We only assert the deterministic prefix.
    client = _client(engine, fake_embedder)  # default real retriever/scorer
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "real"})
    names = [e for e, _ in _events(client, "real")]
    assert names[0] == "understood"
    assert "route" in names


# --------------------------------------------------------------------------- #
# PostgresSaver smoke (persistence across graph instances) — pgserver/CI
# --------------------------------------------------------------------------- #
def test_postgres_checkpointer_persists(
    seed_counts, engine, fake_embedder, database_url, test_schema
) -> None:
    # Build a PostgresSaver whose connections use the run's ISOLATED schema, so
    # its checkpoint tables land there (dropped at teardown) rather than public.
    # The ``database_url`` fixture already skips when no DB is available; when a DB
    # IS present, a setup failure must fail the test (not be swallowed into skip).
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    from tekijin.api.checkpointer import _postgres_conn_string

    def _use_schema(conn) -> None:
        conn.execute(f"SET search_path TO {test_schema}, public")

    pool = ConnectionPool(
        _postgres_conn_string(database_url),
        min_size=1,
        max_size=2,
        timeout=5.0,
        open=True,
        configure=_use_schema,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    try:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()

        client = _client(
            engine,
            fake_embedder,
            retriever=_FakeRetriever(people=[1], people_confidence=0.2),
            scorer=_FakeScorer(_recs(1)),
            checkpointer=checkpointer,
        )
        client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "pg1"})
        first = [e for e, _ in _events(client, "pg1")]
        assert first == ["understood", "route", "recommend", "draft"]

        # A brand-new client (fresh graph) sharing the SAME postgres checkpointer
        # resumes the paused run from durable state.
        client2 = _client(
            engine,
            fake_embedder,
            retriever=_FakeRetriever(people=[1], people_confidence=0.2),
            scorer=_FakeScorer(_recs(1)),
            checkpointer=checkpointer,
        )
        client2.post("/answer", json={"session_id": "pg1", "outcome": "accepted"})
        assert _events(client2, "pg1")[0][0] == "done"
    finally:
        pool.close()


# --------------------------------------------------------------------------- #
# service-level helper for durability / concurrency tests (no TestClient/SSE)
# --------------------------------------------------------------------------- #
def _svc(
    engine,
    embedder,
    *,
    retriever=None,
    scorer=None,
    now_factory=None,
    answerability_model=None,
    answerability_threshold=40,
) -> AgentService:
    return AgentService(
        session_factory=get_sessionmaker(engine),
        checkpointer=MemorySaver(),
        embedder=embedder,
        intent_model=KeywordIntentModel(),
        sufficiency_model=RuleSufficiencyModel(),
        draft_model=TemplateDraftModel(),
        answerability_model=answerability_model,
        answerability_threshold=answerability_threshold,
        retriever=retriever,
        scorer=scorer,
        now_factory=now_factory or (lambda: NOW),
    )


def _latest_question(engine) -> Question:
    check = get_sessionmaker(engine)()
    try:
        q = (
            check.query(Question)
            .filter(Question.body == GOOD_Q)
            .order_by(Question.created_at.desc())
            .first()
        )
        assert q is not None
        return q
    finally:
        check.close()


def _recs_for(engine, question_id: str) -> list[Recommendation]:
    check = get_sessionmaker(engine)()
    try:
        return (
            check.query(Recommendation)
            .filter(Recommendation.question_id == question_id)
            .order_by(Recommendation.rank)
            .all()
        )
    finally:
        check.close()


# --------------------------------------------------------------------------- #
# asker validation (404) / boundary coercion / path-safety
# --------------------------------------------------------------------------- #
def test_ask_unknown_asker_is_404(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    resp = client.post("/ask", json={"asker_id": 999999, "question": GOOD_Q, "session_id": "u404"})
    assert resp.status_code == 404  # clean boundary error, not a mid-flush FK 500


def test_ask_accepts_e_prefixed_asker_and_stores_int(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    resp = client.post("/ask", json={"asker_id": "E10", "question": GOOD_Q, "session_id": "ep"})
    assert resp.status_code == 200
    assert _latest_question(engine).asker_id == 10  # "E10" normalised to int 10


def test_session_id_must_be_path_safe(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    # A '/' would make GET /events/{session_id} unroutable — reject at the boundary.
    assert (
        client.post(
            "/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "a/b"}
        ).status_code
        == 422
    )
    assert (
        client.post("/answer", json={"session_id": "a b", "outcome": "accepted"}).status_code == 422
    )


# --------------------------------------------------------------------------- #
# route-level exception guards: unexpected errors become a generic 500
# --------------------------------------------------------------------------- #
def test_ask_unexpected_error_is_generic_500(seed_counts, engine, fake_embedder) -> None:
    service = _svc(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))

    def _boom(*_a, **_k):
        raise RuntimeError("secret internal detail at 10.0.0.1")

    service.start_question = _boom  # type: ignore[method-assign]
    client = _app_client(service)
    resp = client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "b"})
    assert resp.status_code == 500
    assert "内部エラー" in resp.text
    assert "secret internal" not in resp.text  # detail logged, never leaked


def test_answer_unexpected_error_is_generic_500(seed_counts, engine, fake_embedder) -> None:
    service = _svc(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))

    def _boom(*_a, **_k):
        raise RuntimeError("secret internal detail at 10.0.0.1")

    service.submit_resume = _boom  # type: ignore[method-assign]
    client = _app_client(service)
    resp = client.post("/answer", json={"session_id": "b", "outcome": "accepted"})
    assert resp.status_code == 500
    assert "secret internal" not in resp.text


@pytest.mark.parametrize(
    ("path", "symbol"),
    [
        ("/dashboard", "dashboard_summary"),
        ("/inbox?responder_id=1", "pending_handoffs_for_responder"),
        ("/questions?asker_id=1", "recent_questions_for_asker"),
        ("/documents/doc_anything", "get_document"),
    ],
)
def test_read_endpoint_unexpected_error_is_generic_500(
    path, symbol, seed_counts, engine, fake_embedder, monkeypatch
) -> None:
    # #146: read endpoints must mask an unexpected error as a generic 500 (logged,
    # never leaked), matching /ask, /answer, /handoff. A VALID id is passed so the
    # request reaches the data call rather than short-circuiting on 422.
    import tekijin.api.routes as routes

    def _boom(*_a, **_k):
        raise RuntimeError("secret internal detail at 10.0.0.1")

    monkeypatch.setattr(routes, symbol, _boom)
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    resp = client.get(path)
    assert resp.status_code == 500
    assert "内部エラー" in resp.text
    assert "secret internal" not in resp.text  # detail logged, never leaked


def test_employees_unexpected_error_is_generic_500(
    seed_counts, engine, fake_embedder, monkeypatch
) -> None:
    # /employees builds its repo inline, so boom via a fake Repository (#146).
    import tekijin.api.routes as routes

    class _BoomRepo:
        def __init__(self, *_a, **_k) -> None:
            pass

        def list_employees(self):
            raise RuntimeError("secret internal detail at 10.0.0.1")

    monkeypatch.setattr(routes, "Repository", _BoomRepo)
    client = _client(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    resp = client.get("/employees")
    assert resp.status_code == 500
    assert "内部エラー" in resp.text
    assert "secret internal" not in resp.text


# --------------------------------------------------------------------------- #
# backpressure: shed NEW questions with 503 when the run pool is saturated (#180)
# --------------------------------------------------------------------------- #
def test_run_slot_tracks_active_runs(seed_counts, engine, fake_embedder) -> None:
    service = _svc(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    assert service._active_runs == 0
    with service._run_slot():
        assert service._active_runs == 1
        with service._run_slot():  # nesting (e.g. a concurrent run)
            assert service._active_runs == 2
        assert service._active_runs == 1
    assert service._active_runs == 0  # released by the finally


def test_ask_sheds_with_503_when_saturated(seed_counts, engine, fake_embedder) -> None:
    service = _svc(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    service._max_concurrent_runs = 1
    service._active_runs = 1  # a run is already executing → pool full
    client = _app_client(service)
    resp = client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "busy1"})
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "5"
    assert "混雑" in resp.text


def test_ask_admission_runs_before_db_lookup(seed_counts, engine, fake_embedder) -> None:
    # Saturation is checked before asker validation and before any persist, so even a
    # bad asker_id sheds with 503 (not 404) — proving no DB work on a shed request.
    service = _svc(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    service._max_concurrent_runs = 1
    service._active_runs = 1
    client = _app_client(service)
    resp = client.post("/ask", json={"asker_id": 999999, "question": GOOD_Q, "session_id": "busy2"})
    assert resp.status_code == 503


def test_ask_not_shed_when_backpressure_disabled(seed_counts, engine, fake_embedder) -> None:
    service = _svc(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    service._max_concurrent_runs = 0  # 0 disables the gate
    service._active_runs = 100
    client = _app_client(service)
    resp = client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "ok1"})
    assert resp.status_code == 200


def test_active_runs_released_after_a_full_stream(seed_counts, engine, fake_embedder) -> None:
    # A real ask→events flow must return the slot to zero once the stream completes.
    service = _svc(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    service._max_concurrent_runs = 4
    client = _app_client(service)
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "s_bp"})
    _events(client, "s_bp")  # drain the stream fully
    assert service._active_runs == 0


def test_run_slot_released_when_stream_closed_early(seed_counts, engine, fake_embedder) -> None:
    # A client that drops mid-stream: closing the generator (GeneratorExit) must run
    # the finally and release the slot — the disconnect-release path (#180 review).
    service = _svc(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    service.start_question("s_close", 10, GOOD_Q)
    gen = service.stream_events("s_close")
    assert next(gen).event == "understood"  # execution started → slot held
    assert service._active_runs == 1
    gen.close()  # simulate a dropped client
    assert service._active_runs == 0  # finally released the slot


def test_run_slot_released_on_exception(seed_counts, engine, fake_embedder) -> None:
    # The slot's finally must fire on the exception path, not just normal completion.
    service = _svc(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    with pytest.raises(RuntimeError, match="boom"), service._run_slot():
        assert service._active_runs == 1
        raise RuntimeError("boom")
    assert service._active_runs == 0


def test_resume_not_shed_when_saturated(seed_counts, engine, fake_embedder) -> None:
    # Backpressure gates only NEW questions; a resume reaches submit_resume even while
    # saturated — a missing paused run is a 409, never a 503 (locks in the semantics).
    service = _svc(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    service._max_concurrent_runs = 1
    service._active_runs = 1  # saturated
    client = _app_client(service)
    resp = client.post("/answer", json={"session_id": "no_such", "outcome": "accepted"})
    assert resp.status_code == 409  # SessionConflict, NOT 503


def test_ask_admitted_when_below_limit(seed_counts, engine, fake_embedder) -> None:
    # Enabled but not saturated (active < max) → still admitted.
    service = _svc(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    service._max_concurrent_runs = 2
    service._active_runs = 1  # below the limit
    client = _app_client(service)
    resp = client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "under1"})
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# durability: outcome recorded from the DURABLE state (registry-independent)
# --------------------------------------------------------------------------- #
def test_outcome_recorded_after_registry_cleared(seed_counts, engine, fake_embedder) -> None:
    # Simulates a restart / eviction: the volatile registry is gone, but /answer
    # still records the outcome by reading primary_recommendation_id from the
    # durable checkpoint state.
    svc = _svc(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2]),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    svc.start_question("dur", 10, GOOD_Q)
    list(svc.stream_events("dur"))  # runs to the send interrupt; primary id in state
    svc._registry.clear()  # <- registry no longer knows anything about "dur"
    svc.submit_resume("dur", outcome="accepted")
    recs = _recs_for(engine, _latest_question(engine).id)
    assert recs[0].rank == 1 and recs[0].outcome == "accepted"  # primary recorded
    assert recs[1].outcome is None


def test_disconnect_after_recommend_then_continue_and_outcome(
    seed_counts, engine, fake_embedder
) -> None:
    # Client disconnects right after C6 (recs persisted, but update_state — which
    # writes the primary id into the state — never ran). Reconnect CONTINUES the
    # parked run to the send interrupt (codex#6), and /answer still records the
    # outcome via the DB fallback (latest rank-1 for the question).
    svc = _svc(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2]),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    svc.start_question("mid", 10, GOOD_Q)
    gen = svc.stream_events("mid")
    seen: list[str | None] = []
    for ev in gen:
        seen.append(ev.event)
        if ev.event == "recommend":
            break
    gen.close()  # type: ignore[attr-defined]  # disconnect mid-run (parked node)
    assert "recommend" in seen

    rest = [ev.event for ev in svc.stream_events("mid")]  # continue to completion
    assert "draft" in rest

    svc.submit_resume("mid", outcome="accepted")
    recs = _recs_for(engine, _latest_question(engine).id)
    assert [r.rank for r in recs] == [1, 2]  # NOT double-inserted on continuation
    assert recs[0].outcome == "accepted"  # recorded via durable DB fallback


class _FlakyAnswerability:
    """#70 critic that fails once (transient vLLM error) then accepts — to exercise
    the reconnect-after-error path where C6 already committed but the critic had
    not run in the failed segment."""

    def __init__(self, confidence: int) -> None:
        self._confidence = confidence
        self.calls = 0

    def assess(self, question, candidate_evidence):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("transient critic error")
        from tekijin.agent.protocols import AnswerabilityResult

        return AnswerabilityResult(confidence=self._confidence, reason="ok")


def test_answerability_reconnect_after_critic_error_persists(
    seed_counts, engine, fake_embedder
) -> None:
    # CRITICAL guard (#70 review): the critic raises AFTER C6 has checkpointed
    # (next=answerability) but before it accepted, so the run parks at the
    # `answerability` node with the deferred recs NOT yet persisted. On reconnect,
    # a NEW _run segment resumes at `answerability` with empty pending_* locals —
    # the fix re-derives the shown recs from durable state so the accepted hand-off
    # is still persisted + surfaced (else the outcome record is silently lost).
    critic = _FlakyAnswerability(confidence=85)
    svc = _svc(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2]),
        scorer=_FakeScorer(_recs(1, 2)),
        answerability_model=critic,
    )
    svc.start_question("recon", 10, GOOD_Q)
    first = [ev.event for ev in svc.stream_events("recon")]
    assert "error" in first  # critic raised -> parked at answerability, no persist yet
    q = _latest_question(engine)
    assert _recs_for(engine, q.id) == []  # nothing persisted on the failed segment

    # Reconnect: resumes at answerability; the critic now accepts.
    second = [ev.event for ev in svc.stream_events("recon")]
    assert "recommend" in second and "draft" in second  # released on the resumed segment
    recs = _recs_for(engine, q.id)
    assert [r.rank for r in recs] == [1, 2]  # persisted despite the mid-run error

    svc.submit_resume("recon", outcome="accepted")
    assert _recs_for(engine, q.id)[0].outcome == "accepted"  # outcome recorded, not lost


# --------------------------------------------------------------------------- #
# eviction never drops a paused (mid-interrupt) session
# --------------------------------------------------------------------------- #
def test_sweep_protects_paused_and_evicts_idle(seed_counts, engine, fake_embedder) -> None:
    svc = _svc(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    svc.start_question("keep", 10, GOOD_Q)
    list(svc.stream_events("keep"))  # paused at send
    _ = svc._lock("idle")  # materialise the idle session's lock so we can watch GC
    # "Stale" is measured against the SAME injected clock the sweep uses (#0): a
    # magic 0.0 wrongly reads as "recent" on a freshly-booted CI runner.
    stale = svc._clock() - SESSION_TTL_SECONDS - 1
    svc._registry["keep"].touched_at = stale  # make the paused session look stale
    svc._registry["idle"] = _SessionCtx(pending=None, touched_at=stale)  # stale, no run
    svc._sweep()
    assert "keep" in svc._registry  # protected: a human is being waited on
    assert "idle" not in svc._registry  # evicted: stale and not mid-interrupt
    assert "idle" not in svc._locks  # its per-session lock was GC'd too (codex#2)


# --------------------------------------------------------------------------- #
# concurrency: the per-session lock serialises accept / resume / stream
# --------------------------------------------------------------------------- #
def test_concurrent_ask_single_winner(seed_counts, engine, fake_embedder) -> None:
    import concurrent.futures as cf

    svc = _svc(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )

    def _fire() -> int:
        try:
            svc.start_question("cc", 10, GOOD_Q)
            return 200
        except SessionConflict:
            return 409

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        results = sorted(f.result() for f in [ex.submit(_fire), ex.submit(_fire)])
    assert results == [200, 409]  # exactly one accepted
    check = get_sessionmaker(engine)()
    try:
        assert check.query(Question).filter(Question.body == GOOD_Q).count() == 1
    finally:
        check.close()


def test_concurrent_events_no_double_insert(seed_counts, engine, fake_embedder) -> None:
    import concurrent.futures as cf

    svc = _svc(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2]),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    svc.start_question("ce", 10, GOOD_Q)

    def _drain() -> tuple[str | None, ...]:
        return tuple(ev.event for ev in svc.stream_events("ce"))

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        a = ex.submit(_drain)
        b = ex.submit(_drain)
        streams = {a.result(), b.result()}
    full = ("understood", "route", "recommend", "draft")
    # One streamed the whole run; the other blocked on the lock, then reconnected
    # at the send interrupt. Since #512 the reconnect rebuilds the whole progress,
    # so both consumers see the identical sequence — which is the point: a second
    # reader must not end up with a thinner view of the same run (#38 re-review).
    assert streams == {full}
    assert len(_recs_for(engine, _latest_question(engine).id)) == 2  # inserted once


def test_concurrent_answer_single_winner(seed_counts, engine, fake_embedder) -> None:
    import concurrent.futures as cf

    svc = _svc(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    svc.start_question("ca", 10, GOOD_Q)
    list(svc.stream_events("ca"))  # pause at send

    def _fire() -> str:
        try:
            svc.submit_resume("ca", outcome="accepted")
            return "ok"
        except SessionConflict:
            return "conflict"

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        results = sorted(f.result() for f in [ex.submit(_fire), ex.submit(_fire)])
    assert results == ["conflict", "ok"]  # second resume rejected (no double-queue)
    recs = _recs_for(engine, _latest_question(engine).id)
    assert len([r for r in recs if r.outcome == "accepted"]) == 1  # recorded once


def test_no_candidate_persists_no_recommendation(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1], people_confidence=0.2),
        scorer=_FakeScorer([]),  # C6 runs but yields nothing -> no_candidate
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "nc"})
    names = [e for e, _ in _events(client, "nc")]
    assert "message" in names  # no_candidate terminal
    assert _recs_for(engine, _latest_question(engine).id) == []  # nothing persisted


def test_double_queued_resume_conflicts(seed_counts, engine, fake_embedder) -> None:
    svc = _svc(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    svc.start_question("dq", 10, GOOD_Q)
    list(svc.stream_events("dq"))  # pause at send
    svc.submit_resume("dq", outcome="accepted")  # queues a resume (not yet streamed)
    with pytest.raises(SessionConflict):
        svc.submit_resume("dq", outcome="declined")  # second while queued -> 409


def test_record_outcome_warns_when_no_target(seed_counts, engine, fake_embedder, caplog) -> None:
    # Defensive branch: an outcome with no recommendation to attach it to (no id in
    # state and none in the DB) logs a warning instead of silently dropping data.
    import logging

    svc = _svc(engine, fake_embedder, retriever=_FakeRetriever(), scorer=_FakeScorer([]))
    with caplog.at_level(logging.WARNING):
        svc._record_outcome("orphan", {}, "accepted")
    assert "no recommendation to record outcome" in caplog.text


# --------------------------------------------------------------------------- #
# codex#3: a finished run replays its terminal event on reconnect
# --------------------------------------------------------------------------- #
def test_reconnect_replays_done_after_completion(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "td"})
    _events(client, "td")  # to the send interrupt
    client.post("/answer", json={"session_id": "td", "outcome": "accepted"})
    done = _events(client, "td")
    assert [e for e, _ in done] == ["done"]

    again = _events(client, "td")  # reconnect AFTER completion
    # The terminal is replayed verbatim; since #512 the progress that preceded it
    # live is rebuilt ahead of it, so the reconnecting client sees the same run.
    assert [e for e, _ in again][-1] == "done"
    assert dict(again)["done"] == done[0][1]  # identical terminal payload
    # replay must not re-run the graph / re-insert recommendations.
    assert len(_recs_for(engine, _latest_question(engine).id)) == 1


def test_reconnect_replays_terminal_message(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1], people_confidence=0.2),
        scorer=_FakeScorer([]),  # no_candidate terminal
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "tmsg"})
    first = [e for e, _ in _events(client, "tmsg")]
    assert first[-1] == "message"
    again = [e for e, _ in _events(client, "tmsg")]
    assert again[-1] == "message"  # terminal message replayed
    # ...after the progress that ran (#512). The live run also emitted an EMPTY
    # `recommend` just before the terminal; that one is deliberately not replayed,
    # because an empty C6 result is indistinguishable in the state from C6 never
    # having run, and the terminal message already says "no candidate".
    assert first == ["understood", "route", "recommend", "message"]
    assert again == ["understood", "route", "message"]


# --------------------------------------------------------------------------- #
# codex#4: Recommendation.created_at is the generation time, not the /ask time
# --------------------------------------------------------------------------- #
def test_reroute_created_at_is_generation_time_not_ask_time(
    seed_counts, engine, fake_embedder
) -> None:
    svc = _svc(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    svc.start_question("rt", 10, GOOD_Q)
    list(svc.stream_events("rt"))  # first C6 pass
    svc.submit_resume("rt", outcome="declined")
    list(svc.stream_events("rt"))  # reroute -> a later C6 pass inserts another row

    times = [r.created_at for r in _recs_for(engine, _latest_question(engine).id)]
    # DB server_default(now()) stamps the real insert time — never the injected
    # /ask NOW (which is a fixed future date in these tests).
    assert all(t != NOW for t in times)
    assert max(times) > min(times)  # the reroute pass was inserted strictly later


# --------------------------------------------------------------------------- #
# GET /notifications, POST /notifications/ack : decline notifications (#E7)
# --------------------------------------------------------------------------- #
def test_notifications_lists_decline_then_ack_clears_it(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "nt1"})
    _events(client, "nt1")  # pause at send for E001
    assert client.get("/notifications", params={"asker_id": "E010"}).json()["items"] == []

    client.post("/answer", json={"session_id": "nt1", "outcome": "declined"})
    _events(client, "nt1")  # reroute (auto-advances to E002) -> pause at send again

    body = client.get("/notifications", params={"asker_id": "E010"}).json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    # The declined person's real (seeded) name, not the _FakeScorer's fixture
    # name — Recommendation rows only ever store employee_id in the DB.
    assert item["declined_person_name"]
    assert f"{item['declined_person_name']}さんに断られた" in item["message"]
    assert item["session_id"] == "nt1"

    ack = client.post("/notifications/ack", json={"asker_id": "E010", "ids": [item["id"]]})
    assert ack.status_code == 200
    assert ack.json()["acknowledged"] == 1

    assert client.get("/notifications", params={"asker_id": "E010"}).json()["items"] == []
    # Re-acking an already-seen id is a harmless no-op.
    again = client.post("/notifications/ack", json={"asker_id": "E010", "ids": [item["id"]]})
    assert again.json()["acknowledged"] == 0


def test_notifications_scoped_to_the_owning_asker(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "nt2"})
    _events(client, "nt2")
    client.post("/answer", json={"session_id": "nt2", "outcome": "declined"})
    _events(client, "nt2")

    # A different asker cannot see or ack someone else's decline notification.
    assert client.get("/notifications", params={"asker_id": "E011"}).json()["items"] == []
    mine = client.get("/notifications", params={"asker_id": "E010"}).json()["items"]
    notif_id = mine[0]["id"]
    other_ack = client.post("/notifications/ack", json={"asker_id": "E011", "ids": [notif_id]})
    assert other_ack.json()["acknowledged"] == 0
    assert len(client.get("/notifications", params={"asker_id": "E010"}).json()["items"]) == 1


# --------------------------------------------------------------------------- #
# GET /notifications, POST /notifications/ack : accepted + request_received (#509)
# --------------------------------------------------------------------------- #
def test_notifications_lists_accepted_then_ack_clears_it(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "nt3"})
    _events(client, "nt3")  # pause at send for E001
    assert client.get("/notifications", params={"asker_id": "E010"}).json()["items"] == []

    client.post("/answer", json={"session_id": "nt3", "outcome": "accepted"})
    _events(client, "nt3")

    body = client.get("/notifications", params={"asker_id": "E010"}).json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["kind"] == "accepted"
    assert item["responder_name"]
    assert f"{item['responder_name']}さんが依頼を受け取りました" in item["message"]
    assert item["session_id"] == "nt3"
    assert item["consult_method"] in ("direct", "chat")

    ack = client.post(
        "/notifications/ack", json={"kind": "accepted", "asker_id": "E010", "ids": [item["id"]]}
    )
    assert ack.status_code == 200
    assert ack.json()["acknowledged"] == 1
    assert client.get("/notifications", params={"asker_id": "E010"}).json()["items"] == []
    # A declined-only ack (default kind) must not clear an accepted row, and
    # vice versa — the two kinds write to different columns.
    again = client.post(
        "/notifications/ack", json={"kind": "accepted", "asker_id": "E010", "ids": [item["id"]]}
    )
    assert again.json()["acknowledged"] == 0


def test_notifications_cannot_pre_ack_a_future_acceptance(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "nt-preack"})
    _events(client, "nt-preack")
    recommendation_id = client.get("/handoff/nt-preack").json()["recommendation_id"]

    # The row belongs to this asker's question, but it is not an acceptance
    # notification yet. Pre-acking it must not suppress the future event.
    ack = client.post(
        "/notifications/ack",
        json={"kind": "accepted", "asker_id": "E010", "ids": [recommendation_id]},
    )
    assert ack.status_code == 200
    assert ack.json()["acknowledged"] == 0

    client.post(
        "/answer",
        json={
            "session_id": "nt-preack",
            "outcome": "accepted",
            "recommendation_id": recommendation_id,
        },
    )
    _events(client, "nt-preack")
    items = client.get("/notifications", params={"asker_id": "E010"}).json()["items"]
    assert any(item["kind"] == "accepted" and item["id"] == recommendation_id for item in items)


def test_notifications_lists_incoming_request_then_ack_clears_it(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    assert client.get("/notifications", params={"employee_id": "E001"}).json()["items"] == []

    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "nt4"})
    _events(client, "nt4")  # pause at send for E001 — a still-pending handoff

    body = client.get("/notifications", params={"employee_id": "E001"}).json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["kind"] == "request_received"
    assert item["asker_name"]
    assert f"{item['asker_name']}さんから新しい依頼が届きました" in item["message"]
    assert item["session_id"] == "nt4"

    ack = client.post(
        "/notifications/ack",
        json={"kind": "request_received", "employee_id": "E001", "ids": [item["id"]]},
    )
    assert ack.status_code == 200
    assert ack.json()["acknowledged"] == 1
    assert client.get("/notifications", params={"employee_id": "E001"}).json()["items"] == []


def test_notifications_requires_exactly_one_of_asker_or_employee_id(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    assert client.get("/notifications").status_code == 422
    assert (
        client.get("/notifications", params={"asker_id": "E010", "employee_id": "E001"}).status_code
        == 422
    )


def test_notification_ack_requires_only_the_owner_id_matching_kind(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(engine, fake_embedder)
    assert (
        client.post(
            "/notifications/ack",
            json={
                "kind": "accepted",
                "asker_id": "E010",
                "employee_id": "E001",
                "ids": [1],
            },
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/notifications/ack",
            json={
                "kind": "request_received",
                "asker_id": "E010",
                "employee_id": "E001",
                "ids": [1],
            },
        ).status_code
        == 422
    )


# --------------------------------------------------------------------------- #
# #207: delete a past question (and its FK children), owner/admin only
# --------------------------------------------------------------------------- #
def _insert_question(engine, qid: str, asker_id: int, *, with_children: bool = False) -> None:
    """Directly seed one ``api_``-prefixed question (optionally with FK children)
    so the delete tests do not depend on driving the whole SSE flow."""

    factory = get_sessionmaker(engine)
    with factory() as s:
        s.add(
            Question(
                id=qid,
                asker_id=asker_id,
                body="削除テスト用の質問",
                topics=[],
                status="open",
                created_at=NOW,
                session_id=None,
            )
        )
        s.flush()  # the question must exist before its FK children insert
        if with_children:
            s.add(Answer(id=f"{qid}_a", question_id=qid, responder_id=1, body="回答本文"))
            rec = Recommendation(question_id=qid, employee_id=1, rank=1, score=0.9)
            s.add(rec)
            s.add(Event(question_id=qid, stage="c1", started_at=NOW, ended_at=NOW, meta=None))
            s.flush()  # rec.id (autoincrement) must exist before the message FKs it
            s.add(Message(recommendation_id=rec.id, sender_id=1, body="チャット本文"))
            # A learning signal keyed to this question. Every correction the asker
            # makes (draft edits, person exclusions, re-runs) writes one of these.
            s.add(
                Feedback(
                    question_id=qid,
                    stage="c7",
                    kind="draft_edited",
                    payload={"generated": "旧", "sent": "新"},
                    actor_id=asker_id,
                )
            )
        s.commit()


def _counts_for(engine, qid: str) -> tuple[int, int, int, int, int, int]:
    factory = get_sessionmaker(engine)
    with factory() as s:
        recommendation_ids = [
            rec_id
            for (rec_id,) in s.query(Recommendation.id)
            .filter(Recommendation.question_id == qid)
            .all()
        ]
        message_count = (
            s.query(Message).filter(Message.recommendation_id.in_(recommendation_ids)).count()
            if recommendation_ids
            else 0
        )
        return (
            s.query(Question).filter(Question.id == qid).count(),
            s.query(Answer).filter(Answer.question_id == qid).count(),
            len(recommendation_ids),
            s.query(Event).filter(Event.question_id == qid).count(),
            message_count,
            s.query(Feedback).filter(Feedback.question_id == qid).count(),
        )


def test_delete_question_removes_it_and_its_children(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)  # admin
    _insert_question(engine, "api_del1", 10, with_children=True)
    # question, answer, recommendation, event, and its chat message all exist.
    assert _counts_for(engine, "api_del1") == (1, 1, 1, 1, 1, 1)

    resp = client.delete("/questions/api_del1")
    assert resp.status_code == 200
    assert resp.json() == {"question_id": "api_del1", "deleted": True}
    # question and ALL its FK children — including chat messages, one hop out
    # via the recommendation, and the feedback rows (#286) — are gone.
    assert _counts_for(engine, "api_del1") == (0, 0, 0, 0, 0, 0)


def test_owner_can_delete_their_own_question(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    _insert_question(engine, "api_del2", 10)
    resp = client.delete("/questions/api_del2", headers=_user_headers(10))
    assert resp.status_code == 200
    assert _counts_for(engine, "api_del2")[0] == 0


def test_delete_question_forbidden_for_non_owner(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    _insert_question(engine, "api_del3", 10)
    # employee 11 is not the asker and not an admin -> 403, row untouched.
    resp = client.delete("/questions/api_del3", headers=_user_headers(11))
    assert resp.status_code == 403
    assert _counts_for(engine, "api_del3")[0] == 1


def test_delete_missing_question_returns_404(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    resp = client.delete("/questions/api_does_not_exist")
    assert resp.status_code == 404


def test_deleted_question_drops_out_of_recent_list(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    _insert_question(engine, "api_del4", 10)
    before = client.get("/questions", params={"asker_id": "E010"}).json()["items"]
    assert any(i["question_id"] == "api_del4" for i in before)

    client.delete("/questions/api_del4")
    after = client.get("/questions", params={"asker_id": "E010"}).json()["items"]
    assert all(i["question_id"] != "api_del4" for i in after)


# --------------------------------------------------------------------------- #
# #159: POST /questions/{id}/resolve — the asker marks a question self-resolved
# (人を介さず解決した), feeding the dashboard self-resolution rate
# --------------------------------------------------------------------------- #
def _set_route(engine, qid: str, route: str) -> None:
    from sqlalchemy import update

    with get_sessionmaker(engine)() as s:
        s.execute(update(Question).where(Question.id == qid).values(route=route))
        s.commit()


def test_resolve_marks_self_resolution_and_shows_in_history(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(engine, fake_embedder)
    _insert_question(engine, "api_res1", 10)
    resp = client.post("/questions/api_res1/resolve", headers=_user_headers(10))
    assert resp.status_code == 200 and resp.json() == {"question_id": "api_res1", "resolved": True}

    with get_sessionmaker(engine)() as s:
        q = s.query(Question).filter(Question.id == "api_res1").first()
        assert q.resolution_kind == "self" and q.resolved_at == NOW

    # The asker's history now labels it a self-resolution.
    item = next(
        i
        for i in client.get("/questions", params={"asker_id": "E010"}).json()["items"]
        if i["question_id"] == "api_res1"
    )
    assert item["resolution"] == "self" and item["resolved"] is True


def test_resolve_is_idempotent_and_first_wins(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    _insert_question(engine, "api_res2", 10)
    assert client.post("/questions/api_res2/resolve", headers=_user_headers(10)).status_code == 200
    # Re-marking is a no-op that still acks; resolved_at is not moved.
    assert client.post("/questions/api_res2/resolve", headers=_user_headers(10)).status_code == 200
    with get_sessionmaker(engine)() as s:
        q = s.query(Question).filter(Question.id == "api_res2").first()
        assert q.resolution_kind == "self" and q.resolved_at == NOW


def test_resolve_forbidden_for_non_owner(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    _insert_question(engine, "api_res3", 10)
    resp = client.post("/questions/api_res3/resolve", headers=_user_headers(11))
    assert resp.status_code == 403
    with get_sessionmaker(engine)() as s:
        q = s.query(Question).filter(Question.id == "api_res3").first()
        assert q.resolution_kind is None  # untouched


def test_resolve_missing_question_returns_404(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    assert client.post("/questions/api_missing/resolve").status_code == 404


def test_self_resolution_counts_in_dashboard_rate(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    _insert_question(engine, "api_res4", 10)
    _set_route(engine, "api_res4", "send")  # a routed, person-bound question (real C5 route)
    with get_sessionmaker(engine)() as s:
        before = dashboard_summary(s)["self_resolution_rate"]
    client.post("/questions/api_res4/resolve", headers=_user_headers(10))
    with get_sessionmaker(engine)() as s:
        after = dashboard_summary(s)["self_resolution_rate"]
    # The explicit self-resolution counts in the numerator: the question was routed
    # to a person but the asker solved it WITHOUT anyone actually answering.
    assert after > before


def test_person_answered_question_is_not_counted_self_even_if_marked(
    seed_counts, engine, fake_embedder
) -> None:
    """A question a person actually resolved must never count as self-resolved, even
    if the asker also clicked "自分で解決した" (race: self-resolve while pending, then a
    responder accepts). The dashboard mirrors history's person>self precedence (#159
    review HIGH)."""
    from sqlalchemy import update

    client = _client(engine, fake_embedder)
    _insert_question(engine, "api_res5", 10)
    _set_route(engine, "api_res5", "send")
    # The asker self-resolves while the hand-off is still pending.
    client.post("/questions/api_res5/resolve", headers=_user_headers(10))
    with get_sessionmaker(engine)() as s:
        after_self = dashboard_summary(s)["self_resolution_rate"]
    # A responder's recommendation is THEN accepted (a real person resolution).
    with get_sessionmaker(engine)() as s:
        rec = Recommendation(
            question_id="api_res5", employee_id=1, rank=1, score=0.5, outcome="accepted"
        )
        s.add(rec)
        s.commit()
    with get_sessionmaker(engine)() as s:
        after_person = dashboard_summary(s)["self_resolution_rate"]
        # The self-mark row remains, but the metric no longer counts it as self.
        q = s.query(Question).filter(Question.id == "api_res5").first()
        assert q.resolution_kind == "self"  # the label persists (history shows person)
    assert after_person < after_self  # the accepted person-resolution removed it from self

    # And guarding the write path: a question ALREADY person-resolved (resolved_at
    # set) cannot be re-labelled self.
    _insert_question(engine, "api_res6", 10)
    with get_sessionmaker(engine)() as s:
        s.execute(update(Question).where(Question.id == "api_res6").values(resolved_at=NOW))
        s.commit()
    client.post("/questions/api_res6/resolve", headers=_user_headers(10))
    with get_sessionmaker(engine)() as s:
        q = s.query(Question).filter(Question.id == "api_res6").first()
        assert q.resolution_kind is None  # guard held: not re-labelled self


# --------------------------------------------------------------------------- #
# #208: /questions honours a `limit` (history screen requests many, panel few)
# --------------------------------------------------------------------------- #
def test_questions_limit_caps_and_orders_newest_first(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    # Seven questions strictly newer than any seeded row (created_at in the future),
    # so they are deterministically the newest regardless of seed history.
    factory = get_sessionmaker(engine)
    with factory() as s:
        for i in range(7):
            s.add(
                Question(
                    id=f"api_hist{i}",
                    asker_id=10,
                    body=f"履歴テスト{i}",
                    topics=[],
                    status="open",
                    created_at=NOW + dt.timedelta(minutes=i),
                    session_id=None,
                )
            )
        s.commit()

    # limit=3 -> exactly the 3 newest (hist6, hist5, hist4), newest first.
    got = client.get("/questions", params={"asker_id": "E010", "limit": 3}).json()["items"]
    assert [i["question_id"] for i in got] == ["api_hist6", "api_hist5", "api_hist4"]

    # A larger limit returns more of them (all 7 api_ ones are newest).
    many = client.get("/questions", params={"asker_id": "E010", "limit": 50}).json()["items"]
    top7 = [i["question_id"] for i in many[:7]]
    assert top7 == [f"api_hist{i}" for i in range(6, -1, -1)]


def test_questions_limit_is_validated(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    # Out-of-range limits are rejected (1..200).
    assert client.get("/questions", params={"asker_id": "E010", "limit": 0}).status_code == 422
    assert client.get("/questions", params={"asker_id": "E010", "limit": 201}).status_code == 422


# --------------------------------------------------------------------------- #
# #237 Phase 1: record feedback (the correction signal the runtime discarded)
# --------------------------------------------------------------------------- #
def _feedback_rows(engine) -> list[Feedback]:
    with get_sessionmaker(engine)() as s:
        return s.query(Feedback).order_by(Feedback.id).all()


# --------------------------------------------------------------------------- #
# #247: 直接相談のふりかえり (offline consult retrospective)
# --------------------------------------------------------------------------- #
def _consult_rows(engine):
    from tekijin.models.tables import OfflineConsult

    with session_scope(get_sessionmaker(engine)) as session:
        return session.query(OfflineConsult).order_by(OfflineConsult.id).all()


def _question_id_for_session(engine, session_id: str) -> str:
    from sqlalchemy import select

    with session_scope(get_sessionmaker(engine)) as session:
        return session.execute(
            select(Question.id).where(Question.session_id == session_id)
        ).scalar_one()


def _direct_consultation(
    engine,
    fake_embedder,
    *,
    session_id: str,
    asker_id: int = 10,
    consult_method: str = "direct",
    accept: bool = True,
) -> tuple[TestClient, str]:
    """Drive a REAL hand-off up to the point a 直接相談 could have happened.

    The retrospective is only writable about the person who actually accepted, so
    every test here has to produce a genuine ``recommendations`` row — the seeded
    fixtures carry none (that is exactly why the first version of this endpoint
    could be pointed at any employee at all).
    """

    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": asker_id, "question": GOOD_Q, "session_id": session_id})
    _events(client, session_id)
    client.post(
        "/handoff/draft",
        json={"session_id": session_id, "draft": "本文", "consult_method": consult_method},
    )
    if accept:
        client.post("/answer", json={"session_id": session_id, "outcome": "accepted"})
        _events(client, session_id)
    return client, _question_id_for_session(engine, session_id)


def _retro_body(question_id: str, **over):
    body = {
        "question_id": question_id,
        "responder_id": "E001",
        "topics": ["ネットワーク・VPN"],
        "asked": "拠点間VPNが不安定な件",
        "answer_body": "MTU を下げると直る、という話でした",
        "resolution": "resolved",
    }
    body.update(over)
    return body


def test_topics_endpoint_serves_the_scorer_vocabulary(seed_counts, engine, fake_embedder) -> None:
    """The form must offer exactly what the scorer joins on — no frontend copy (#116)."""

    from tekijin.scorer.topics import TOPIC_VOCABULARY

    client = _client(engine, fake_embedder)
    resp = client.get("/topics", headers=_user_headers(33))
    assert resp.status_code == 200
    assert resp.json()["topics"] == list(TOPIC_VOCABULARY)


# --- the read side: what the form is built from ----------------------------- #
def test_retrospective_context_survives_the_acceptance(seed_counts, engine, fake_embedder) -> None:
    """The write-up happens AFTER the consultation, so the read must outlive it.

    The first version of #247 built the form from ``GET /handoff``, which 404s the
    moment the responder records an outcome — i.e. it was readable only during the
    window where the consultation had not happened yet.
    """

    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-ctx1")

    # The regression this endpoint exists for: the pending-hand-off view is gone.
    assert client.get("/handoff/retro-ctx1", headers=_user_headers(10)).status_code == 404

    resp = client.get("/consult-retrospective/retro-ctx1", headers=_user_headers(10))
    assert resp.status_code == 200
    body = resp.json()
    assert body["question_id"] == question_id
    assert body["consult_method"] == "direct"
    assert body["question"] == GOOD_Q
    assert body["responder"]["person_id"] == "E001"
    assert body["responder"]["name"]
    assert body["already_recorded"] is False


def test_retrospective_context_reports_a_chat_handoff_as_chat(
    seed_counts, engine, fake_embedder
) -> None:
    """A chat consultation already leaves a transcript; the form must not appear."""

    client, _ = _direct_consultation(
        engine, fake_embedder, session_id="retro-ctx2", consult_method="chat"
    )
    body = client.get("/consult-retrospective/retro-ctx2", headers=_user_headers(10)).json()
    assert body["consult_method"] == "chat"


def test_retrospective_context_reports_an_already_written_write_up(
    seed_counts, engine, fake_embedder
) -> None:
    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-ctx3")
    client.post("/consult-retrospective", json=_retro_body(question_id), headers=_user_headers(10))
    body = client.get("/consult-retrospective/retro-ctx3", headers=_user_headers(10)).json()
    assert body["already_recorded"] is True


def test_retrospective_context_from_a_stranger_is_403(seed_counts, engine, fake_embedder) -> None:
    client, _ = _direct_consultation(engine, fake_embedder, session_id="retro-ctx4")
    assert (
        client.get("/consult-retrospective/retro-ctx4", headers=_user_headers(44)).status_code
        == 403
    )


def test_retrospective_context_for_an_unknown_session_is_404(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(engine, fake_embedder)
    assert client.get("/consult-retrospective/nope", headers=_user_headers(33)).status_code == 404


# --- the write side --------------------------------------------------------- #
def test_retrospective_is_recorded_for_the_questions_own_asker(
    seed_counts, engine, fake_embedder
) -> None:
    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w1")
    resp = client.post(
        "/consult-retrospective", json=_retro_body(question_id), headers=_user_headers(10)
    )
    assert resp.status_code == 200 and resp.json()["status"] == "recorded"

    rows = _consult_rows(engine)
    assert len(rows) == 1
    row = rows[0]
    assert row.question_id == question_id
    assert row.responder_id == 1
    assert row.topics == ["ネットワーク・VPN"]
    assert row.resolution == "resolved"
    # asker_id comes from the TOKEN, never the body — the same rule as /feedback.
    assert row.asker_id == 10


def test_a_second_retrospective_for_the_same_question_is_409(
    seed_counts, engine, fake_embedder
) -> None:
    """One accepted hand-off = one consultation = one write-up.

    Without this the asker could write the SAME real conversation up
    ``OFFLINE_CONSULT_EVIDENCE_CAP`` times and reach the full 1.0 of evidence from
    it — the accepted-responder check alone does not bound how OFTEN they write.
    """

    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w14")
    first = client.post(
        "/consult-retrospective", json=_retro_body(question_id), headers=_user_headers(10)
    )
    assert first.status_code == 200
    second = client.post(
        "/consult-retrospective",
        json=_retro_body(question_id, answer_body="別の書き方でもう一度"),
        headers=_user_headers(10),
    )
    assert second.status_code == 409
    assert "すでに記録" in second.json()["detail"]
    assert len(_consult_rows(engine)) == 1


def test_the_database_itself_rejects_a_second_write_up(seed_counts, engine, fake_embedder) -> None:
    """The 409 above is the polite answer; this is the constraint behind it.

    Written against the table rather than the route so a future caller that skips
    the API check still cannot double-count one consultation.
    """

    from sqlalchemy.exc import IntegrityError

    from tekijin.models.tables import OfflineConsult

    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w15")
    assert (
        client.post(
            "/consult-retrospective", json=_retro_body(question_id), headers=_user_headers(10)
        ).status_code
        == 200
    )

    with pytest.raises(IntegrityError), session_scope(get_sessionmaker(engine)) as session:
        session.add(
            OfflineConsult(
                question_id=question_id,
                responder_id=1,
                asker_id=10,
                topics=["ネットワーク・VPN"],
                asked=None,
                answer_body="直接 INSERT した2件目",
                resolution="resolved",
            )
        )
        session.flush()
    assert len(_consult_rows(engine)) == 1


def test_the_asker_cannot_record_the_responders_acceptance(
    seed_counts, engine, fake_embedder
) -> None:
    """An outcome is the RESPONDER's act; the asker must not be able to forge it.

    The asker is a legitimate session participant (they own the clarification
    reply), so ``require_session_participant`` alone let them POST
    ``outcome="accepted"`` for their own question. That forges the one durable
    record of "this person took the hand-off" — which the dashboard's acceptance
    rate reads, the inbox is filtered by, and #247 uses to decide whom a
    retrospective may credit.
    """

    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "self-accept"})
    _events(client, "self-accept")
    client.post(
        "/handoff/draft",
        json={"session_id": "self-accept", "draft": "本文", "consult_method": "direct"},
    )

    forged = client.post(
        "/answer",
        json={"session_id": "self-accept", "outcome": "accepted"},
        headers=_user_headers(10),
    )
    assert forged.status_code == 403
    question_id = _question_id_for_session(engine, "self-accept")
    from sqlalchemy import select

    with session_scope(get_sessionmaker(engine)) as session:
        outcomes = (
            session.execute(
                select(Recommendation.outcome).where(Recommendation.question_id == question_id)
            )
            .scalars()
            .all()
        )
    assert all(o is None for o in outcomes)

    # ...and with no acceptance on record, there is nothing to write up.
    refused = client.post(
        "/consult-retrospective", json=_retro_body(question_id), headers=_user_headers(10)
    )
    assert refused.status_code == 422
    assert "まだ受諾されていない" in refused.json()["detail"]
    assert _consult_rows(engine) == []


def test_the_asker_can_still_decline_nothing_and_reply_to_a_clarification(
    seed_counts, engine, fake_embedder
) -> None:
    """The outcome restriction must not take the asker's own resume path with it."""

    client = _client(engine, fake_embedder)
    client.post("/ask", json={"asker_id": 10, "question": VAGUE_Q, "session_id": "asker-reply"})
    _events(client, "asker-reply")
    replied = client.post(
        "/answer",
        json={"session_id": "asker-reply", "reply": "UTMの移行です"},
        headers=_user_headers(10),
    )
    assert replied.status_code == 200


def test_deleting_a_question_takes_its_retrospective_with_it(
    seed_counts, engine, fake_embedder
) -> None:
    """``offline_consults`` FKs ``questions`` with no CASCADE — the same shape that
    made #207's delete 500 when ``feedback`` was forgotten."""

    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-del")
    assert (
        client.post(
            "/consult-retrospective", json=_retro_body(question_id), headers=_user_headers(10)
        ).status_code
        == 200
    )
    deleted = client.delete(f"/questions/{question_id}", headers=_user_headers(10))
    assert deleted.status_code == 200
    assert _consult_rows(engine) == []


def test_retrospective_from_a_stranger_is_403(seed_counts, engine, fake_embedder) -> None:
    # The retrospective becomes expertise evidence for the responder, so letting a
    # non-owner write one would be a way to inflate (or fabricate) someone's standing.
    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w2")
    resp = client.post(
        "/consult-retrospective", json=_retro_body(question_id), headers=_user_headers(44)
    )
    assert resp.status_code == 403
    assert _consult_rows(engine) == []


def test_retrospective_for_an_unknown_question_is_404(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    resp = client.post(
        "/consult-retrospective",
        json=_retro_body("q_does_not_exist"),
        headers=_user_headers(33),
    )
    assert resp.status_code == 404
    assert _consult_rows(engine) == []


def test_retrospective_cannot_name_yourself_as_the_responder(
    seed_counts, engine, fake_embedder
) -> None:
    """Self-attribution: the whole point of the row is that SOMEONE ELSE helped."""

    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w3")
    resp = client.post(
        "/consult-retrospective",
        json=_retro_body(question_id, responder_id="E010"),
        headers=_user_headers(10),
    )
    assert resp.status_code == 422
    # the guard that fired, not just "some 422"
    assert "自分自身を相談相手として" in resp.json()["detail"]
    assert _consult_rows(engine) == []


def test_retrospective_cannot_name_someone_never_recommended_for_the_question(
    seed_counts, engine, fake_embedder
) -> None:
    """Fabrication: naming an arbitrary colleague must not become their evidence."""

    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w4")
    resp = client.post(
        "/consult-retrospective",
        json=_retro_body(question_id, responder_id="E033"),
        headers=_user_headers(10),
    )
    assert resp.status_code == 422
    # the guard that fired, not just "some 422"
    assert "実際に相談を受けた相手" in resp.json()["detail"]
    assert _consult_rows(engine) == []


def test_retrospective_cannot_name_a_shown_but_unaccepted_candidate(
    seed_counts, engine, fake_embedder
) -> None:
    """Even a recommended person is not evidence unless they took the hand-off.

    Employees 1/2/3 are all shown for this question; only 1 accepted. Allowing any
    of the three would hand the asker a free choice of whom to credit.
    """

    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w5")
    resp = client.post(
        "/consult-retrospective",
        json=_retro_body(question_id, responder_id="E002"),
        headers=_user_headers(10),
    )
    assert resp.status_code == 422
    # the guard that fired, not just "some 422"
    assert "実際に相談を受けた相手" in resp.json()["detail"]
    assert _consult_rows(engine) == []


def test_retrospective_for_a_nonexistent_employee_is_422_not_500(
    seed_counts, engine, fake_embedder
) -> None:
    """A bad id must not reach the FK constraint and surface as a 500 (cf. #263)."""

    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w6")
    resp = client.post(
        "/consult-retrospective",
        json=_retro_body(question_id, responder_id="E999999"),
        headers=_user_headers(10),
    )
    assert resp.status_code == 422
    # the guard that fired, not just "some 422"
    assert "実際に相談を受けた相手" in resp.json()["detail"]
    assert _consult_rows(engine) == []


def test_retrospective_before_the_handoff_was_accepted_is_422(
    seed_counts, engine, fake_embedder
) -> None:
    """Nothing was consulted yet, so there is nothing to write up."""

    client, question_id = _direct_consultation(
        engine, fake_embedder, session_id="retro-w7", accept=False
    )
    resp = client.post(
        "/consult-retrospective", json=_retro_body(question_id), headers=_user_headers(10)
    )
    assert resp.status_code == 422
    # the guard that fired, not just "some 422"
    assert "まだ受諾されていない" in resp.json()["detail"]
    assert _consult_rows(engine) == []


def test_retrospective_on_a_chat_consultation_is_422(seed_counts, engine, fake_embedder) -> None:
    """The transcript already exists; a hearsay summary on top of it is a weaker copy."""

    client, question_id = _direct_consultation(
        engine, fake_embedder, session_id="retro-w8", consult_method="chat"
    )
    resp = client.post(
        "/consult-retrospective", json=_retro_body(question_id), headers=_user_headers(10)
    )
    assert resp.status_code == 422
    # the guard that fired, not just "some 422"
    assert "直接相談ではない" in resp.json()["detail"]
    assert _consult_rows(engine) == []


def test_retrospective_can_be_switched_off(seed_counts, engine, fake_embedder, monkeypatch) -> None:
    """Kill switch (#247): the only write path that mutates expertise from the UI."""

    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w9")
    monkeypatch.setenv("TEKIJIN_CONSULT_RETROSPECTIVE_ENABLED", "false")
    get_settings.cache_clear()
    try:
        resp = client.post(
            "/consult-retrospective", json=_retro_body(question_id), headers=_user_headers(10)
        )
        assert resp.status_code == 503
        assert _consult_rows(engine) == []
        # The read side stays up: an in-flight form must be able to explain itself.
        assert (
            client.get("/consult-retrospective/retro-w9", headers=_user_headers(10)).status_code
            == 200
        )
    finally:
        monkeypatch.delenv("TEKIJIN_CONSULT_RETROSPECTIVE_ENABLED", raising=False)
        get_settings.cache_clear()


def test_retrospective_rejects_a_topic_outside_the_vocabulary(
    seed_counts, engine, fake_embedder
) -> None:
    # The scorer joins on these strings; a free-text topic matches NO evidence and
    # would silently do nothing (#116). Reject at the boundary instead.
    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w10")
    resp = client.post(
        "/consult-retrospective",
        json=_retro_body(question_id, topics=["まったく新しいトピック"]),
        headers=_user_headers(10),
    )
    assert resp.status_code == 422
    assert _consult_rows(engine) == []


def test_retrospective_requires_topics_answer_and_resolution(
    seed_counts, engine, fake_embedder
) -> None:
    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w11")
    for over in ({"topics": []}, {"answer_body": "  "}, {"resolution": "まあまあ"}):
        resp = client.post(
            "/consult-retrospective",
            json=_retro_body(question_id, **over),
            headers=_user_headers(10),
        )
        assert resp.status_code == 422, over
    # `asked` is optional (#247 の項目2は任意).
    ok = client.post(
        "/consult-retrospective",
        json=_retro_body(question_id, asked=None),
        headers=_user_headers(10),
    )
    assert ok.status_code == 200


def _topic_score(engine, person_id: int, topic: str) -> float:
    from tekijin.data.repository import Repository
    from tekijin.scorer.scorer import ExpertiseScorer

    with session_scope(get_sessionmaker(engine)) as session:
        scorer = ExpertiseScorer(Repository(session))
        result = scorer.rank(
            topic, [person_id], asker_id=None, now=dt.datetime(2026, 8, 26), top_k=1
        )
        return result["recommendations"][0]["score"]


def test_retrospective_becomes_expertise_evidence_for_the_responder(
    seed_counts, engine, fake_embedder
) -> None:
    """The point of #247: the record must actually reach C6, not just sit in a table."""

    topic = "ネットワーク・VPN"
    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w12")
    before = _topic_score(engine, 1, topic)
    client.post("/consult-retrospective", json=_retro_body(question_id), headers=_user_headers(10))
    assert _topic_score(engine, 1, topic) > before


def test_unresolved_retrospective_is_stored_but_is_not_evidence(
    seed_counts, engine, fake_embedder
) -> None:
    """断り≠非専門 applied to consultations: recorded, but neither adds nor subtracts."""

    topic = "ネットワーク・VPN"
    client, question_id = _direct_consultation(engine, fake_embedder, session_id="retro-w13")
    before = _topic_score(engine, 1, topic)
    resp = client.post(
        "/consult-retrospective",
        json=_retro_body(question_id, resolution="unresolved"),
        headers=_user_headers(10),
    )
    assert resp.status_code == 200
    assert len(_consult_rows(engine)) == 1  # stored
    assert _topic_score(engine, 1, topic) == before  # but not evidence, in EITHER direction


def test_feedback_endpoint_records_with_actor_from_principal(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(engine, fake_embedder)
    resp = client.post(
        "/feedback",
        json={
            "stage": "c6",
            "kind": "person_rejected",
            "target": "E001",
            "payload": {"reason": "多忙"},
        },
        headers=_user_headers(10),
    )
    assert resp.status_code == 200 and resp.json() == {"status": "recorded"}
    rows = _feedback_rows(engine)
    assert len(rows) == 1
    row = rows[0]
    assert row.stage == "c6" and row.kind == "person_rejected" and row.target == "E001"
    assert row.payload == {"reason": "多忙"}
    # actor_id comes from the token (employee 10), NOT the body.
    assert row.actor_id == 10


def test_feedback_owner_may_tag_their_question_stranger_gets_403(
    seed_counts, engine, fake_embedder
) -> None:
    # q_0001 is owned by asker_id 33 (fixtures). Object-level auth (#263).
    client = _client(engine, fake_embedder)
    owner = client.post(
        "/feedback",
        json={"stage": "c1", "kind": "x", "question_id": "q_0001"},
        headers=_user_headers(33),
    )
    assert owner.status_code == 200
    # A non-owner cannot tag someone else's question (would pollute its signal).
    stranger = client.post(
        "/feedback",
        json={"stage": "c1", "kind": "x", "question_id": "q_0001"},
        headers=_user_headers(10),
    )
    assert stranger.status_code == 403
    # Only the owner's row was recorded, with the link kept.
    rows = _feedback_rows(engine)
    assert len(rows) == 1 and rows[0].actor_id == 33 and rows[0].question_id == "q_0001"


def test_feedback_unknown_question_id_is_dropped_not_403(
    seed_counts, engine, fake_embedder
) -> None:
    # An unknown id must not 403 (that would be an existence oracle) nor 500 (FK) —
    # the link is dropped and the signal still recorded (#263, preserves #237).
    client = _client(engine, fake_embedder)
    resp = client.post(
        "/feedback",
        json={"stage": "c1", "kind": "x", "question_id": "q_does_not_exist"},
        headers=_user_headers(10),
    )
    assert resp.status_code == 200
    rows = _feedback_rows(engine)
    assert len(rows) == 1 and rows[0].question_id is None


def test_feedback_non_participant_cannot_tag_a_live_session(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "fbauth"})
    _events(client, "fbauth")  # live session, asker=10 / responder=E001
    # A stranger (not asker/responder) cannot tag the live session.
    stranger = client.post(
        "/feedback",
        json={"stage": "c6", "kind": "x", "session_id": "fbauth"},
        headers=_user_headers(999),
    )
    assert stranger.status_code == 403
    # The asker can.
    ok = client.post(
        "/feedback",
        json={"stage": "c6", "kind": "x", "session_id": "fbauth"},
        headers=_user_headers(10),
    )
    assert ok.status_code == 200


def test_feedback_is_rate_limited_per_actor(seed_counts, engine, fake_embedder) -> None:
    from tekijin.api.rate_limit import SlidingWindowLimiter

    client = _client(engine, fake_embedder)
    # Shrink the limiter so the flood guard trips deterministically.
    client.app.state.feedback_rate_limiter = SlidingWindowLimiter(max_events=2, window_seconds=60.0)
    body = {"stage": "c1", "kind": "x"}
    assert client.post("/feedback", json=body, headers=_user_headers(10)).status_code == 200
    assert client.post("/feedback", json=body, headers=_user_headers(10)).status_code == 200
    # Third within the window is refused.
    assert client.post("/feedback", json=body, headers=_user_headers(10)).status_code == 429
    # A DIFFERENT actor is tracked independently (not penalised by 10's flood).
    assert client.post("/feedback", json=body, headers=_user_headers(11)).status_code == 200


def test_feedback_endpoint_rejects_an_unknown_stage(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    resp = client.post("/feedback", json={"stage": "c9", "kind": "x"}, headers=_user_headers(10))
    assert resp.status_code == 422
    assert _feedback_rows(engine) == []


def test_editing_the_draft_records_implicit_c7_feedback(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine,
        fake_embedder,
        retriever=_FakeRetriever(people=[1, 2], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2)),
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "fb1"})
    _events(client, "fb1")  # pause at send; C7 draft ready
    generated = client.get("/handoff/fb1").json()["draft"]

    edited = f"{generated}（追記: 10月中にお願いできると助かります）"
    saved = client.post("/handoff/draft", json={"session_id": "fb1", "draft": edited})
    assert saved.status_code == 200

    rows = _feedback_rows(engine)
    assert len(rows) == 1
    row = rows[0]
    assert row.stage == "c7" and row.kind == "draft_edited"
    assert row.session_id == "fb1"
    assert row.payload["generated"] == generated and row.payload["sent"] == edited


def test_saving_the_draft_unchanged_records_no_feedback(seed_counts, engine, fake_embedder) -> None:
    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "fb2"})
    _events(client, "fb2")
    generated = client.get("/handoff/fb2").json()["draft"]

    # Re-saving the identical generated text is not a correction -> no feedback.
    saved = client.post("/handoff/draft", json={"session_id": "fb2", "draft": generated})
    assert saved.status_code == 200
    assert _feedback_rows(engine) == []


def test_dashboard_reports_feedback_by_stage(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    for stage in ("c1", "c6", "c6"):
        client.post("/feedback", json={"stage": stage, "kind": "x"}, headers=_user_headers(10))
    data = client.get("/dashboard").json()
    assert data["feedback_by_stage"] == {"c1": 1, "c6": 2, "c7": 0, "total": 3}


def test_draft_feedback_write_failure_does_not_break_the_save(
    seed_counts, engine, fake_embedder, monkeypatch
) -> None:
    """A feedback-write failure must never surface as a 500: the draft edit is
    already committed by then (#237 review HIGH)."""
    import tekijin.api.service as svc

    client = _client(
        engine, fake_embedder, retriever=_FakeRetriever(people=[1]), scorer=_FakeScorer(_recs(1))
    )
    client.post("/ask", json={"asker_id": 10, "question": GOOD_Q, "session_id": "fb3"})
    _events(client, "fb3")
    generated = client.get("/handoff/fb3").json()["draft"]

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("feedback backend is down")

    monkeypatch.setattr(svc, "record_feedback", _boom)

    edited = f"{generated}（追記: 期日を早めたいです）"
    saved = client.post("/handoff/draft", json={"session_id": "fb3", "draft": edited})
    # The save still succeeds despite the feedback write blowing up...
    assert saved.status_code == 200 and saved.json()["status"] == "draft_saved"
    # ...and the edit is durably persisted...
    assert client.get("/handoff/fb3").json()["draft"] == edited
    # ...while no feedback row was recorded (the write failed and was swallowed).
    assert _feedback_rows(engine) == []


def test_feedback_rejects_an_oversized_payload(seed_counts, engine, fake_embedder) -> None:
    """An authenticated caller must not be able to grow the JSONB column unbounded
    (storage-exhaustion DoS, #237 security review)."""
    client = _client(engine, fake_embedder)
    big = {"x": "あ" * 20000}  # ~60KB UTF-8, over the 16KB cap
    resp = client.post(
        "/feedback", json={"stage": "c1", "kind": "x", "payload": big}, headers=_user_headers(10)
    )
    assert resp.status_code == 422
    assert _feedback_rows(engine) == []


def test_feedback_with_unknown_question_id_records_without_the_link(
    seed_counts, engine, fake_embedder
) -> None:
    """An unknown question_id must not 500 (FK violation) nor leak existence — it is
    recorded without the dangling link (#237 security review)."""
    client = _client(engine, fake_embedder)
    resp = client.post(
        "/feedback",
        json={"stage": "c6", "kind": "x", "question_id": "api_does_not_exist"},
        headers=_user_headers(10),
    )
    assert resp.status_code == 200
    rows = _feedback_rows(engine)
    assert len(rows) == 1 and rows[0].question_id is None


# --------------------------------------------------------------------------- #
# #293, #301: GET /knowledge — company-wide accumulated knowledge, answers AND
# documents (NOT scoped to one asker, and NOT admin-only, unlike /dashboard).
# source_id/kind match a self-answer's citation (#291) for the same entity.
# --------------------------------------------------------------------------- #
def _approved_ku_count(engine) -> int:
    """Approved knowledge units count toward the library's global total (#533). Queried
    live so the total stays correct even when another test leaves approved units in the
    shared schema (the seed itself has none)."""
    from sqlalchemy import func, select

    from tekijin.models.tables import KnowledgeUnit

    with engine.connect() as conn:
        return (
            conn.scalar(
                select(func.count())
                .select_from(KnowledgeUnit)
                .where(KnowledgeUnit.review_status == "approved")
            )
            or 0
        )


def test_knowledge_lists_seeded_answers_and_documents(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    resp = client.get("/knowledge")
    assert resp.status_code == 200
    body = resp.json()

    # The seed's answers are 1:1 with questions (test_dashboard_route_shape_
    # and_seed_values), so the total is global (both kinds), independent of
    # the default response cap.
    total = seed_counts["answers"] + seed_counts["documents"] + _approved_ku_count(engine)
    assert body["summary"]["total_items"] == total
    assert body["summary"]["self_resolution_rate"] == 0.0  # matches the dashboard's seed baseline
    assert "top_responders" not in body["summary"]  # de-scoped to /dashboard (PR #340 review)
    assert len(body["items"]) <= 8  # the endpoint's default limit
    assert len(body["items"]) > 0
    assert body["total_matching"] == total  # unfiltered: every item matches
    for item in body["items"]:
        assert item["kind"] in ("qa", "document", "knowledge")
        assert item["summary"]  # the item's own text, not just metadata


def test_knowledge_default_page_mixes_both_kinds(seed_counts, engine, fake_embedder) -> None:
    """The seed's documents are all older than its newest answers, so a plain
    recency sort buried every document past the unsearched view's one page —
    no document ever reached the front page without a keyword search (PR #340
    review follow-up). Round-robin interleaving fixes that: the default
    (unsearched, unpaginated) browse page must show at least one of each kind
    whenever both exist."""
    client = _client(engine, fake_embedder)
    kinds = {i["kind"] for i in client.get("/knowledge").json()["items"]}
    # Both seed kinds must reach the front page; a leaked approved knowledge unit (#533)
    # from another test may add "knowledge", so assert the subset, not exact equality.
    assert {"qa", "document"} <= kinds


def test_knowledge_qa_item_shape(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    items = client.get("/knowledge", params={"limit": 200}).json()["items"]
    qa = next(i for i in items if i["kind"] == "qa")
    assert qa["responder_name"]
    assert qa["question_id"]
    assert qa["session_id"] is None  # seeded history has no live session


def test_knowledge_document_item_shape(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    items = client.get("/knowledge", params={"limit": 200}).json()["items"]
    doc = next(i for i in items if i["kind"] == "document")
    # Document-only items carry none of the QA-specific fields.
    assert doc["responder_name"] is None
    assert doc["responder_department"] is None
    assert doc["question_id"] is None
    assert doc["session_id"] is None
    assert doc["topics"] == []


def test_knowledge_excludes_a_question_with_no_formal_answer(
    seed_counts, engine, fake_embedder
) -> None:
    """A live-chat-accepted question with no ``answers`` row has no answer TEXT
    to show as "回答のまとめ", so it must not appear (#293, #301 review) — unlike
    the earlier "resolved by a person" definition, which also counted an
    accepted recommendation alone."""
    factory = get_sessionmaker(engine)
    with factory() as s:
        s.add(
            Question(
                id="api_kn1",
                asker_id=10,
                body="チャットのみでやり取りして解決した質問",
                topics=[],
                status="open",
                created_at=NOW,
                session_id="sess-kn1",
            )
        )
        s.flush()
        s.add(
            Recommendation(
                question_id="api_kn1", employee_id=1, rank=1, score=0.9, outcome="accepted"
            )
        )
        s.commit()

    client = _client(engine, fake_embedder)
    resp = client.get("/knowledge", params={"limit": 200})
    assert resp.status_code == 200
    ids = {i["question_id"] for i in resp.json()["items"]}
    assert "api_kn1" not in ids


def test_knowledge_available_to_non_admin_user(seed_counts, engine, fake_embedder) -> None:
    # Unlike /dashboard (admin-only), every authenticated user can browse
    # knowledge — the whole point is discovering someone ELSE'S past answer.
    client = _client(engine, fake_embedder)
    resp = client.get("/knowledge", headers=_user_headers(10))
    assert resp.status_code == 200
    total = seed_counts["answers"] + seed_counts["documents"] + _approved_ku_count(engine)
    assert resp.json()["summary"]["total_items"] == total


def test_knowledge_respects_limit(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    resp = client.get("/knowledge", params={"limit": 1})
    body = resp.json()
    assert len(body["items"]) == 1
    # The summary total stays global — it does not shrink with the page limit.
    total = seed_counts["answers"] + seed_counts["documents"] + _approved_ku_count(engine)
    assert body["summary"]["total_items"] == total
    assert body["total_matching"] == total  # unfiltered: still every item


def test_knowledge_offset_pages_through_results_without_overlap(
    seed_counts, engine, fake_embedder
) -> None:
    client = _client(engine, fake_embedder)
    page1 = client.get("/knowledge", params={"limit": 15, "offset": 0}).json()
    page2 = client.get("/knowledge", params={"limit": 15, "offset": 15}).json()

    assert len(page1["items"]) == 15
    assert len(page2["items"]) == 15
    ids1 = {i["source_id"] for i in page1["items"]}
    ids2 = {i["source_id"] for i in page2["items"]}
    assert ids1.isdisjoint(ids2)  # no overlap between pages
    # total_matching is the same on both pages (the count BEFORE paging).
    total = seed_counts["answers"] + seed_counts["documents"] + _approved_ku_count(engine)
    assert page1["total_matching"] == page2["total_matching"] == total

    empty = client.get("/knowledge", params={"limit": 15, "offset": total}).json()
    assert empty["items"] == []
    assert empty["total_matching"] == total


def test_knowledge_search_filters_by_keyword(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    baseline = client.get("/knowledge", params={"limit": 200}).json()["items"]
    target = baseline[0]
    keyword = target["title"][:8]  # a substring guaranteed to be in at least one title

    resp = client.get("/knowledge", params={"q": keyword, "limit": 200})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any(i["source_id"] == target["source_id"] for i in items)
    assert all(keyword in i["title"] for i in items)
    assert len(items) <= len(baseline)


def test_knowledge_filters_by_topic(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    baseline = client.get("/knowledge", params={"limit": 200}).json()["items"]
    with_topic = next((i for i in baseline if i["topics"]), None)
    assert with_topic is not None, "seed fixtures are expected to carry topics"
    topic = with_topic["topics"][0]

    resp = client.get("/knowledge", params={"topic": topic, "limit": 200})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any(i["source_id"] == with_topic["source_id"] for i in items)
    assert all(topic in i["topics"] for i in items)
    # topic is QA-specific: documents (which carry no topics) are excluded.
    assert all(i["kind"] == "qa" for i in items)


def test_knowledge_filters_by_department(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    baseline = client.get("/knowledge", params={"limit": 200}).json()["items"]
    with_dept = next((i for i in baseline if i["responder_department"]), None)
    assert with_dept is not None, "seed fixtures are expected to carry a department"
    dept = with_dept["responder_department"]

    resp = client.get("/knowledge", params={"department": dept, "limit": 200})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any(i["source_id"] == with_dept["source_id"] for i in items)
    assert all(i["responder_department"] == dept for i in items)
    # department is QA-specific: documents (which carry no department) are excluded.
    assert all(i["kind"] == "qa" for i in items)


def test_knowledge_since_includes_items_from_the_start_day_itself(
    seed_counts, engine, fake_embedder
) -> None:
    """`since` is the screen's only period filter and had no test at all (#394).

    The boundary is the whole point: `since` is bound against a TIMESTAMP column,
    so a bare date becomes that day's 00:00 — inclusive for a start bound. An
    item answered at 05:24 on the start day must still be listed.
    """
    client = _client(engine, fake_embedder)
    baseline = client.get("/knowledge", params={"limit": 200}).json()["items"]
    newest = max(baseline, key=lambda i: i["resolved_at"])
    day = newest["resolved_at"][:10]

    resp = client.get("/knowledge", params={"since": day, "limit": 200})
    assert resp.status_code == 200
    items = resp.json()["items"]
    assert any(i["source_id"] == newest["source_id"] for i in items)
    assert all(i["resolved_at"][:10] >= day for i in items)
    assert len(items) < len(baseline)  # the filter actually narrows the set


def test_knowledge_has_no_end_date_filter(seed_counts, engine, fake_embedder) -> None:
    """`until` was removed (#394), and re-adding it needs UI + a boundary test.

    It existed on the endpoint, `list_knowledge` and `api-client`, but no screen
    ever sent it (KnowledgeScreen exposes 「この日以降」 only, by request) and
    nothing tested it — and it was WRONG: bound against a TIMESTAMP column, a bare
    date means that day's 00:00, so `until=<day>` silently dropped every item from
    the end day (measured: an answer at 2026-08-21 05:24 was excluded by
    `until=2026-08-21`). An inclusive-sounding 「この日まで」 that loses the last
    day is worse than no filter.

    This pins the removal at the HTTP boundary: an unknown query param is ignored
    by FastAPI, so `until` must not narrow anything.
    """
    client = _client(engine, fake_embedder)
    baseline = client.get("/knowledge", params={"limit": 200}).json()
    filtered = client.get("/knowledge", params={"until": "2000-01-01", "limit": 200}).json()
    assert filtered["total_matching"] == baseline["total_matching"]
    assert [i["source_id"] for i in filtered["items"]] == [
        i["source_id"] for i in baseline["items"]
    ]

    params = signature(knowledge_route).parameters
    assert "until" not in params
    assert "since" in params  # the surviving half of the period filter


def test_knowledge_detail_returns_full_qa_item(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    qa = next(
        i
        for i in client.get("/knowledge", params={"limit": 200}).json()["items"]
        if i["kind"] == "qa"
    )

    resp = client.get(f"/knowledge/{qa['source_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source_id"] == qa["source_id"]
    assert body["kind"] == "qa"
    assert body["title"] == qa["title"]
    assert body["summary"] == qa["summary"]
    assert body["responder_name"] == qa["responder_name"]


def test_knowledge_detail_404_for_unknown_id(seed_counts, engine, fake_embedder) -> None:
    client = _client(engine, fake_embedder)
    resp = client.get("/knowledge/does-not-exist")
    assert resp.status_code == 404

"""Sending a hand-off from Discord and answering it in a thread (#114).

Drives the internal endpoints the bot calls, against the real graph and database:
``/ask`` keeps the hand-off only while Discord output is on; 「依頼する」 queues the
request thread; the thread's buttons work only for the requested member; what they
write becomes an answer and a case draft; ``/tekijin review`` is admin-only.
"""

from __future__ import annotations

import itertools
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from tekijin.app import create_app
from tekijin.config import get_settings
from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import articles_from_question
from tekijin.models.tables import (
    Answer,
    DiscordOutbox,
    DiscordRequest,
    Identity,
    Message,
    Organization,
    Question,
    Recommendation,
)
from test_api_integration import (  # noqa: F401 - autouse cleanup of committed api_ rows
    GOOD_Q,
    _cleanup_api_rows,
)
from test_discord_routes_integration import (  # noqa: F401 - autouse member cleanup
    GUILD,
    HEADERS,
    SECRET,
    _bot_service,
    _cleanup_discord_members,
)

OUTPUT = "5555"
ASKER = "1001"
RESPONDER = "2001"  # employee 1, the top recommendation
SECOND = "2002"  # employee 2
ADMIN = "9001"
BASE = "/internal/discord"


@pytest.fixture(autouse=True)
def _cleanup_answers_and_knowledge(engine, _cleanup_discord_members):  # noqa: F811
    """Torn down before the member cleanup, which deletes the questions."""

    yield
    session = get_sessionmaker(engine)()
    try:
        members = "SELECT id FROM employees WHERE org_id IS NOT NULL"
        session.execute(
            text(
                f"DELETE FROM answers WHERE question_id IN "
                f"(SELECT id FROM questions WHERE asker_id IN ({members}))"
            )
        )
        session.execute(text("DELETE FROM knowledge_sources"))
        session.execute(text("DELETE FROM knowledge_articles"))
        session.execute(text("DELETE FROM discord_outbox"))
        session.execute(text("DELETE FROM discord_requests"))
        session.commit()
    finally:
        session.close()


def _env(monkeypatch, *, output: bool) -> None:
    monkeypatch.setenv("TEKIJIN_DISCORD_BOT_SECRET", SECRET)
    monkeypatch.setenv("TEKIJIN_DISCORD_GUILD_ID", GUILD)
    monkeypatch.setenv("TEKIJIN_DISCORD_ADMIN_USER_IDS", f" {ADMIN} ,")
    if output:
        monkeypatch.setenv("TEKIJIN_DISCORD_OUTPUT_CHANNEL_ID", OUTPUT)
    else:
        monkeypatch.delenv("TEKIJIN_DISCORD_OUTPUT_CHANNEL_ID", raising=False)
    get_settings.cache_clear()


@pytest.fixture
def output_on(monkeypatch):
    _env(monkeypatch, output=True)
    yield
    get_settings.cache_clear()


@pytest.fixture
def output_off(monkeypatch):
    _env(monkeypatch, output=False)
    yield
    get_settings.cache_clear()


def _snowflake(n: int, *, at: float | None = None) -> str:
    """A Discord message id written at ``at`` (default: now), made unique by ``n``."""

    ms = int((time.time() if at is None else at) * 1000) - 1420070400000
    return str((ms << 22) + n)


def _user(discord_id: str) -> dict:
    return {"discord_user_id": discord_id, "guild_id": GUILD, "display_name": f"u{discord_id}"}


class _Flow:
    def __init__(self, engine, embedder) -> None:
        self.engine = engine
        self.service = _bot_service(engine, embedder)
        self.client = TestClient(create_app(agent_service=self.service))
        self._message_ids = itertools.count(1000)

    def post(self, path: str, body: dict) -> dict:
        resp = self.client.post(f"{BASE}{path}", json=body, headers=HEADERS)
        assert resp.status_code == 200, resp.text
        return resp.json()

    def ask(self) -> dict:
        return self.post("/ask", {**_user(ASKER), "question": GOOD_Q})

    def link(self, discord_id: str, employee_id: int) -> None:
        session = get_sessionmaker(self.engine)()
        try:
            org = session.scalars(
                select(Organization).where(Organization.discord_guild_id == GUILD)
            ).one()
            session.add(
                Identity(
                    org_id=org.id,
                    employee_id=employee_id,
                    provider="discord",
                    external_id=discord_id,
                )
            )
            session.commit()
        finally:
            session.close()

    def scalar(self, stmt):
        session = get_sessionmaker(self.engine)()
        try:
            return session.scalar(stmt)
        finally:
            session.close()

    def scalars(self, stmt) -> list:
        session = get_sessionmaker(self.engine)()
        try:
            return list(session.scalars(stmt))
        finally:
            session.close()

    def outbox(self) -> list[dict]:
        resp = self.client.get(f"{BASE}/outbox", headers=HEADERS)
        assert resp.status_code == 200, resp.text
        return resp.json()["items"]

    def deliver_requests(self) -> dict[int, str]:
        """Pretend the bot created a thread for every queued request."""

        threads = {}
        for item in self.outbox():
            if item["kind"] != "request":
                continue
            rec = item["payload"]["recommendation_id"]
            thread = str(800000 + rec)
            out = self.post(
                f"/outbox/{item['id']}", {"ok": True, "channel_id": OUTPUT, "thread_id": thread}
            )
            assert out["status"] == "ok"
            threads[rec] = thread
        return threads

    def requested(self) -> tuple[str, int, str]:
        """Ask, link the responders, press 依頼する and deliver the thread."""

        out = self.ask()
        assert out["kind"] == "handoff" and out["session_id"]
        self.link(RESPONDER, 1)
        self.link(SECOND, 2)
        sent = self.post("/request", {**_user(ASKER), "session_id": out["session_id"]})
        assert sent["status"] == "requested"
        ((rec, thread),) = self.deliver_requests().items()
        return out["session_id"], rec, thread

    def act(self, discord_id: str, rec: int, action: str, **extra) -> dict:
        return self.post(
            "/request/action",
            {**_user(discord_id), "recommendation_id": rec, "action": action, **extra},
        )

    def write(self, discord_id: str, thread: str, body: str) -> dict:
        return self.post(
            "/thread-message",
            {
                **_user(discord_id),
                "thread_id": thread,
                "message_id": _snowflake(next(self._message_ids)),
                "body": body,
            },
        )


@pytest.fixture
def flow(engine, fake_embedder, seed_counts) -> _Flow:
    return _Flow(engine, fake_embedder)


def test_without_an_output_channel_the_hand_off_is_not_kept(output_off, flow) -> None:
    out = flow.ask()
    assert out["kind"] == "handoff" and out["session_id"] is None
    assert not [s for s in flow.service._registry if s.startswith("discord_")]
    # Nothing can be sent and nothing is queued.
    sent = flow.post("/request", {**_user(ASKER), "session_id": "discord_" + "0" * 32})
    assert sent["status"] == "disabled"
    assert flow.client.get(f"{BASE}/outbox", headers=HEADERS).status_code == 404


def test_only_the_asker_can_send_and_only_once(output_on, flow) -> None:
    out = flow.ask()
    session_id = out["session_id"]
    assert session_id and session_id.startswith("discord_")
    flow.link(RESPONDER, 1)

    assert flow.post("/request", {**_user(RESPONDER), "session_id": session_id})["status"] == (
        "forbidden"
    )
    assert flow.post("/request", {**_user("7777"), "session_id": session_id})["status"] == (
        "forbidden"
    )
    sent = flow.post("/request", {**_user(ASKER), "session_id": session_id})
    assert sent["status"] == "requested" and sent["responder_name"]
    again = flow.post("/request", {**_user(ASKER), "session_id": session_id})
    assert again["status"] == "already"

    (item,) = flow.outbox()
    assert item["kind"] == "request"
    assert item["payload"]["responder_discord_id"] == RESPONDER
    assert item["payload"]["question"] == GOOD_Q
    assert flow.scalar(
        select(Question.consult_method).where(Question.session_id == session_id)
    ) == ("chat")


def test_a_thread_outside_the_output_channel_is_refused(output_on, flow) -> None:
    out = flow.ask()
    flow.post("/request", {**_user(ASKER), "session_id": out["session_id"]})
    (item,) = flow.outbox()
    refused = flow.post(f"/outbox/{item['id']}", {"ok": True, "channel_id": "1", "thread_id": "9"})
    assert refused["status"] == "refused"
    assert flow.scalar(select(DiscordRequest.thread_id)) is None


def test_only_the_requested_member_can_accept(output_on, flow) -> None:
    _, rec, thread = flow.requested()
    assert flow.act(ASKER, rec, "accept")["status"] == "forbidden"
    assert flow.act(SECOND, rec, "accept")["status"] == "forbidden"
    assert flow.act("7777", rec, "accept")["status"] == "forbidden"
    assert flow.act(RESPONDER, 999999, "accept")["status"] == "gone"

    assert flow.act(RESPONDER, rec, "accept")["status"] == "accepted"
    assert flow.scalar(select(Recommendation.outcome).where(Recommendation.id == rec)) == (
        "accepted"
    )
    posts = [i for i in flow.outbox() if i["kind"] == "post"]
    assert [p["thread_id"] for p in posts] == [thread]
    assert "依頼を受けました" in posts[0]["payload"]["text"]
    # Pressed again: the hand-off is over.
    assert flow.act(RESPONDER, rec, "accept")["status"] == "gone"


def test_the_answer_in_the_thread_becomes_an_answer_and_a_draft(output_on, flow) -> None:
    _, rec, thread = flow.requested()
    assert flow.write(RESPONDER, thread, "まだ承諾していない")["status"] == "not_accepted"
    flow.act(RESPONDER, rec, "accept")

    assert flow.write("7777", thread, "通りすがり")["status"] == "ignored"
    assert flow.write(SECOND, thread, "関係ない人")["status"] == "ignored"
    assert flow.write(RESPONDER, "123", "別のスレッド")["status"] == "ignored"
    assert flow.write(ASKER, thread, "よろしくお願いします")["status"] == "stored"
    assert flow.write(RESPONDER, thread, "拠点ごとにIPsecを張ってください")["status"] == "answered"

    question_id = flow.scalar(select(Recommendation.question_id).where(Recommendation.id == rec))
    assert flow.scalars(select(Answer.body).where(Answer.question_id == question_id)) == [
        "拠点ごとにIPsecを張ってください"
    ]
    bodies = flow.scalars(select(Message.body).where(Message.recommendation_id == rec))
    assert "よろしくお願いします" in bodies and "拠点ごとにIPsecを張ってください" in bodies
    session = get_sessionmaker(flow.engine)()
    try:
        (draft,) = articles_from_question(session, question_id)
    finally:
        session.close()
    assert (draft.kind, draft.state) == ("case", "draft")
    assert draft.resolution == "拠点ごとにIPsecを張ってください"


def test_a_web_message_is_copied_into_the_thread(output_on, flow) -> None:
    from tekijin.channels.discord_notifier import DiscordNotifier

    _, rec, thread = flow.requested()
    flow.act(RESPONDER, rec, "accept")
    DiscordNotifier().relay_message(
        flow.service.session_factory,
        employee_a=0,
        employee_b=1,
        sender_name="質問者",
        body="Webから追記",
        thread_id=rec,
    )
    texts = [i["payload"]["text"] for i in flow.outbox() if i["kind"] == "post"]
    assert any("Webから追記" in t for t in texts)


def test_a_decline_passes_the_request_on(output_on, flow) -> None:
    _, rec, thread = flow.requested()
    out = flow.act(RESPONDER, rec, "decline")
    assert out["status"] == "declined" and out["next_name"]
    assert flow.scalar(select(Recommendation.outcome).where(Recommendation.id == rec)) == (
        "declined"
    )
    new = flow.deliver_requests()
    (next_rec,) = new
    assert next_rec != rec
    assert flow.scalar(select(Recommendation.employee_id).where(Recommendation.id == next_rec)) == 2
    old_posts = [i["payload"]["text"] for i in flow.outbox() if i["thread_id"] == thread]
    assert any("辞退" in t for t in old_posts) and any("次の候補" in t for t in old_posts)
    # The old request's buttons no longer act.
    assert flow.act(RESPONDER, rec, "accept")["status"] == "stale"
    assert flow.act(SECOND, next_rec, "accept")["status"] == "accepted"


def test_refer_needs_someone_tekijin_knows(output_on, flow) -> None:
    _, rec, _ = flow.requested()
    assert flow.act(RESPONDER, rec, "refer")["status"] == "refer_unknown"
    assert flow.act(RESPONDER, rec, "refer", referred_discord_user_id="7777")["status"] == (
        "refer_unknown"
    )
    assert flow.act(RESPONDER, rec, "refer", referred_discord_user_id=ASKER)["status"] == (
        "refer_invalid"
    )
    assert flow.act(SECOND, rec, "refer", referred_discord_user_id=SECOND)["status"] == (
        "forbidden"
    )
    out = flow.act(RESPONDER, rec, "refer", referred_discord_user_id=SECOND)
    assert out["status"] == "referred" and out["next_name"]
    assert flow.scalar(select(Recommendation.outcome).where(Recommendation.id == rec)) == (
        "referred"
    )


def test_review_is_for_admins_only(output_on, flow) -> None:
    _, rec, thread = flow.requested()
    flow.act(RESPONDER, rec, "accept")
    flow.write(RESPONDER, thread, "拠点ごとにIPsecを張ってください")

    assert flow.post("/review", _user(RESPONDER))["status"] == "forbidden"
    listed = flow.post("/review", _user(ADMIN))
    assert listed["status"] == "ok" and listed["total"] == 1
    (article,) = listed["articles"]

    action = {"article_id": article["id"]}
    assert (
        flow.post("/review/action", {**_user(RESPONDER), **action, "action": "verify"})["status"]
        == "forbidden"
    )
    assert (
        flow.post("/review/action", {**_user(ADMIN), "article_id": 999999, "action": "verify"})[
            "status"
        ]
        == "gone"
    )
    sent_back = flow.post("/review/action", {**_user(ADMIN), **action, "action": "send_back"})
    assert sent_back["status"] == "sent_back"
    verified = flow.post("/review/action", {**_user(ADMIN), **action, "action": "verify"})
    assert verified["status"] == "verified"
    assert flow.post("/review", _user(ADMIN))["total"] == 0


def test_review_refuses_another_server(output_on, flow) -> None:
    other = {**_user(ADMIN), "guild_id": "1"}
    assert flow.post("/review", other)["status"] == "unlinked"


def test_outbox_counts_failures(output_on, flow) -> None:
    out = flow.ask()
    flow.post("/request", {**_user(ASKER), "session_id": out["session_id"]})
    (item,) = flow.outbox()
    assert flow.post(f"/outbox/{item['id']}", {"ok": False})["status"] == "ok"
    assert flow.scalar(select(DiscordOutbox.attempts)) == 1


def test_an_unsent_hand_off_is_not_in_the_inbox(output_on, flow) -> None:
    from tekijin.data.inbox import pending_handoffs_for_responder
    from tekijin.data.notifications import pending_request_notifications_for_responder

    def pending() -> tuple[list, list]:
        session = get_sessionmaker(flow.engine)()
        try:
            inbox = [
                p["session_id"]
                for p in pending_handoffs_for_responder(session, 1)
                if p["session_id"].startswith("discord_")
            ]
            bell = [
                n
                for n in pending_request_notifications_for_responder(session, 1)
                if str(n.get("session_id", "")).startswith("discord_")
            ]
            return inbox, bell
        finally:
            session.close()

    out = flow.ask()
    assert pending() == ([], [])
    flow.post("/request", {**_user(ASKER), "session_id": out["session_id"]})
    inbox, bell = pending()
    assert inbox == [out["session_id"]] and len(bell) == 1


def test_a_thread_message_is_handled_once(output_on, flow, monkeypatch) -> None:
    scheduled = []
    monkeypatch.setattr(flow.service, "schedule_knowledge_embed", lambda: scheduled.append(1))
    _, rec, thread = flow.requested()

    def write(message_id: str, body: str, **extra) -> str:
        return flow.post(
            "/thread-message",
            {
                **_user(RESPONDER),
                "thread_id": thread,
                "message_id": message_id,
                "body": body,
                **extra,
            },
        )["status"]

    before, after = _snowflake(10), None
    # Written before accepting: not kept, and not kept when sent again later.
    assert write(before, "受ける前") == "not_accepted"
    # Written before accepting but first sent afterwards (the bot was away).
    unsent = _snowflake(11)
    flow.act(RESPONDER, rec, "accept")
    assert write(before, "受ける前") == "duplicate"
    assert write(unsent, "受ける前その2") == "not_accepted"
    time.sleep(0.01)
    after = _snowflake(12)
    assert write(after, "答え") == "answered"
    assert write(after, "答え") == "duplicate"
    question_id = flow.scalar(select(Recommendation.question_id).where(Recommendation.id == rec))
    assert flow.scalars(select(Answer.body).where(Answer.question_id == question_id)) == ["答え"]
    # The draft is queued for embedding so it can be matched and rise in review.
    assert scheduled == [1]

    # An edit corrects the kept answer and the draft.
    assert write(after, "正しい答え", edited=True) == "updated"
    assert flow.scalars(select(Answer.body).where(Answer.question_id == question_id)) == [
        "正しい答え"
    ]
    assert "正しい答え" in flow.scalars(
        select(Message.body).where(Message.recommendation_id == rec)
    )
    session = get_sessionmaker(flow.engine)()
    try:
        assert articles_from_question(session, question_id)[0].resolution == "正しい答え"
    finally:
        session.close()
    # An edit of an ignored message is still ignored.
    assert write(before, "受ける前(編集)", edited=True) == "duplicate"

    # Deleting the answer in Discord removes it, and the draft loses it.
    other = _snowflake(13)
    assert write(other, "補足") == "answered"
    out = flow.post("/thread-message/delete", {"guild_id": GUILD, "message_ids": [after]})
    assert out == {"status": "ok", "removed": 1}
    assert flow.scalars(select(Answer.body).where(Answer.question_id == question_id)) == ["補足"]
    assert "正しい答え" not in flow.scalars(
        select(Message.body).where(Message.recommendation_id == rec)
    )
    session = get_sessionmaker(flow.engine)()
    try:
        assert articles_from_question(session, question_id)[0].resolution == "補足"
    finally:
        session.close()
    # Never handled again, even if sent once more.
    assert write(after, "正しい答え") == "duplicate"
    # Deleted before it ever reached the backend: refused when it arrives late.
    lost = _snowflake(14)
    flow.post(
        "/thread-message/delete", {"guild_id": GUILD, "message_ids": [lost], "thread_id": thread}
    )
    assert write(lost, "消した助言") == "duplicate"
    assert (
        flow.post("/thread-message/delete", {"guild_id": "1", "message_ids": [other]})["status"]
        == "unlinked"
    )


def test_review_pages_past_drafts_sent_back(output_on, flow) -> None:
    from tekijin.data.knowledge import create_draft

    session = get_sessionmaker(flow.engine)()
    try:
        flow.ask()  # makes the organization
        org = session.scalars(
            select(Organization).where(Organization.discord_guild_id == GUILD)
        ).one()
        import datetime as dt

        for i in range(6):
            create_draft(
                session,
                org_id=org.id,
                kind="fact",
                issue=f"課題{i}",
                resolution="答え",
                now=dt.datetime(2026, 11, 5),
            )
        session.commit()
    finally:
        session.close()
    first = flow.post("/review", _user(ADMIN))
    assert first["total"] == 6 and len(first["articles"]) == 4 and first["offset"] == 0
    shown = [a["id"] for a in first["articles"]]
    second = flow.post("/review", {**_user(ADMIN), "offset": 4, "shown": shown})
    assert len(second["articles"]) == 2 and second["offset"] == 4
    assert not set(shown) & {a["id"] for a in second["articles"]}

    # Two of the first page are checked; 「次へ」 from it still shows the rest.
    for article_id in shown[:2]:
        flow.post("/review/action", {**_user(ADMIN), "article_id": article_id, "action": "verify"})
    after_checks = flow.post("/review", {**_user(ADMIN), "offset": 4, "shown": shown})
    assert after_checks["offset"] == 2
    assert {a["id"] for a in after_checks["articles"]} == {a["id"] for a in second["articles"]}

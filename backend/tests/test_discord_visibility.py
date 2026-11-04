"""The bot's channel-permission reports and the endpoints that take them (#85)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from tekijin.app import create_app
from tekijin.config import get_settings
from tekijin.data.db import get_sessionmaker
from tekijin.data.visibility import PUBLIC, SourceScope
from tekijin.discord_bot.backend import ask_backend, post_visibility
from tekijin.discord_bot.visibility import (
    REPORT_BATCH,
    VisibilityReporter,
    batches,
    channel_report,
    reader_scope,
    thread_report,
)
from tekijin.models.tables import DiscordChannel, DiscordThread
from test_api_integration import _service

SECRET = "s3cret-for-tests"
GUILD = "434343"
HEADERS = {"X-Tekijin-Bot-Secret": SECRET}


def _perms(view=True, history=True, manage=False):
    return SimpleNamespace(view_channel=view, read_message_history=history, manage_threads=manage)


class _Channel:
    def __init__(self, id_, by_role):
        self.id = id_
        self._by_role = by_role

    def permissions_for(self, who):
        return self._by_role[who.name]


MEMBER = SimpleNamespace(name="member", roles=[])
EVERYONE = SimpleNamespace(name="everyone")
CHANNELS = [
    _Channel(101, {"member": _perms(manage=True), "everyone": _perms()}),
    _Channel(102, {"member": _perms(), "everyone": _perms(view=False)}),
    _Channel(103, {"member": _perms(history=False), "everyone": _perms(view=False)}),
]


def test_reader_scope_lists_what_the_member_can_read() -> None:
    assert reader_scope(MEMBER, CHANNELS) == {
        "readable_channel_ids": [101, 102],
        "thread_manager_channel_ids": [101],
    }


def test_someone_outside_the_server_gets_no_scope() -> None:
    assert reader_scope(SimpleNamespace(name="member"), CHANNELS) is None


def test_reports_say_what_everyone_can_read_and_which_threads_are_private() -> None:
    assert channel_report(CHANNELS, EVERYONE) == [
        {"id": 101, "public": True},
        {"id": 102, "public": False},
        {"id": 103, "public": False},
    ]
    threads = [
        SimpleNamespace(id=201, type=SimpleNamespace(name="public_thread")),
        SimpleNamespace(id=202, type=SimpleNamespace(name="private_thread")),
    ]
    assert thread_report(threads) == [
        {"id": 201, "private": False},
        {"id": 202, "private": True},
    ]


def test_large_reports_are_split() -> None:
    threads = [{"id": i, "private": False} for i in range(REPORT_BATCH + 1)]
    parts = batches([{"id": 1, "public": True}], threads)
    assert [len(p["channels"]) for p in parts] == [1, 0, 0]
    assert [len(p["threads"]) for p in parts] == [0, REPORT_BATCH, 1]


def _call(handler, coro):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await coro(http)

    return asyncio.run(run())


def test_ask_and_reports_reach_the_backend() -> None:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"status": "ok", "kind": "answered"})

    scope = {"readable_channel_ids": [101], "thread_manager_channel_ids": []}
    _call(
        handler,
        lambda http: ask_backend(
            http, base_url="http://api", secret="k", user={}, question="q", scope=scope
        ),
    )
    ok = _call(
        handler,
        lambda http: post_visibility(
            http,
            base_url="http://api",
            secret="k",
            guild_id=42,
            report={"channels": [{"id": 101, "public": True}], "threads": []},
        ),
    )

    assert ok is True
    assert seen[0] == ("/internal/discord/ask", {"question": "q", **scope})
    assert seen[1][0] == "/internal/discord/visibility"
    assert seen[1][1]["guild_id"] == "42"


@pytest.fixture
def client(engine, fake_embedder, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEKIJIN_DISCORD_BOT_SECRET", SECRET)
    monkeypatch.setenv("TEKIJIN_DISCORD_GUILD_ID", GUILD)
    get_settings.cache_clear()
    service = _service(engine, fake_embedder)
    yield TestClient(create_app(agent_service=service)), service
    get_settings.cache_clear()
    with get_sessionmaker(engine)() as session:
        for table in ("discord_channels", "discord_threads", "identities"):
            session.execute(text(f"DELETE FROM {table}"))
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


def test_visibility_endpoint_stores_the_report(client, engine) -> None:
    app, _ = client
    body = {
        "guild_id": GUILD,
        "channels": [{"id": 101, "public": True}],
        "threads": [{"id": 201, "private": True}],
    }
    assert app.post("/internal/discord/visibility", json=body, headers=HEADERS).json() == {
        "status": "ok"
    }
    other = {**body, "guild_id": "1"}
    assert app.post("/internal/discord/visibility", json=other, headers=HEADERS).json() == {
        "status": "unlinked"
    }
    assert app.post("/internal/discord/visibility", json=body).status_code in (401, 403)
    with get_sessionmaker(engine)() as session:
        channels = session.execute(select(DiscordChannel.channel_id, DiscordChannel.public)).all()
        threads = session.execute(select(DiscordThread.thread_id, DiscordThread.private)).all()
    assert channels == [(101, True)] and threads == [(201, True)]


def _ask_scope(app, service, monkeypatch, **extra) -> SourceScope:
    seen: dict = {}

    def capture(session_id, asker_id, question, *, source_scope=PUBLIC):
        seen["scope"] = source_scope
        raise RuntimeError("stop here")

    monkeypatch.setattr(service, "start_question", capture)
    body = {"discord_user_id": "1001", "guild_id": GUILD, "question": "q", **extra}
    with pytest.raises(RuntimeError):
        app.post("/internal/discord/ask", json=body, headers=HEADERS)
    return seen["scope"]


def test_ask_hands_the_asker_scope_to_the_question(client, monkeypatch) -> None:
    app, service = client
    scope = _ask_scope(
        app,
        service,
        monkeypatch,
        readable_channel_ids=[101, 102],
        thread_manager_channel_ids=[102, 999],
    )
    assert scope == SourceScope(channels=frozenset({101, 102}), thread_channels=frozenset({102}))


def test_an_older_bot_gets_what_everyone_can_read(client, monkeypatch) -> None:
    app, service = client
    assert _ask_scope(app, service, monkeypatch) == PUBLIC


def _reporter(results):
    sent: list = []

    async def send(part):
        sent.append(part)
        return results.pop(0) if results else True

    return VisibilityReporter(send), sent


def test_a_failed_report_is_kept_and_sent_again() -> None:
    reporter, sent = _reporter([False])

    async def run():
        first = await reporter.report([{"id": 101, "public": True}], [])
        newer = [{"id": 101, "public": False}]
        reporter._channels.update({c["id"]: c for c in newer})
        second = await reporter.flush()
        return first, second

    first, second = asyncio.run(run())
    assert first is False and second is True
    assert sent[-1]["channels"] == [{"id": 101, "public": False}]
    assert not reporter.pending


def test_a_newer_report_replaces_a_pending_one() -> None:
    reporter, sent = _reporter([False, True])

    async def run():
        await reporter.report([], [{"id": 201, "private": False}])
        return await reporter.report([], [{"id": 201, "private": True}])

    assert asyncio.run(run()) is True
    assert sent[-1]["threads"] == [{"id": 201, "private": True}]

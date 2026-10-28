"""Integration tests for the bot's message-ingestion endpoints (#76, R2).

The bot forwards what it reads in the server's channels: new and edited messages,
deletions, and — when it starts — asks where each channel's history left off.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from tekijin.app import create_app
from tekijin.config import get_settings
from tekijin.data.db import get_sessionmaker
from tekijin.models.tables import SourceItem
from test_api_integration import _service

SECRET = "s3cret-for-tests"
GUILD = "434343"
HEADERS = {"X-Tekijin-Bot-Secret": SECRET}


@pytest.fixture
def client(engine, fake_embedder, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEKIJIN_DISCORD_BOT_SECRET", SECRET)
    monkeypatch.setenv("TEKIJIN_DISCORD_GUILD_ID", GUILD)
    get_settings.cache_clear()
    yield TestClient(create_app(agent_service=_service(engine, fake_embedder)))
    get_settings.cache_clear()
    with get_sessionmaker(engine)() as session:
        session.execute(text("DELETE FROM chunks"))
        session.execute(text("DELETE FROM source_items"))
        session.execute(text("DELETE FROM channel_messages"))
        session.execute(text("DELETE FROM identities"))
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


def _message(message_id: int, content: str, **kw) -> dict:
    return {
        "message_id": message_id,
        "channel_id": kw.get("channel_id", 20),
        "channel_name": "運営",
        "thread_id": kw.get("thread_id"),
        "thread_name": kw.get("thread_name"),
        "author_id": kw.get("author_id", "501"),
        "author_name": kw.get("author_name", "アリス"),
        "content": content,
        # 03:00 UTC = 12:00 in Japan.
        "created_at": kw.get("created_at", "2026-10-28T03:00:00+00:00"),
        "edited_at": kw.get("edited_at"),
    }


def _titles(engine) -> list[str]:
    with get_sessionmaker(engine)() as session:
        return sorted(t or "" for t in session.scalars(select(SourceItem.title)))


def test_messages_become_a_conversation_and_set_the_cursor(client, engine) -> None:
    resp = client.post(
        "/internal/discord/messages",
        json={
            "guild_id": GUILD,
            "messages": [_message(7001, "部室 鍵"), _message(7002, "守衛室", author_id="502")],
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "stored": 2, "conversations": 1}
    assert _titles(engine) == ["#運営 2026-10-28"]

    cursors = client.get("/internal/discord/cursors", params={"guild_id": GUILD}, headers=HEADERS)
    assert cursors.json() == {"status": "ok", "cursors": {"20": 7002}}


def test_deletions_remove_the_conversation(client, engine) -> None:
    body = {"guild_id": GUILD, "messages": [_message(7101, "消える")]}
    client.post("/internal/discord/messages", json=body, headers=HEADERS)

    resp = client.post(
        "/internal/discord/messages/delete",
        json={"guild_id": GUILD, "message_ids": [7101, 9999]},
        headers=HEADERS,
    )
    assert resp.json() == {"status": "ok", "deleted": 1}
    assert _titles(engine) == []


def test_another_server_is_ignored(client, engine) -> None:
    other = {"guild_id": "1", "messages": [_message(7201, "よそ")]}
    assert client.post("/internal/discord/messages", json=other, headers=HEADERS).json() == {
        "status": "unlinked",
        "stored": 0,
        "conversations": 0,
    }
    deleted = client.post(
        "/internal/discord/messages/delete",
        json={"guild_id": "1", "message_ids": [1]},
        headers=HEADERS,
    )
    assert deleted.json() == {"status": "unlinked", "deleted": 0}
    cursors = client.get("/internal/discord/cursors", params={"guild_id": "1"}, headers=HEADERS)
    assert cursors.json() == {"status": "unlinked", "cursors": {}}
    assert _titles(engine) == []


def test_the_secret_is_required(client, monkeypatch) -> None:
    body = {"guild_id": GUILD, "messages": [_message(7301, "x")]}
    assert client.post("/internal/discord/messages", json=body).status_code == 401
    monkeypatch.delenv("TEKIJIN_DISCORD_BOT_SECRET")
    get_settings.cache_clear()
    assert client.post("/internal/discord/messages", json=body, headers=HEADERS).status_code == 404


@pytest.mark.parametrize(
    "messages",
    [
        [_message(7401, "")],
        [_message(7402, "x" * 6001)],
        [_message(n, "x") for n in range(201)],
    ],
)
def test_bad_batches_are_rejected(client, messages) -> None:
    body = {"guild_id": GUILD, "messages": messages}
    assert client.post("/internal/discord/messages", json=body, headers=HEADERS).status_code == 422

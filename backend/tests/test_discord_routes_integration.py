"""Integration tests for ``POST /internal/discord/ask`` (Discord D1, #60).

The Discord bot is a separate process that calls this internal endpoint with a shared
secret. The endpoint must be invisible when the secret is unset (OFF by default), reject
a wrong secret, refuse Discord users not mapped to an employee, and otherwise run the
real graph and hand back one summarized outcome.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tekijin.app import create_app
from tekijin.config import get_settings
from tekijin.data.inbox import pending_handoffs_for_responder
from tekijin.models.tables import Question
from test_api_integration import (  # noqa: F401 - autouse cleanup of committed api_ rows
    GOOD_Q,
    _cleanup_api_rows,
    _FakeRetriever,
    _FakeScorer,
    _recs,
    _service,
)

SECRET = "s3cret-for-tests"
PATH = "/internal/discord/ask"


@pytest.fixture
def discord_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEKIJIN_DISCORD_BOT_SECRET", SECRET)
    monkeypatch.setenv("TEKIJIN_DISCORD_USER_MAP", '{"1001": 10}')
    monkeypatch.setenv("TEKIJIN_DISCORD_WEB_URL", "https://tekijin.example/")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _bot_service(engine, embedder):
    return _service(
        engine,
        embedder,
        retriever=_FakeRetriever(people=[1, 2, 3], people_confidence=0.2),
        scorer=_FakeScorer(_recs(1, 2, 3)),
    )


def _bot_client(engine, embedder) -> TestClient:
    return TestClient(create_app(agent_service=_bot_service(engine, embedder)))


def test_endpoint_is_hidden_without_a_secret(monkeypatch, engine, fake_embedder) -> None:
    monkeypatch.delenv("TEKIJIN_DISCORD_BOT_SECRET", raising=False)
    get_settings.cache_clear()
    try:
        resp = _bot_client(engine, fake_embedder).post(
            PATH, json={"discord_user_id": "1001", "question": GOOD_Q}
        )
    finally:
        get_settings.cache_clear()
    assert resp.status_code == 404


def test_wrong_or_missing_secret_is_rejected(discord_env, engine, fake_embedder) -> None:
    client = _bot_client(engine, fake_embedder)
    body = {"discord_user_id": "1001", "question": GOOD_Q}
    assert client.post(PATH, json=body).status_code == 401
    assert client.post(PATH, json=body, headers={"X-Tekijin-Bot-Secret": "no"}).status_code == 401


def test_unmapped_user_is_told_to_link(discord_env, engine, fake_embedder) -> None:
    resp = _bot_client(engine, fake_embedder).post(
        PATH,
        json={"discord_user_id": "9999", "question": GOOD_Q},
        headers={"X-Tekijin-Bot-Secret": SECRET},
    )
    assert resp.status_code == 200
    assert resp.json()["kind"] == "unlinked"


def test_person_route_returns_the_recommendations(
    discord_env, seed_counts, engine, fake_embedder
) -> None:
    resp = _bot_client(engine, fake_embedder).post(
        PATH,
        json={"discord_user_id": "1001", "question": GOOD_Q},
        headers={"X-Tekijin-Bot-Secret": SECRET},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["kind"] == "handoff"
    assert [r["person_id"] for r in body["recommendations"]] == ["E001", "E002", "E003"]
    assert body["web_url"] == "https://tekijin.example/"


def test_blank_question_is_rejected(discord_env, engine, fake_embedder) -> None:
    resp = _bot_client(engine, fake_embedder).post(
        PATH,
        json={"discord_user_id": "1001", "question": "   "},
        headers={"X-Tekijin-Bot-Secret": SECRET},
    )
    assert resp.status_code == 422


def test_unsent_handoff_is_discarded(discord_env, seed_counts, engine, fake_embedder) -> None:
    """D1 cannot send the hand-off, so the recommended person must not see it as
    pending, and the paused run must not linger (it would never be swept)."""

    service = _bot_service(engine, fake_embedder)
    resp = TestClient(create_app(agent_service=service)).post(
        PATH,
        json={"discord_user_id": "1001", "question": GOOD_Q},
        headers={"X-Tekijin-Bot-Secret": SECRET},
    )
    assert resp.json()["kind"] == "handoff"

    session = service.session_factory()
    try:
        pending = pending_handoffs_for_responder(session, 1)
        linked = session.query(Question).filter(Question.session_id.like("discord_%")).count()
    finally:
        session.close()
    assert not [p for p in pending if p["session_id"].startswith("discord_")]
    assert not [sid for sid in service._registry if sid.startswith("discord_")]
    # The checkpoint is gone, so /history must not link to a result page that 404s.
    assert linked == 0

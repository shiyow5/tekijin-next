"""Integration tests for ``POST /internal/discord/ask`` (Discord D1, #60).

The Discord bot is a separate process that calls this internal endpoint with a shared
secret. The endpoint must be invisible when the secret is unset (OFF by default), reject
a wrong secret, refuse Discord users not mapped to an employee, and otherwise run the
real graph and hand back one summarized outcome.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from tekijin.api.discord_routes import _with_document_preview
from tekijin.app import create_app
from tekijin.channels.ask_summary import AskOutcome
from tekijin.config import get_settings
from tekijin.data.inbox import pending_handoffs_for_responder
from tekijin.models.tables import Employee, Identity, Organization, Question
from test_api_integration import (  # noqa: F401 - autouse cleanup of committed api_ rows
    GOOD_Q,
    _cleanup_api_rows,
    _FakeRetriever,
    _FakeScorer,
    _recs,
    _service,
)

SECRET = "s3cret-for-tests"
GUILD = "424242424242"
HEADERS = {"X-Tekijin-Bot-Secret": SECRET}
PATH = "/internal/discord/ask"


@pytest.fixture(autouse=True)
def _cleanup_discord_members(engine):
    """Members created from Discord (org_id set) and everything they asked.

    Runs its teardown BEFORE the imported ``_cleanup_api_rows`` (fixtures tear down
    in reverse), so it removes the rows that reference these members itself.
    """

    yield
    from sqlalchemy import text

    from tekijin.data.db import get_sessionmaker

    members = "SELECT id FROM employees WHERE org_id IS NOT NULL"
    asked = f"SELECT id FROM questions WHERE asker_id IN ({members})"
    session = get_sessionmaker(engine)()
    try:
        session.execute(
            text(
                "DELETE FROM messages WHERE recommendation_id IN "
                f"(SELECT id FROM recommendations WHERE question_id IN ({asked}))"
            )
        )
        session.execute(text("DELETE FROM feedback"))
        session.execute(text(f"DELETE FROM recommendations WHERE question_id IN ({asked})"))
        session.execute(text(f"DELETE FROM events WHERE question_id IN ({asked})"))
        session.execute(text(f"DELETE FROM questions WHERE asker_id IN ({members})"))
        session.execute(text("DELETE FROM chunks"))
        session.execute(text("DELETE FROM source_items"))
        session.execute(text("DELETE FROM identities"))
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()
    finally:
        session.close()


@pytest.fixture
def discord_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEKIJIN_DISCORD_BOT_SECRET", SECRET)
    monkeypatch.setenv("TEKIJIN_DISCORD_GUILD_ID", GUILD)
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


def test_another_server_is_refused(discord_env, engine, fake_embedder) -> None:
    resp = _bot_client(engine, fake_embedder).post(
        PATH,
        json={"discord_user_id": "9999", "guild_id": "1", "question": GOOD_Q},
        headers=HEADERS,
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


def test_document_preview_reads_title_and_opening(seed_counts, engine, fake_embedder) -> None:
    service = _bot_service(engine, fake_embedder)
    out = _with_document_preview(service, AskOutcome(kind="answered", doc_id="doc_027"))
    assert out.document_title
    assert out.document_excerpt and len(out.document_excerpt) <= 300

    missing = _with_document_preview(service, AskOutcome(kind="answered", doc_id="doc_none"))
    assert missing.document_title is None


MEMBER = "/internal/discord/member"


def test_first_ask_makes_the_user_a_member(discord_env, seed_counts, engine, fake_embedder) -> None:
    service = _bot_service(engine, fake_embedder)
    client = TestClient(create_app(agent_service=service))
    body = {"discord_user_id": "1001", "display_name": "たろう", "question": GOOD_Q}
    assert client.post(PATH, json=body, headers=HEADERS).json()["kind"] == "handoff"
    body["display_name"] = "たろう(新)"
    client.post(PATH, json=body, headers=HEADERS)

    session = service.session_factory()
    try:
        ids = session.scalars(
            select(Identity).where(Identity.provider == "discord", Identity.external_id == "1001")
        ).all()
        assert len(ids) == 1
        member = session.get(Employee, ids[0].employee_id)
        org = session.get(Organization, ids[0].org_id)
    finally:
        session.close()
    assert member.name == "たろう(新)" and member.email is None
    assert org.discord_guild_id == GUILD and org.expertise_from_messages is False


def test_join_issues_a_one_time_web_login(discord_env, engine, fake_embedder) -> None:
    client = _bot_client(engine, fake_embedder)
    resp = client.post(
        MEMBER,
        json={"discord_user_id": "2002", "display_name": "はなこ", "action": "join"},
        headers=HEADERS,
    )
    body = resp.json()
    assert body["status"] == "joined"
    assert body["login_url"].startswith("https://tekijin.example/login#discord_login=")
    token = body["login_url"].split("#discord_login=", 1)[1]

    first = client.post("/auth/discord/redeem", json={"token": token})
    assert first.status_code == 200
    assert first.json()["principal"]["name"] == "はなこ"
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {first.json()['access_token']}"})
    assert me.status_code == 200 and me.json()["name"] == "はなこ"

    assert client.post("/auth/discord/redeem", json={"token": token}).status_code == 401
    assert client.post("/auth/discord/redeem", json={"token": "nope"}).status_code == 401


def test_redeem_requires_a_joined_member(discord_env, engine, fake_embedder) -> None:
    """A login token for someone who never ran ``join`` (e.g. forged with the
    right secret but for another member) does not sign in."""

    from tekijin.api.discord_routes import _login_token

    client = _bot_client(engine, fake_embedder)
    client.post(
        MEMBER,
        json={"discord_user_id": "3003", "display_name": "x", "action": "optin"},
        headers=HEADERS,
    )
    service_member = _member_by_discord_id(engine, "3003")
    token = _login_token(get_settings(), service_member)
    assert client.post("/auth/discord/redeem", json={"token": token}).status_code == 401


def test_optout_leaves_the_candidate_pool(discord_env, engine, fake_embedder) -> None:
    client = _bot_client(engine, fake_embedder)
    user = {"discord_user_id": "4004", "display_name": "じろう"}
    assert (
        client.post(MEMBER, json={**user, "action": "optout"}, headers=HEADERS).json()["status"]
        == "opted_out"
    )
    member_id = _member_by_discord_id(engine, "4004")
    assert member_id not in _pool(engine)

    assert (
        client.post(MEMBER, json={**user, "action": "optin"}, headers=HEADERS).json()["status"]
        == "opted_in"
    )
    assert member_id in _pool(engine)


def test_member_actions_refuse_another_server(discord_env, engine, fake_embedder) -> None:
    resp = _bot_client(engine, fake_embedder).post(
        MEMBER,
        json={"discord_user_id": "5005", "guild_id": "1", "action": "join"},
        headers=HEADERS,
    )
    assert resp.json() == {"status": "unlinked", "login_url": None}


def _member_by_discord_id(engine, discord_user_id: str) -> int:
    from tekijin.data.db import get_sessionmaker

    session = get_sessionmaker(engine)()
    try:
        return session.scalars(
            select(Identity.employee_id).where(Identity.external_id == discord_user_id)
        ).one()
    finally:
        session.close()


def _pool(engine) -> set[int]:
    from tekijin.data.db import get_sessionmaker
    from tekijin.data.repository import Repository

    session = get_sessionmaker(engine)()
    try:
        return {e.id for e in Repository(session).list_employees()}
    finally:
        session.close()


def test_nothing_is_served_without_a_configured_server(
    monkeypatch, discord_env, engine, fake_embedder
) -> None:
    monkeypatch.delenv("TEKIJIN_DISCORD_GUILD_ID")
    get_settings.cache_clear()
    client = _bot_client(engine, fake_embedder)
    ask = {"discord_user_id": "6006", "guild_id": "777", "question": GOOD_Q}
    assert client.post(PATH, json=ask, headers=HEADERS).json()["kind"] == "unlinked"
    join = {"discord_user_id": "6006", "guild_id": "777", "action": "join"}
    assert client.post(MEMBER, json=join, headers=HEADERS).json()["status"] == "unlinked"


def test_first_contact_survives_a_concurrent_insert(engine) -> None:
    """The loser of a create race rolls back to its savepoint and adopts the
    winner's rows instead of failing with an IntegrityError."""

    import datetime as dt

    from tekijin.data.db import get_sessionmaker
    from tekijin.data.members import get_or_create_org_for_guild, resolve_discord_member

    factory = get_sessionmaker(engine)
    now = dt.datetime(2026, 10, 27, 12, 0)
    winner, loser = factory(), factory()
    try:
        # Both look before either writes.
        assert loser.scalars(select(Organization)).first() is None
        org_w = get_or_create_org_for_guild(winner, "999", name="w")
        resolve_discord_member(winner, org_w, "7007", display_name="w", now=now)
        winner.commit()

        # The loser's first lookup is forced to miss, as if it ran before the commit.
        import tekijin.data.members as members

        real_org, real_identity = members._org_for_guild, members._discord_identity
        calls = {"org": 0, "identity": 0}

        def stale_org(session, guild_id):
            calls["org"] += 1
            return None if calls["org"] == 1 else real_org(session, guild_id)

        def stale_identity(session, org, user_id):
            calls["identity"] += 1
            return None if calls["identity"] == 1 else real_identity(session, org, user_id)

        members._org_for_guild, members._discord_identity = stale_org, stale_identity
        try:
            org_l = get_or_create_org_for_guild(loser, "999", name="l")
            member_l = resolve_discord_member(loser, org_l, "7007", display_name="w", now=now)
        finally:
            members._org_for_guild, members._discord_identity = real_org, real_identity
        loser.commit()
        assert org_l.id == org_w.id
        assert loser.scalars(select(Identity).where(Identity.external_id == "7007")).one()
        assert member_l.org_id == org_w.id
    finally:
        winner.close()
        loser.close()


ADD = "/internal/discord/add"


def _join(client, discord_user_id: str = "3003") -> dict:
    user = {"discord_user_id": discord_user_id, "display_name": "会計係"}
    client.post(MEMBER, json={**user, "action": "join"}, headers=HEADERS)
    return user


def test_add_requires_a_joined_member(discord_env, engine, fake_embedder) -> None:
    client = _bot_client(engine, fake_embedder)
    body = {"discord_user_id": "3004", "display_name": "未参加", "text": "部室 鍵"}
    assert client.post(ADD, json=body, headers=HEADERS).json() == {
        "status": "not_joined",
        "chunks": 0,
    }


def test_add_stores_a_note_and_resending_a_url_updates_it(
    discord_env, engine, fake_embedder
) -> None:
    from tekijin.data.db import get_sessionmaker
    from tekijin.models.tables import Chunk, SourceItem

    client = _bot_client(engine, fake_embedder)
    user = _join(client)
    note = {
        **user,
        "title": "会計の引継ぎ",
        "text": "通帳 印鑑 部室 金庫",
        "url": "https://ex.com/a",
    }
    first = client.post(ADD, json=note, headers=HEADERS).json()
    assert first == {"status": "added", "chunks": 1}
    same = client.post(ADD, json=note, headers=HEADERS).json()
    assert same["status"] == "unchanged"
    changed = client.post(ADD, json={**note, "text": "通帳 印鑑 会計 ロッカー"}, headers=HEADERS)
    assert changed.json()["status"] == "updated"
    # Without a URL every add is a new note.
    plain = {**user, "text": "合宿 宿 予約"}
    assert client.post(ADD, json=plain, headers=HEADERS).json()["status"] == "added"
    assert client.post(ADD, json=plain, headers=HEADERS).json()["status"] == "added"

    with get_sessionmaker(engine)() as session:
        items = session.scalars(select(SourceItem).order_by(SourceItem.id)).all()
        assert len(items) == 3
        assert items[0].url == "https://ex.com/a" and items[0].body == "通帳 印鑑 会計 ロッカー"
        assert items[0].author_id == _member_by_discord_id(engine, "3003")
        assert items[1].title is None
        assert session.scalar(select(func.count()).select_from(Chunk)) == 3


@pytest.mark.parametrize(
    "extra",
    [
        {"text": "   "},
        {"text": "x" * 4001},
        {"text": "ok", "url": "javascript:alert(1)"},
        {"text": "ok", "title": "t" * 201},
    ],
)
def test_add_rejects_bad_input(discord_env, engine, fake_embedder, extra) -> None:
    client = _bot_client(engine, fake_embedder)
    user = _join(client)
    assert client.post(ADD, json={**user, **extra}, headers=HEADERS).status_code == 422


def test_add_refuses_another_server(discord_env, engine, fake_embedder) -> None:
    client = _bot_client(engine, fake_embedder)
    body = {"discord_user_id": "3005", "guild_id": "1", "text": "部室 鍵"}
    assert client.post(ADD, json=body, headers=HEADERS).json()["status"] == "unlinked"

"""The web's knowledge screen: members read verified knowledge, the admin reviews (#112).

``GET /articles`` lists the organization's verified knowledge to every member; the
review queue (``view=review``) and the review actions (edit, verify, send back,
retire) are for the organization's administrator only.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from tekijin.app import create_app
from tekijin.config import get_settings
from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import SourceRef, create_draft, verify_article
from tekijin.models.tables import (
    ChannelMessage,
    DiscordChannel,
    Employee,
    KnowledgeArticle,
    Organization,
    SourceItem,
)
from test_api_integration import _admin_headers, _service, _user_headers

GUILD = "737373"
NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
PRIVATE = 5005


@pytest.fixture
def world(engine, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEKIJIN_DISCORD_GUILD_ID", GUILD)
    get_settings.cache_clear()
    factory = get_sessionmaker(engine)
    with factory() as session:
        org = Organization(name="サークル", discord_guild_id=GUILD)
        other = Organization(name="ほか", discord_guild_id="848484")
        session.add_all([org, other])
        session.flush()
        alice = Employee(name="アリス", org_id=org.id)
        carol = Employee(name="キャロル", org_id=other.id)
        session.add_all([alice, carol])
        session.flush()
        item = SourceItem(
            org_id=org.id,
            provider="manual",
            external_id="manual:kr",
            title="合宿の手引き",
            url="https://ex.com/camp",
            body="鍵は箱",
            author_id=alice.id,
            content_hash="h",
            updated_at=NOW,
        )
        session.add(item)
        session.add(DiscordChannel(org_id=org.id, channel_id=PRIVATE, public=False, updated_at=NOW))
        session.flush()
        secret = ChannelMessage(
            org_id=org.id,
            message_id=77,
            channel_id=PRIVATE,
            channel_name="運営",
            author_id=alice.id,
            content="暗証番号",
            created_at=NOW,
        )
        session.add(secret)
        session.flush()
        verified = create_draft(
            session,
            org_id=org.id,
            kind="procedure",
            issue="部室の鍵はどこ",
            resolution="部室の箱",
            author_id=alice.id,
            sources=[SourceRef(source_item_id=item.id, quote="鍵は箱")],
            now=NOW,
        )
        verify_article(session, verified.id, org_id=org.id, verifier_id=alice.id, now=NOW)
        draft = create_draft(
            session, org_id=org.id, kind="fact", issue="会費はいくら", resolution="月500円", now=NOW
        )
        private = create_draft(
            session,
            org_id=org.id,
            kind="fact",
            issue="暗証番号は",
            resolution="会計係に聞く",
            sources=[SourceRef(channel_message_id=secret.id)],
            now=NOW,
        )
        verify_article(session, private.id, org_id=org.id, verifier_id=None, now=NOW)
        foreign = create_draft(
            session, org_id=other.id, kind="fact", issue="よその話", resolution="x", now=NOW
        )
        verify_article(session, foreign.id, org_id=other.id, verifier_id=None, now=NOW)
        session.commit()
        ids = {
            "alice": alice.id,
            "carol": carol.id,
            "verified": verified.id,
            "draft": draft.id,
            "private": private.id,
            "foreign": foreign.id,
            "item": item.id,
        }
    yield ids
    get_settings.cache_clear()
    with factory() as session:
        for table in (
            "knowledge_sources",
            "knowledge_articles",
            "channel_messages",
            "discord_channels",
            "chunks",
            "source_items",
        ):
            session.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed names
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


@pytest.fixture
def client(engine, fake_embedder) -> TestClient:
    return TestClient(create_app(agent_service=_service(engine, fake_embedder)))


def _state(engine, article_id: int) -> str:
    with get_sessionmaker(engine)() as session:
        return session.get(KnowledgeArticle, article_id).state


# --------------------------------------------------------------------------- #
# members
# --------------------------------------------------------------------------- #
def test_a_member_lists_verified_knowledge_of_their_organization(client, world) -> None:
    body = client.get("/articles", headers=_user_headers(world["alice"])).json()
    assert body["can_review"] is False
    assert [i["id"] for i in body["items"]] == [world["verified"]]
    assert body["total"] == 1
    item = body["items"][0]
    assert (item["issue"], item["resolution"], item["state"]) == (
        "部室の鍵はどこ",
        "部室の箱",
        "verified",
    )
    assert item["author_name"] == "アリス" and item["verifier_name"] == "アリス"


def test_a_member_reads_a_verified_article_with_its_sources(client, world) -> None:
    resp = client.get(f"/articles/{world['verified']}", headers=_user_headers(world["alice"]))
    assert resp.status_code == 200
    sources = resp.json()["sources"]
    assert sources == [
        {
            "kind": "source",
            "source_item_id": world["item"],
            "title": "合宿の手引き",
            "url": "https://ex.com/camp",
            "quote": "鍵は箱",
        }
    ]


def test_a_member_cannot_read_drafts_or_private_or_foreign_knowledge(client, world) -> None:
    alice = _user_headers(world["alice"])
    for key in ("draft", "private", "foreign"):
        assert client.get(f"/articles/{world[key]}", headers=alice).status_code == 404
    assert client.get("/articles?view=review", headers=alice).status_code == 403


def test_another_organization_sees_only_its_own(client, world) -> None:
    body = client.get("/articles", headers=_user_headers(world["carol"])).json()
    assert [i["id"] for i in body["items"]] == [world["foreign"]]
    resp = client.get(f"/articles/{world['verified']}", headers=_user_headers(world["carol"]))
    assert resp.status_code == 404


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("patch", "/articles/{id}", {"issue": "x"}),
        ("post", "/articles/{id}/verify", None),
        ("post", "/articles/{id}/send-back", None),
        ("post", "/articles/{id}/retire", {}),
    ],
)
def test_a_member_cannot_review(client, world, engine, method, path, body) -> None:
    url = path.format(id=world["draft"])
    resp = getattr(client, method)(url, headers=_user_headers(world["alice"]), json=body)
    assert resp.status_code == 403
    assert _state(engine, world["draft"]) == "draft"


def test_login_is_required(client) -> None:
    assert client.get("/articles").status_code == 401
    assert client.post("/articles/1/verify").status_code == 401


# --------------------------------------------------------------------------- #
# the admin
# --------------------------------------------------------------------------- #
def test_the_admin_sees_the_review_queue(client, world) -> None:
    body = client.get("/articles?view=review", headers=_admin_headers()).json()
    assert body["can_review"] is True
    assert [i["id"] for i in body["items"]] == [world["draft"]]
    verified = client.get("/articles", headers=_admin_headers()).json()
    # The admin also reads knowledge from channels members cannot see.
    assert {i["id"] for i in verified["items"]} == {world["verified"], world["private"]}


def test_the_admin_edits_and_verifies_a_draft(client, world, engine) -> None:
    admin = _admin_headers()
    url = f"/articles/{world['draft']}"
    edited = client.patch(url, headers=admin, json={"resolution": "月600円", "cause": "値上げ"})
    assert edited.status_code == 200
    assert edited.json()["resolution"] == "月600円" and edited.json()["state"] == "draft"
    done = client.post(f"{url}/verify", headers=admin)
    assert done.status_code == 200 and done.json()["state"] == "verified"
    with get_sessionmaker(engine)() as session:
        row = session.get(KnowledgeArticle, world["draft"])
        assert row.embedding is not None  # searchable right away
        assert row.verified_at is not None
    listed = client.get("/articles", headers=_user_headers(world["alice"])).json()["items"]
    assert world["draft"] in {i["id"] for i in listed}


def test_send_back_and_retire(client, world, engine) -> None:
    admin = _admin_headers()
    back = client.post(f"/articles/{world['verified']}/send-back", headers=admin)
    assert back.status_code == 200 and back.json()["state"] == "wip"
    retired = client.post(
        f"/articles/{world['draft']}/retire",
        headers=admin,
        json={"superseded_by": world["verified"]},
    )
    assert retired.status_code == 200
    assert retired.json()["state"] == "retired"
    assert retired.json()["superseded_by"] == world["verified"]


def test_a_wip_article_cannot_be_verified(client, world) -> None:
    admin = _admin_headers()
    client.patch(f"/articles/{world['draft']}", headers=admin, json={"resolution": ""})
    assert client.post(f"/articles/{world['draft']}/verify", headers=admin).status_code == 409


def test_the_admin_cannot_touch_another_organization(client, world) -> None:
    admin = _admin_headers()
    assert client.post(f"/articles/{world['foreign']}/verify", headers=admin).status_code == 404
    resp = client.post(
        f"/articles/{world['draft']}/retire",
        headers=admin,
        json={"superseded_by": world["foreign"]},
    )
    assert resp.status_code == 404


@pytest.mark.parametrize(
    "body",
    [{"issue": ""}, {"kind": "rumor"}, {"issue": "x" * 2001}, {"state": "verified"}],
)
def test_bad_edits_are_refused(client, world, body) -> None:
    resp = client.patch(f"/articles/{world['draft']}", headers=_admin_headers(), json=body)
    assert resp.status_code == 422

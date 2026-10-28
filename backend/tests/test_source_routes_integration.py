"""Integration tests for the web's source endpoints (#72, R2).

``GET/POST/DELETE /sources`` let a signed-in member list, add and remove their
organization's documents. The organization comes from the member; the admin, who
belongs to none, works on the configured Discord server's organization.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from tekijin.app import create_app
from tekijin.config import get_settings
from tekijin.data.db import get_sessionmaker
from tekijin.models.tables import Employee, Organization
from test_api_integration import _admin_headers, _service, _user_headers

GUILD = "515151"


@pytest.fixture
def orgs(engine, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEKIJIN_DISCORD_GUILD_ID", GUILD)
    get_settings.cache_clear()
    factory = get_sessionmaker(engine)
    with factory() as session:
        org = Organization(name="サークル", discord_guild_id=GUILD)
        other = Organization(name="ほか", discord_guild_id="626262")
        session.add_all([org, other])
        session.flush()
        alice = Employee(name="アリス", org_id=org.id)
        bob = Employee(name="ボブ", org_id=org.id)
        carol = Employee(name="キャロル", org_id=other.id)
        session.add_all([alice, bob, carol])
        session.commit()
        ids = {"alice": alice.id, "bob": bob.id, "carol": carol.id}
    yield ids
    get_settings.cache_clear()
    with factory() as session:
        session.execute(text("DELETE FROM chunks"))
        session.execute(text("DELETE FROM source_items"))
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


@pytest.fixture
def client(engine, fake_embedder) -> TestClient:
    return TestClient(create_app(agent_service=_service(engine, fake_embedder)))


def _as(employee_id: int) -> dict[str, str]:
    return _user_headers(employee_id)


NOTE = {"title": "会計の引継ぎ", "body": "通帳 印鑑 金庫", "url": "https://ex.com/kaikei"}


def test_add_list_and_delete_own_source(client, orgs) -> None:
    added = client.post("/sources", json=NOTE, headers=_as(orgs["alice"]))
    assert added.status_code == 201
    assert added.json() == {"status": "added", "id": added.json()["id"], "chunks": 1}

    listed = client.get("/sources", headers=_as(orgs["bob"])).json()["items"]
    assert len(listed) == 1
    item = listed[0]
    assert (item["title"], item["url"], item["author_name"]) == (
        "会計の引継ぎ",
        "https://ex.com/kaikei",
        "アリス",
    )
    assert item["provider"] == "manual" and item["chunks"] == 1
    # Bob may read it but not delete it.
    assert item["can_delete"] is False
    assert client.delete(f"/sources/{item['id']}", headers=_as(orgs["bob"])).status_code == 403

    mine = client.get("/sources", headers=_as(orgs["alice"])).json()["items"][0]
    assert mine["can_delete"] is True
    assert client.delete(f"/sources/{item['id']}", headers=_as(orgs["alice"])).status_code == 204
    assert client.get("/sources", headers=_as(orgs["alice"])).json()["items"] == []


def test_resending_a_url_updates_and_a_note_without_url_is_new(client, orgs) -> None:
    alice = _as(orgs["alice"])
    first = client.post("/sources", json=NOTE, headers=alice).json()
    assert client.post("/sources", json=NOTE, headers=alice).json()["status"] == "unchanged"
    again = client.post("/sources", json={**NOTE, "body": "通帳 印鑑 ロッカー"}, headers=alice)
    assert again.json() == {"status": "updated", "id": first["id"], "chunks": 1}
    plain = {"body": "合宿 宿 予約"}
    ids = {client.post("/sources", json=plain, headers=alice).json()["id"] for _ in range(2)}
    assert len(ids) == 2


def test_another_organization_cannot_see_or_delete(client, orgs) -> None:
    item_id = client.post("/sources", json=NOTE, headers=_as(orgs["alice"])).json()["id"]
    carol = _as(orgs["carol"])
    assert client.get("/sources", headers=carol).json()["items"] == []
    assert client.delete(f"/sources/{item_id}", headers=carol).status_code == 404


def test_admin_works_on_the_configured_server(client, orgs) -> None:
    item_id = client.post("/sources", json=NOTE, headers=_as(orgs["alice"])).json()["id"]
    admin = _admin_headers()
    listed = client.get("/sources", headers=admin).json()["items"]
    assert [i["id"] for i in listed] == [item_id]
    assert listed[0]["can_delete"] is True
    added = client.post("/sources", json={"body": "部室 鍵"}, headers=admin)
    assert added.status_code == 201
    assert client.delete(f"/sources/{item_id}", headers=admin).status_code == 204


def test_admin_without_an_organization_is_told_so(client, engine, monkeypatch) -> None:
    monkeypatch.delenv("TEKIJIN_DISCORD_GUILD_ID", raising=False)
    get_settings.cache_clear()
    try:
        resp = client.get("/sources", headers=_admin_headers())
    finally:
        get_settings.cache_clear()
    assert resp.status_code == 409


def test_a_member_outside_any_organization_is_refused(client, seed_counts) -> None:
    # Seeded employees predate organizations (org_id NULL).
    assert client.get("/sources", headers=_as(1)).status_code == 403


def test_login_is_required(client) -> None:
    assert client.get("/sources").status_code == 401
    assert client.post("/sources", json=NOTE).status_code == 401


@pytest.mark.parametrize(
    "body",
    [
        {"body": "   "},
        {"body": "x" * 20001},
        {"body": "ok", "url": "javascript:alert(1)"},
        {"body": "ok", "title": "t" * 201},
    ],
)
def test_bad_input_is_rejected(client, orgs, body) -> None:
    assert client.post("/sources", json=body, headers=_as(orgs["alice"])).status_code == 422

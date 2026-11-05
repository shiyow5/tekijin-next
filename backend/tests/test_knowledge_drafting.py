"""Knowledge drafting wired to ingestion: documents, messages, reactions (#113).

The app hands drafting to :class:`KnowledgeDrafting`; here it runs inline with
fake models, so each request has drafted by the time it returns.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from tekijin.app import create_app
from tekijin.config import Settings, get_settings
from tekijin.data.db import get_sessionmaker
from tekijin.knowledge.documents import DocumentDraftExtractor
from tekijin.knowledge.drafting import KnowledgeDrafting
from tekijin.knowledge.extract import DraftExtractor
from tekijin.llm.schemas import DocumentKnowledgeItem, DocumentKnowledgeSchema, KnowledgeDraftSchema
from tekijin.models.tables import KnowledgeArticle, Organization
from test_api_integration import _admin_headers, _service

SECRET = "s3cret-for-tests"
GUILD = "4311311"
HEADERS = {"X-Tekijin-Bot-Secret": SECRET}
NOW = dt.datetime(2026, 11, 5, 12, 0, 0)


class _CaseModel:
    def invoke(self, prompt):
        return KnowledgeDraftSchema(
            extractable=True, kind="case", issue="ピントが合わない", resolution="延長筒を付ける"
        )


class _DocModel:
    def invoke(self, prompt):
        return DocumentKnowledgeSchema(
            items=[DocumentKnowledgeItem(kind="fact", issue="鍵はどこ", resolution="守衛室")]
        )


class _Raising:
    def invoke(self, prompt):
        raise RuntimeError("model down")


def _drafting(engine, embedder, *, documents: bool = True, inline: bool = True, case=None):
    return KnowledgeDrafting(
        session_factory=get_sessionmaker(engine),
        embedder=embedder,
        extractor=DraftExtractor(model=case or _CaseModel()),
        document_extractor=DocumentDraftExtractor(model=_DocModel()),
        documents_enabled=documents,
        inline=inline,
        now=lambda: NOW,
    )


@pytest.fixture
def app(engine, fake_embedder, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEKIJIN_DISCORD_BOT_SECRET", SECRET)
    monkeypatch.setenv("TEKIJIN_DISCORD_GUILD_ID", GUILD)
    get_settings.cache_clear()
    application = create_app(agent_service=_service(engine, fake_embedder))
    application.state.knowledge_drafting = _drafting(engine, fake_embedder)
    yield application
    get_settings.cache_clear()
    with get_sessionmaker(engine)() as session:
        for table in (
            "knowledge_sources",
            "knowledge_articles",
            "chunks",
            "self_intros",
            "source_items",
            "channel_messages",
            "identities",
        ):
            session.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed names
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


@pytest.fixture
def client(app):
    return TestClient(app)


def _message(message_id: int, content: str, author: str) -> dict:
    return {
        "message_id": message_id,
        "channel_id": 20,
        "channel_name": "機材",
        "thread_id": 77,
        "thread_name": "ピント",
        "author_id": author,
        "author_name": f"人{author}",
        "content": content,
        "created_at": f"2026-10-28T03:0{message_id % 10}:00+00:00",
    }


def _articles(engine) -> list[tuple[str, str, bool]]:
    with get_sessionmaker(engine)() as session:
        rows = session.scalars(select(KnowledgeArticle).order_by(KnowledgeArticle.id)).all()
        return [(r.kind, r.state, r.embedding is not None) for r in rows]


def _switch_conversations(engine, on: bool) -> None:
    with get_sessionmaker(engine)() as session:
        session.execute(
            text("UPDATE organizations SET knowledge_from_conversations = :on"), {"on": on}
        )
        session.commit()


def _post_messages(client, *messages) -> None:
    resp = client.post(
        "/internal/discord/messages",
        json={"guild_id": GUILD, "messages": list(messages)},
        headers=HEADERS,
    )
    assert resp.json()["status"] == "ok"


QUESTION = _message(1, "ピントが合いません", "501")
ANSWER = _message(2, "延長筒を付けて", "502")
THANKS = _message(3, "合いました、ありがとうございます", "501")


def test_the_askers_thanks_drafts_a_case_when_the_org_has_it_on(client, engine) -> None:
    _post_messages(client, QUESTION, ANSWER)
    _post_messages(client, THANKS)
    assert _articles(engine) == []  # off by default
    _switch_conversations(engine, True)
    _post_messages(client, _message(4, "もう一つ質問です", "501"), _message(5, "どうぞ", "502"))
    _post_messages(client, _message(6, "直りました、ありがとうございます", "501"))
    assert _articles(engine) == [("case", "draft", True)]


def test_a_check_mark_drafts_a_case(client, engine) -> None:
    _post_messages(client, QUESTION, ANSWER)
    _switch_conversations(engine, True)
    body = {"guild_id": GUILD, "channel_id": 77, "message_id": 2, "user_id": "501"}
    thumbs = client.post(
        "/internal/discord/reactions", json={**body, "emoji": "👍"}, headers=HEADERS
    )
    assert thumbs.json() == {"status": "ignored"}
    other = client.post(
        "/internal/discord/reactions",
        json={**body, "guild_id": "1", "emoji": "✅"},
        headers=HEADERS,
    )
    assert other.json() == {"status": "unlinked"}
    assert _articles(engine) == []
    unknown = client.post(
        "/internal/discord/reactions",
        json={**body, "message_id": 99, "emoji": "✅"},
        headers=HEADERS,
    )
    assert unknown.json() == {"status": "pending"}
    resp = client.post("/internal/discord/reactions", json={**body, "emoji": "✅"}, headers=HEADERS)
    assert resp.json() == {"status": "ok"}
    assert _articles(engine) == [("case", "draft", True)]


def test_the_reaction_route_needs_the_secret(client) -> None:
    body = {"guild_id": GUILD, "channel_id": 1, "message_id": 2, "user_id": "3", "emoji": "✅"}
    assert client.post("/internal/discord/reactions", json=body).status_code in (401, 403)


def test_a_document_added_on_the_web_is_drafted(client, engine, app) -> None:
    with get_sessionmaker(engine)() as session:
        session.add(Organization(name="サークル", discord_guild_id=GUILD))
        session.commit()
    body = {"title": "機材マニュアル", "body": "鍵は守衛室で借ります", "url": "https://ex.com/m"}
    resp = client.post("/sources", json=body, headers=_admin_headers())
    assert resp.status_code == 201, resp.text
    assert _articles(engine) == [("fact", "draft", True)]
    # The same text again is unchanged: nothing is read.
    client.post("/sources", json=body, headers=_admin_headers())
    assert len(_articles(engine)) == 1


def test_documents_stay_unread_with_the_setting_off(engine, fake_embedder) -> None:
    runner = _drafting(engine, fake_embedder, documents=False)
    assert runner.document_changed(org_id=1, item_id=1) is False


def test_from_settings_reads_nothing_on_the_stub_backend(engine, fake_embedder) -> None:
    settings = Settings(llm_backend="stub", knowledge_from_documents_enabled=True)
    runner = KnowledgeDrafting.from_settings(
        settings, session_factory=get_sessionmaker(engine), embedder=fake_embedder
    )
    assert runner.document_changed(org_id=1, item_id=1) is False
    assert runner.conversation_messages(org_id=1, message_ids=[1]) is False
    assert runner.reaction(org_id=1, message_id=1, reactor_discord_id="1") is False
    real = KnowledgeDrafting.from_settings(
        Settings(llm_backend="vllm"),
        session_factory=get_sessionmaker(engine),
        embedder=fake_embedder,
    )
    assert real.conversation_messages(org_id=1, message_ids=[]) is False


def test_jobs_run_on_a_worker_thread_and_failures_are_logged(client, engine, fake_embedder):
    _post_messages(client, QUESTION, ANSWER, THANKS)
    _switch_conversations(engine, True)
    with get_sessionmaker(engine)() as session:
        org_id = session.scalar(select(Organization.id))
    broken = _drafting(engine, fake_embedder, inline=False, case=_Raising())
    assert broken.conversation_messages(org_id=org_id, message_ids=[3])
    broken.join()
    assert _articles(engine) == []
    threaded = _drafting(engine, fake_embedder, inline=False)
    threaded.reaction(org_id=org_id, message_id=2, reactor_discord_id="501")
    threaded.join()
    assert _articles(engine) == [("case", "draft", True)]

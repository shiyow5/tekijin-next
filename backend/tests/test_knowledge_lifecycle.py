"""What happens to knowledge when what it was made from changes or goes (#113).

* A document that changes sends the verified knowledge made from it back to review,
  with a note saying the source changed. So does an edited Discord message.
* A source that is lost (a Discord message deleted, a conversation hidden from the
  web, a document deleted) sends verified knowledge back to review, and a draft with
  no source left is deleted.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select, text

from tekijin.data.conversations import (
    IncomingMessage,
    conversation_key,
    delete_conversation,
    delete_messages,
    ingest_messages,
)
from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import (
    SOURCE_CHANGED_NOTE,
    SOURCE_EDITED_NOTE,
    SOURCE_LOST_NOTE,
    SourceRef,
    create_draft,
    reconcile_lost_sources,
    return_to_review,
    update_article,
    verify_article,
)
from tekijin.data.sources import MANUAL, delete_source_item, upsert_source_item
from tekijin.models.tables import ChannelMessage, KnowledgeArticle, KnowledgeSource, Organization

GUILD = "113113"
NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
LATER = NOW + dt.timedelta(hours=1)


@pytest.fixture
def org(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        row = Organization(name="サークル", discord_guild_id=GUILD)
        session.add(row)
        session.commit()
        org_id = row.id
    yield org_id
    with factory() as session:
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
def db(engine, org):
    factory = get_sessionmaker(engine)
    with factory() as session:
        yield session
        session.rollback()


def _msg(message_id: int, content: str, minute: int, author: str = "1") -> IncomingMessage:
    return IncomingMessage(
        message_id=message_id,
        channel_id=10,
        channel_name="機材",
        thread_id=None,
        thread_name=None,
        author_discord_id=author,
        author_name=f"人{author}",
        content=content,
        created_at=NOW + dt.timedelta(minutes=minute),
        edited_at=None,
    )


def _document(db, org, embedder, body: str = "鍵は守衛室で借ります") -> int:
    return upsert_source_item(
        db,
        embedder,
        org_id=org,
        provider=MANUAL,
        external_id="circle:manual",
        title="マニュアル",
        body=body,
        url=None,
        author_id=None,
        now=NOW,
    ).item_id


def _org(db, org) -> Organization:
    return db.get(Organization, org)


def _messages(db, org, embedder, *messages: IncomingMessage) -> dict[int, int]:
    ingest_messages(db, embedder, org=_org(db, org), guild_id=GUILD, messages=messages, now=NOW)
    rows = db.scalars(select(ChannelMessage).where(ChannelMessage.org_id == org)).all()
    return {r.message_id: r.id for r in rows}


def _article(db, org, *sources: SourceRef, verified: bool = True, answer: str | None = "答え"):
    article = create_draft(
        db,
        org_id=org,
        kind="procedure",
        issue="鍵はどこで借りる",
        resolution=answer,
        sources=sources,
        now=NOW,
    )
    if verified:
        verify_article(db, article.id, org_id=org, verifier_id=None, now=NOW)
    return article.id


def _state(db, article_id: int) -> tuple[str, str | None] | None:
    db.expire_all()
    row = db.get(KnowledgeArticle, article_id)
    return None if row is None else (row.state, row.review_note)


# --------------------------------------------------------------------------- #
# the store
# --------------------------------------------------------------------------- #
def test_return_to_review_marks_every_article_made_from_the_source(db, org, fake_embedder):
    item = _document(db, org, fake_embedder)
    verified = _article(db, org, SourceRef(source_item_id=item))
    drafted = _article(db, org, SourceRef(source_item_id=item), verified=False)
    unrelated = _article(db, org)
    returned = return_to_review(
        db, org_id=org, source_item_ids=[item], note=SOURCE_CHANGED_NOTE, now=LATER
    )
    assert returned == [verified, drafted]
    assert _state(db, verified) == ("draft", SOURCE_CHANGED_NOTE)
    row = db.get(KnowledgeArticle, verified)
    # Who last vouched for it, and when, is kept next to the note.
    assert (row.verified_at, row.updated_at) == (NOW, LATER)
    # A draft gets the note too, so it is not verified without the admin seeing it.
    assert _state(db, drafted) == ("draft", SOURCE_CHANGED_NOTE)
    assert _state(db, unrelated) == ("verified", None)
    assert return_to_review(db, org_id=org, note=SOURCE_CHANGED_NOTE, now=LATER) == []


def test_verifying_again_clears_the_note(db, org, fake_embedder) -> None:
    item = _document(db, org, fake_embedder)
    article = _article(db, org, SourceRef(source_item_id=item))
    return_to_review(db, org_id=org, source_item_ids=[item], note=SOURCE_CHANGED_NOTE, now=LATER)
    update_article(db, article, org_id=org, changes={"cause": "古い"}, now=LATER)
    assert _state(db, article) == ("draft", SOURCE_CHANGED_NOTE)
    verify_article(db, article, org_id=org, verifier_id=None, now=LATER)
    assert _state(db, article) == ("verified", None)


def test_reconcile_with_nothing_lost_does_nothing(db, org) -> None:
    article = _article(db, org)
    report = reconcile_lost_sources(db, org_id=org, now=LATER)
    assert (report.returned, report.deleted) == ((), ())
    assert _state(db, article) == ("verified", None)


def test_reconcile_returns_verified_and_deletes_sourceless_drafts(db, org, fake_embedder) -> None:
    item = _document(db, org, fake_embedder)
    other = upsert_source_item(
        db,
        fake_embedder,
        org_id=org,
        provider=MANUAL,
        external_id="circle:other",
        title="引継ぎ",
        body="会計は月末に締める",
        url=None,
        author_id=None,
        now=NOW,
    ).item_id
    verified_alone = _article(db, org, SourceRef(source_item_id=item))
    verified_shared = _article(
        db, org, SourceRef(source_item_id=item), SourceRef(source_item_id=other)
    )
    draft_alone = _article(db, org, SourceRef(source_item_id=item), verified=False)
    wip_alone = _article(db, org, SourceRef(source_item_id=item), verified=False, answer=None)
    draft_shared = _article(
        db, org, SourceRef(source_item_id=item), SourceRef(source_item_id=other), verified=False
    )
    retired = _article(db, org, SourceRef(source_item_id=item), verified=False)
    db.get(KnowledgeArticle, retired).state = "retired"
    db.flush()

    assert delete_source_item(db, item_id=item, org_id=org, now=LATER)

    assert _state(db, verified_alone) == ("draft", SOURCE_LOST_NOTE)
    assert _state(db, verified_shared) == ("draft", SOURCE_LOST_NOTE)
    assert _state(db, draft_alone) is None
    assert _state(db, wip_alone) is None
    assert _state(db, retired) is None
    # Nobody verified it but a source is left: it stays, with the note.
    assert _state(db, draft_shared) == ("draft", SOURCE_LOST_NOTE)
    # The lost rows are gone, so a re-verified article is shown by its live sources.
    lost = db.scalars(
        select(KnowledgeSource).where(
            KnowledgeSource.source_item_id.is_(None),
            KnowledgeSource.channel_message_id.is_(None),
            KnowledgeSource.question_id.is_(None),
        )
    ).all()
    assert lost == []
    # Run again: nothing left to do.
    again = reconcile_lost_sources(db, org_id=org, now=LATER)
    assert (again.returned, again.deleted) == ((), ())


def test_reviewed_knowledge_survives_losing_its_sources_one_by_one(db, org, fake_embedder):
    """A verified article back in review is not deleted with its last source (Codex)."""

    first = _document(db, org, fake_embedder)
    second = upsert_source_item(
        db,
        fake_embedder,
        org_id=org,
        provider=MANUAL,
        external_id="circle:second",
        title="引継ぎ",
        body="会計は月末に締める",
        url=None,
        author_id=None,
        now=NOW,
    ).item_id
    article = _article(db, org, SourceRef(source_item_id=first), SourceRef(source_item_id=second))
    delete_source_item(db, item_id=first, org_id=org, now=LATER)
    assert _state(db, article) == ("draft", SOURCE_LOST_NOTE)
    delete_source_item(db, item_id=second, org_id=org, now=LATER)
    assert _state(db, article) == ("draft", SOURCE_LOST_NOTE)


def test_a_changed_then_deleted_document_keeps_its_reviewed_knowledge(db, org, fake_embedder):
    item = _document(db, org, fake_embedder)
    article = _article(db, org, SourceRef(source_item_id=item))
    _document(db, org, fake_embedder, body="鍵は事務室で借ります")
    assert _state(db, article) == ("draft", SOURCE_CHANGED_NOTE)
    delete_source_item(db, item_id=item, org_id=org, now=LATER)
    assert _state(db, article) == ("draft", SOURCE_LOST_NOTE)


def test_a_changed_document_drops_its_unreviewed_drafts(db, org, fake_embedder) -> None:
    item = _document(db, org, fake_embedder)
    draft = _article(db, org, SourceRef(source_item_id=item), verified=False)
    _document(db, org, fake_embedder, body="鍵は事務室で借ります")
    assert _state(db, draft) is None


def test_reconcile_stays_inside_the_organization(db, org, fake_embedder) -> None:
    other_org = Organization(name="よそ", discord_guild_id="999113")
    db.add(other_org)
    db.flush()
    item = upsert_source_item(
        db,
        fake_embedder,
        org_id=other_org.id,
        provider=MANUAL,
        external_id="x",
        title="よそ",
        body="よその資料",
        url=None,
        author_id=None,
        now=NOW,
    ).item_id
    theirs = _article(db, other_org.id, SourceRef(source_item_id=item), verified=False)
    db.execute(text("UPDATE knowledge_sources SET source_item_id = NULL"))
    report = reconcile_lost_sources(db, org_id=org, now=LATER)
    assert report.deleted == ()
    assert _state(db, theirs) == ("draft", None)


# --------------------------------------------------------------------------- #
# the ingestion paths
# --------------------------------------------------------------------------- #
def test_an_updated_document_sends_its_verified_knowledge_back(db, org, fake_embedder) -> None:
    item = _document(db, org, fake_embedder)
    article = _article(db, org, SourceRef(source_item_id=item))
    # Unchanged: nothing happens.
    _document(db, org, fake_embedder)
    assert _state(db, article) == ("verified", None)
    _document(db, org, fake_embedder, body="鍵は事務室で借ります")
    assert _state(db, article) == ("draft", SOURCE_CHANGED_NOTE)


def test_a_deleted_discord_message_sends_its_knowledge_back(db, org, fake_embedder) -> None:
    ids = _messages(
        db, org, fake_embedder, _msg(1, "ピントが合いません", 0), _msg(2, "延長筒を付けて", 1, "2")
    )
    case = _article(
        db,
        org,
        SourceRef(channel_message_id=ids[1]),
        SourceRef(channel_message_id=ids[2]),
    )
    single = _article(db, org, SourceRef(channel_message_id=ids[2]), verified=False)
    delete_messages(db, fake_embedder, org_id=org, guild_id=GUILD, message_ids=[2], now=LATER)
    assert _state(db, case) == ("draft", SOURCE_LOST_NOTE)
    assert _state(db, single) is None


def test_a_conversation_hidden_from_the_web_loses_its_sources(db, org, fake_embedder) -> None:
    ids = _messages(
        db, org, fake_embedder, _msg(1, "ピントが合いません", 0), _msg(2, "延長筒を付けて", 1, "2")
    )
    case = _article(db, org, SourceRef(channel_message_id=ids[1]))
    draft = _article(db, org, SourceRef(channel_message_id=ids[2]), verified=False)
    key = conversation_key(10, None, NOW)
    assert delete_conversation(db, org_id=org, external_id=key, now=LATER) == 2
    assert _state(db, case) == ("draft", SOURCE_LOST_NOTE)
    assert _state(db, draft) is None
    # The hidden rows stay as markers but no source points at them any more.
    assert db.scalars(select(KnowledgeSource.channel_message_id)).all() == []


def test_an_edited_message_sends_its_knowledge_back(db, org, fake_embedder) -> None:
    ids = _messages(db, org, fake_embedder, _msg(1, "延長筒を付けて", 0))
    case = _article(db, org, SourceRef(channel_message_id=ids[1]))
    _messages(db, org, fake_embedder, _msg(1, "延長筒を付けて", 0))  # replayed unchanged
    assert _state(db, case) == ("verified", None)
    edited = _msg(1, "延長筒は右のポケットです", 0)
    _messages(db, org, fake_embedder, edited)
    assert _state(db, case) == ("draft", SOURCE_EDITED_NOTE)

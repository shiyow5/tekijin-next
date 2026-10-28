"""Integration tests for ingested source items and their chunks (#70, R2).

Everything TEKIJIN ingests — a note added with ``/tekijin add``, later a Discord
channel or a shared document — is one ``source_items`` row whose body is split into
embedded ``chunks``. Re-sending the same body must not re-embed it.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import func, select, text

from tekijin.data.db import get_sessionmaker
from tekijin.data.sources import (
    content_hash,
    delete_source_item,
    list_source_items,
    search_chunks,
    upsert_source_item,
)
from tekijin.models.tables import Chunk, Employee, Organization, SourceItem

NOW = dt.datetime(2026, 10, 28, 12, 0, 0)


@pytest.fixture
def org_and_member(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        org = Organization(name="テストサークル", discord_guild_id="777")
        other = Organization(name="ほかの組織", discord_guild_id="888")
        session.add_all([org, other])
        session.flush()
        member = Employee(name="会計係", org_id=org.id)
        session.add(member)
        session.commit()
        ids = (org.id, other.id, member.id)
    yield ids
    with factory() as session:
        session.execute(text("DELETE FROM chunks"))
        session.execute(text("DELETE FROM source_items"))
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


def _add(session, embedder, org_id, *, external_id="note:1", body="部室 鍵 守衛室 借りる", **kw):
    return upsert_source_item(
        session,
        embedder,
        org_id=org_id,
        provider="manual",
        external_id=external_id,
        title=kw.get("title", "部室の使い方"),
        body=body,
        url=kw.get("url"),
        author_id=kw.get("author_id"),
        now=NOW,
    )


def _chunk_count(session, item_id: int) -> int:
    return session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.source_item_id == item_id)
    )


def test_a_new_item_is_stored_with_embedded_chunks(engine, fake_embedder, org_and_member):
    org_id, _, member_id = org_and_member
    with get_sessionmaker(engine)() as session:
        result = _add(session, fake_embedder, org_id, author_id=member_id)
        session.commit()
        assert result.status == "added"
        item = session.get(SourceItem, result.item_id)
        assert item is not None
        assert (item.org_id, item.provider, item.author_id) == (org_id, "manual", member_id)
        chunks = session.scalars(select(Chunk).where(Chunk.source_item_id == item.id)).all()
        assert [c.seq for c in chunks] == [0]
        assert chunks[0].org_id == org_id
        assert chunks[0].embedding is not None


def test_resending_the_same_body_changes_nothing(engine, fake_embedder, org_and_member):
    org_id, _, _ = org_and_member
    with get_sessionmaker(engine)() as session:
        first = _add(session, fake_embedder, org_id)
        session.commit()
        chunk_ids = session.scalars(select(Chunk.id)).all()
        again = _add(session, fake_embedder, org_id)
        session.commit()
        assert again.status == "unchanged"
        assert again.item_id == first.item_id
        assert session.scalars(select(Chunk.id)).all() == chunk_ids


def test_a_changed_body_replaces_the_chunks(engine, fake_embedder, org_and_member):
    org_id, _, _ = org_and_member
    with get_sessionmaker(engine)() as session:
        first = _add(session, fake_embedder, org_id)
        session.commit()
        long_body = "\n\n".join(["合宿 宿 予約 " * 60, "会計 引継ぎ 通帳 " * 60])
        updated = _add(session, fake_embedder, org_id, body=long_body, title="合宿と会計")
        session.commit()
        assert updated.status == "updated"
        assert updated.item_id == first.item_id
        assert session.get(SourceItem, first.item_id).title == "合宿と会計"
        assert _chunk_count(session, first.item_id) == updated.chunks == 2


def test_search_finds_the_nearest_chunk_in_the_same_organization(
    engine, fake_embedder, org_and_member
):
    org_id, other_id, _ = org_and_member
    with get_sessionmaker(engine)() as session:
        _add(session, fake_embedder, org_id, external_id="note:keys")
        _add(
            session,
            fake_embedder,
            org_id,
            external_id="note:camp",
            title="合宿",
            body="合宿 宿 予約 方法",
        )
        _add(session, fake_embedder, other_id, external_id="note:keys")
        session.commit()
        query = fake_embedder.encode(["部室 鍵 借りる"])[0]
        hits = search_chunks(session, org_id=org_id, query_vec=query, top_k=5)
        assert [h.title for h in hits] == ["部室の使い方", "合宿"]
        assert hits[0].similarity > hits[1].similarity
        # The other organization's identical note never shows up.
        assert {h.source_item_id for h in hits} <= {
            i.id for i in session.scalars(select(SourceItem).where(SourceItem.org_id == org_id))
        }


def test_list_and_delete(engine, fake_embedder, org_and_member):
    org_id, other_id, _ = org_and_member
    with get_sessionmaker(engine)() as session:
        a = _add(session, fake_embedder, org_id, external_id="note:a", url="https://example.com/a")
        _add(session, fake_embedder, org_id, external_id="note:b", title="二つ目")
        session.commit()
        items = list_source_items(session, org_id=org_id)
        assert [i.title for i in items] == ["二つ目", "部室の使い方"]
        assert items[1].url == "https://example.com/a"
        assert items[1].chunks == 1
        # Another organization cannot delete it.
        assert delete_source_item(session, item_id=a.item_id, org_id=other_id) is False
        assert delete_source_item(session, item_id=a.item_id, org_id=org_id) is True
        session.commit()
        assert session.get(SourceItem, a.item_id) is None
        assert _chunk_count(session, a.item_id) == 0


def test_a_blank_body_is_rejected(engine, fake_embedder, org_and_member):
    org_id, _, _ = org_and_member
    with get_sessionmaker(engine)() as session, pytest.raises(ValueError):
        _add(session, fake_embedder, org_id, body=" \n ")


def test_a_concurrent_update_cannot_mix_two_requests(engine, fake_embedder, org_and_member):
    """Another request updates the item while this one is embedding (Codex #71)."""

    org_id, _, _ = org_and_member
    factory = get_sessionmaker(engine)
    with factory() as session:
        _add(session, fake_embedder, org_id, title="元の題名", body="元の 本文")
        session.commit()

    class RacingEmbedder:
        """Commits a title-only update from another session mid-embedding."""

        def encode(self, texts, *, kind="passage"):
            with factory() as other:
                _add(other, fake_embedder, org_id, title="別の題名", body="元の 本文")
                other.commit()
            return fake_embedder.encode(texts, kind=kind)

    with factory() as session:
        result = _add(session, RacingEmbedder(), org_id, title="元の題名", body="新しい 本文")
        session.commit()
        assert result.status == "updated"
        item = session.get(SourceItem, result.item_id)
        assert (item.title, item.body) == ("元の題名", "新しい 本文")
        assert item.content_hash == content_hash("元の題名", "新しい 本文")
        chunks = session.scalars(select(Chunk.text).where(Chunk.source_item_id == item.id)).all()
        assert chunks == ["新しい 本文"]

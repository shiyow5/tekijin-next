"""Verified knowledge as evidence for the self-answer (#118, R6).

Verified articles are offered to the composer next to the documents, and an answer
that uses one cites both the article and what it was made from.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text

from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import (
    ArticleDTO,
    SourceRef,
    create_draft,
    retire_article,
    verify_article,
)
from tekijin.data.visibility import PUBLIC
from tekijin.knowledge.answer import (
    KNOWLEDGE_KIND,
    KnowledgeCandidates,
    evidence_text,
    expand_citations,
    knowledge_article_id,
    knowledge_citation_id,
)
from tekijin.knowledge.index import embed_articles
from tekijin.models.tables import Answer, Employee, Organization, Question, SourceItem

NOW = dt.datetime(2026, 11, 5, 12, 0, 0)


def _article(aid: int, **fields) -> ArticleDTO:
    base = {
        "id": aid,
        "org_id": 1,
        "kind": "procedure",
        "issue": "部室の鍵はどこ",
        "environment": "合宿の前",
        "resolution": "部室の箱に入っている",
        "cause": None,
        "state": "verified",
        "reuse_count": 0,
        "demand_count": 0,
        "last_used_at": None,
        "superseded_by": None,
        "author_id": None,
        "verifier_id": None,
        "created_at": NOW,
        "updated_at": NOW,
        "verified_at": NOW,
    }
    return ArticleDTO(**{**base, **fields})


def test_evidence_text_labels_the_fields() -> None:
    body = evidence_text(_article(1, cause="戻し忘れ"))
    assert "問い: 部室の鍵はどこ" in body and "場面: 合宿の前" in body
    assert "答え: 部室の箱に入っている" in body and "原因: 戻し忘れ" in body
    assert "None" not in body


def test_evidence_text_has_no_company_wording() -> None:
    body = evidence_text(_article(1))
    for word in ("社内", "顧客", "課題:", "打ち手", "【"):
        assert word not in body


def test_citation_id_round_trips() -> None:
    assert knowledge_citation_id(7) == "ka_7"
    assert knowledge_article_id("ka_7") == 7
    for other in ("src_7", "ka_", "ka_x", "ku_7", "7"):
        assert knowledge_article_id(other) is None


def test_expand_puts_each_articles_sources_right_after_it() -> None:
    cited = [
        {"source_id": "src_1", "kind": "source", "title": "手引き"},
        {"source_id": "ka_5", "kind": KNOWLEDGE_KIND, "title": "鍵"},
    ]
    groups = {
        "ka_5": [
            {"source_id": "ka_5", "kind": KNOWLEDGE_KIND, "title": "鍵"},
            {"source_id": "src_1", "kind": "source", "title": "手引き"},
            {"source_id": "msg_9", "kind": "message", "title": "#連絡"},
        ]
    }
    assert expand_citations(cited, groups) == [
        {"source_id": "src_1", "kind": "source", "title": "手引き"},
        {"source_id": "ka_5", "kind": KNOWLEDGE_KIND, "title": "鍵"},
        {"source_id": "msg_9", "kind": "message", "title": "#連絡"},
    ]


def test_expand_keeps_an_article_whose_sources_are_unknown() -> None:
    cited = [{"source_id": "ka_5", "kind": KNOWLEDGE_KIND, "title": "鍵"}]
    assert expand_citations(cited, {}) == cited


# --------------------------------------------------------------------------- #
# against the database
# --------------------------------------------------------------------------- #
@pytest.fixture
def org(engine, seed_counts):
    factory = get_sessionmaker(engine)
    with factory() as session:
        row = Organization(name="天文サークル", discord_guild_id="991")
        other = Organization(name="ほか", discord_guild_id="992")
        session.add_all([row, other])
        session.flush()
        member = Employee(name="アリス", org_id=row.id)
        session.add(member)
        session.flush()
        item = SourceItem(
            org_id=row.id,
            provider="manual",
            external_id="manual:kb",
            title="合宿の手引き",
            url="https://ex.com/camp",
            body="鍵は部室の箱",
            content_hash="h",
            updated_at=NOW,
        )
        question = Question(id="q_kb_1", asker_id=member.id, body="鍵は？", created_at=NOW)
        session.add_all([item, question])
        session.flush()
        session.add(
            Answer(
                id="a_kb_1", question_id="q_kb_1", responder_id=member.id, body="箱", created_at=NOW
            )
        )
        session.commit()
        ids = {"org": row.id, "other": other.id, "item": item.id, "member": member.id}
    yield ids
    with factory() as session:
        session.execute(text("DELETE FROM knowledge_sources"))
        session.execute(text("DELETE FROM knowledge_articles"))
        session.execute(text("DELETE FROM answers WHERE id = 'a_kb_1'"))
        session.execute(text("DELETE FROM questions WHERE id = 'q_kb_1'"))
        session.execute(text("DELETE FROM source_items"))
        session.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


@pytest.fixture
def db(engine, org):
    factory = get_sessionmaker(engine)
    with factory() as session:
        yield session
        session.rollback()


def _verified(db, org_id, issue, resolution, sources=()) -> int:
    article = create_draft(
        db,
        org_id=org_id,
        kind="procedure",
        issue=issue,
        resolution=resolution,
        sources=sources,
        now=NOW,
    )
    verify_article(db, article.id, org_id=org_id, verifier_id=None, now=NOW)
    return article.id


def _query(fake_embedder, question: str) -> list[float]:
    return fake_embedder.encode([question])[0]


def test_search_offers_verified_knowledge_of_the_organization(db, org, fake_embedder) -> None:
    mine = _verified(db, org["org"], "部室 の 鍵 は どこ", "部室 の 箱")
    _verified(db, org["other"], "部室 の 鍵 は どこ", "よそ の 箱")
    create_draft(
        db, org_id=org["org"], kind="fact", issue="部室 の 鍵 は どこ", resolution="x", now=NOW
    )
    embed_articles(db, fake_embedder)

    found = KnowledgeCandidates(db, min_similarity=0.3).search(
        org["org"], _query(fake_embedder, "部室 の 鍵 は どこ"), PUBLIC
    )
    assert [h["article_id"] for h in found.hits] == [mine]
    assert found.confidence == pytest.approx(found.hits[0]["similarity"])
    assert found.confidence > 0.3
    assert found.draft_ids == []


def test_search_below_the_floor_offers_nothing_and_counts_drafts(db, org, fake_embedder) -> None:
    draft = create_draft(
        db, org_id=org["org"], kind="fact", issue="部室 の 鍵", resolution="箱", now=NOW
    )
    _verified(db, org["org"], "まったく 別 の 話", "x")
    embed_articles(db, fake_embedder)

    found = KnowledgeCandidates(db, min_similarity=0.5).search(
        org["org"], _query(fake_embedder, "部室 の 鍵"), PUBLIC
    )
    assert found.hits == [] and found.confidence == 0.0
    assert found.draft_ids == [draft.id]


def test_search_without_a_vector_offers_nothing(db, org) -> None:
    found = KnowledgeCandidates(db, min_similarity=0.0).search(org["org"], [], PUBLIC)
    assert found.hits == [] and found.draft_ids == []


def test_evidence_is_read_again_and_drops_a_retired_article(db, org, fake_embedder) -> None:
    kept = _verified(db, org["org"], "部室 の 鍵 は どこ", "部室 の 箱")
    gone = _verified(db, org["org"], "部室 の 鍵 の 予備", "顧問 の 机")
    embed_articles(db, fake_embedder)
    candidates = KnowledgeCandidates(db, min_similarity=0.0)
    found = candidates.search(org["org"], _query(fake_embedder, "部室 の 鍵"), PUBLIC)
    assert {h["article_id"] for h in found.hits} == {kept, gone}

    retire_article(db, gone, org_id=org["org"], now=NOW)
    evidence = candidates.evidence(found.hits, org["org"], PUBLIC)
    assert [e.source_id for e in evidence] == [f"ka_{kept}"]
    assert evidence[0].kind == KNOWLEDGE_KIND
    assert evidence[0].title == "部室 の 鍵 は どこ"
    assert "答え: 部室 の 箱" in evidence[0].text
    # Another organization reading the same hits gets nothing.
    assert candidates.evidence(found.hits, org["other"], PUBLIC) == []


def test_citations_name_the_article_and_its_sources(db, org, fake_embedder) -> None:
    article_id = _verified(
        db,
        org["org"],
        "部室 の 鍵 は どこ",
        "部室 の 箱",
        sources=[SourceRef(source_item_id=org["item"]), SourceRef(question_id="q_kb_1")],
    )
    groups = KnowledgeCandidates(db, min_similarity=0.0).citations([article_id], org["org"], PUBLIC)
    assert groups[f"ka_{article_id}"] == [
        {"source_id": f"ka_{article_id}", "kind": "knowledge", "title": "部室 の 鍵 は どこ"},
        {
            "source_id": f"src_{org['item']}",
            "kind": "source",
            "title": "合宿の手引き",
            "url": "https://ex.com/camp",
        },
        {"source_id": "a_kb_1", "kind": "qa"},
    ]
    # Another organization's reader gets no citations for this article.
    assert (
        KnowledgeCandidates(db, min_similarity=0.0).citations([article_id], org["other"], PUBLIC)
        == {}
    )

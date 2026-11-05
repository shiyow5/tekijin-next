"""Answering from verified knowledge (#112, R6).

The answer is built from the articles' own fields, with no LLM, and cites both the
article and what it was made from.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text

from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import ArticleDTO, SourceRef, create_draft, verify_article
from tekijin.data.visibility import PUBLIC
from tekijin.knowledge.answer import (
    answer_from_knowledge,
    compose_knowledge_answer,
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


def test_compose_nothing_is_not_an_answer() -> None:
    assert compose_knowledge_answer([]) is None


def test_compose_builds_the_answer_from_the_fields() -> None:
    answer = compose_knowledge_answer([_article(1), _article(2, cause="戻し忘れ")])
    assert answer is not None
    assert "問い: 部室の鍵はどこ" in answer and "場面: 合宿の前" in answer
    assert "答え: 部室の箱に入っている" in answer and "原因: 戻し忘れ" in answer
    assert "None" not in answer


def test_compose_has_no_company_wording() -> None:
    answer = compose_knowledge_answer([_article(1)])
    assert answer is not None
    for word in ("社内", "顧客", "課題:", "打ち手", "【"):
        assert word not in answer


def test_compose_skips_an_article_without_a_resolution() -> None:
    assert compose_knowledge_answer([_article(1, resolution=None)]) is None


def test_citation_id_is_prefixed() -> None:
    assert knowledge_citation_id(7) == "ka_7"


# --------------------------------------------------------------------------- #
# against the database
# --------------------------------------------------------------------------- #
@pytest.fixture
def org(engine, seed_counts):
    factory = get_sessionmaker(engine)
    with factory() as session:
        row = Organization(name="天文サークル", discord_guild_id="991")
        session.add(row)
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
        ids = {"org": row.id, "item": item.id, "member": member.id}
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


def test_answer_cites_the_article_and_its_sources(db, org, fake_embedder) -> None:
    article_id = _verified(
        db,
        org["org"],
        "部室 の 鍵 は どこ",
        "部室 の 箱",
        sources=[SourceRef(source_item_id=org["item"]), SourceRef(question_id="q_kb_1")],
    )
    embed_articles(db, fake_embedder)
    query = fake_embedder.encode(["部室 の 鍵 は どこ"])[0]
    result = answer_from_knowledge(
        db, org_id=org["org"], query_vec=query, scope=PUBLIC, min_similarity=0.3
    )
    assert result is not None
    assert result.article_ids == (article_id,)
    assert "答え: 部室 の 箱" in result.answer
    assert result.citations[0] == {
        "source_id": f"ka_{article_id}",
        "kind": "knowledge",
        "title": "部室 の 鍵 は どこ",
    }
    assert {
        "source_id": f"src_{org['item']}",
        "kind": "source",
        "title": "合宿の手引き",
        "url": "https://ex.com/camp",
    } in result.citations
    assert {"source_id": "a_kb_1", "kind": "qa"} in result.citations


def test_no_answer_below_the_floor_or_from_drafts(db, org, fake_embedder) -> None:
    create_draft(db, org_id=org["org"], kind="fact", issue="部室 の 鍵", resolution="箱", now=NOW)
    _verified(db, org["org"], "まったく 別 の 話", "x")
    embed_articles(db, fake_embedder)
    query = fake_embedder.encode(["部室 の 鍵"])[0]
    assert (
        answer_from_knowledge(
            db, org_id=org["org"], query_vec=query, scope=PUBLIC, min_similarity=0.5
        )
        is None
    )

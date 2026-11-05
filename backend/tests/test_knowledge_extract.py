"""Turning a record into a knowledge draft with an LLM (#112).

The extractor asks the model for one draft in the KCS shape and stores it with
:func:`tekijin.data.knowledge.create_draft`. The model is a fake here.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text

from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import SourceRef, article_sources
from tekijin.data.visibility import EVERYTHING
from tekijin.knowledge.extract import DraftExtractor, ExtractionSource, extract_and_store
from tekijin.llm.schemas import KnowledgeDraftSchema
from tekijin.models.tables import KnowledgeArticle, Organization, SourceItem

NOW = dt.datetime(2026, 11, 5, 12, 0, 0)


class _FakeModel:
    def __init__(self, schema: KnowledgeDraftSchema | None) -> None:
        self._schema = schema
        self.calls: list[list[tuple[str, str]]] = []

    def invoke(self, prompt: list[tuple[str, str]]) -> KnowledgeDraftSchema | None:
        self.calls.append(prompt)
        return self._schema


class _RaisingModel:
    def invoke(self, prompt):
        raise RuntimeError("model down")


def _schema(**fields) -> KnowledgeDraftSchema:
    base = {
        "extractable": True,
        "kind": "case",
        "issue": "部室の鍵が見つからない",
        "resolution": "部室の箱に入っている",
    }
    return KnowledgeDraftSchema(**{**base, **fields})


# --------------------------------------------------------------------------- #
# schema and prompt (no DB)
# --------------------------------------------------------------------------- #
def test_schema_needs_an_issue_when_extractable() -> None:
    assert KnowledgeDraftSchema(extractable=False).issue == ""
    with pytest.raises(ValueError):
        KnowledgeDraftSchema(extractable=True, issue="  ")
    # An answer is not required: a draft without one is stored as wip.
    assert KnowledgeDraftSchema(extractable=True, issue="問い").resolution is None


def test_schema_kind_is_one_of_the_kcs_kinds() -> None:
    with pytest.raises(ValueError):
        KnowledgeDraftSchema(extractable=True, issue="問い", kind="rumor")


def test_prompt_has_no_vocabulary_or_company_wording() -> None:
    source = ExtractionSource(org_id=1, text="質問: 鍵は？", sources=())
    system = DraftExtractor.prompt(source)[0][1]
    for word in ("社内", "顧客", "商材", "業種", "トピック", "CRM"):
        assert word not in system
    assert "resolution" in system and "issue" in system


def test_extract_returns_none_for_a_non_record() -> None:
    extractor = DraftExtractor(model=_FakeModel(KnowledgeDraftSchema(extractable=False)))
    assert extractor.extract(ExtractionSource(org_id=1, text="こんにちは", sources=())) is None


def test_an_empty_model_output_is_an_error() -> None:
    with pytest.raises(ValueError):
        DraftExtractor(model=_FakeModel(None)).extract(
            ExtractionSource(org_id=1, text="x", sources=())
        )


# --------------------------------------------------------------------------- #
# storing drafts
# --------------------------------------------------------------------------- #
@pytest.fixture
def org(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        row = Organization(name="天文サークル", discord_guild_id="551")
        session.add(row)
        session.flush()
        item = SourceItem(
            org_id=row.id,
            provider="manual",
            external_id="manual:x",
            title="手引き",
            body="鍵は箱",
            content_hash="h",
            updated_at=NOW,
        )
        session.add(item)
        session.commit()
        ids = {"org": row.id, "item": item.id}
    yield ids
    with factory() as session:
        session.execute(text("DELETE FROM knowledge_sources"))
        session.execute(text("DELETE FROM knowledge_articles"))
        session.execute(text("DELETE FROM source_items"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


@pytest.fixture
def db(engine, org):
    factory = get_sessionmaker(engine)
    with factory() as session:
        yield session
        session.rollback()


def test_extract_and_store_creates_drafts_with_sources(db, org) -> None:
    extractor = DraftExtractor(model=_FakeModel(_schema(environment="合宿前", cause="戻し忘れ")))
    source = ExtractionSource(
        org_id=org["org"],
        text="鍵は箱",
        sources=(SourceRef(source_item_id=org["item"], quote="鍵は箱"),),
    )
    report = extract_and_store(db, [source], extractor, now=NOW)
    assert (report.seen, report.stored, report.skipped, report.errored) == (1, 1, 0, 0)
    row = db.get(KnowledgeArticle, report.article_ids[0])
    assert (row.state, row.kind, row.environment, row.cause) == (
        "draft",
        "case",
        "合宿前",
        "戻し忘れ",
    )
    sources = article_sources(db, [row.id], org_id=org["org"], scope=EVERYTHING)[row.id]
    assert [s.quote for s in sources] == ["鍵は箱"]


def test_a_draft_without_an_answer_is_stored_as_wip(db, org) -> None:
    extractor = DraftExtractor(model=_FakeModel(_schema(resolution=None)))
    report = extract_and_store(
        db, [ExtractionSource(org_id=org["org"], text="x", sources=())], extractor, now=NOW
    )
    assert db.get(KnowledgeArticle, report.article_ids[0]).state == "wip"


def test_non_records_are_skipped_and_errors_isolated(db, org) -> None:
    good = DraftExtractor(model=_FakeModel(_schema()))
    skip = DraftExtractor(model=_FakeModel(KnowledgeDraftSchema(extractable=False)))
    bad = DraftExtractor(model=_RaisingModel())

    class _Dispatch:
        def extract(self, source: ExtractionSource):
            return {"good": good, "skip": skip, "bad": bad}[source.text].extract(source)

    sources = [
        ExtractionSource(org_id=org["org"], text=t, sources=()) for t in ("good", "skip", "bad")
    ]
    report = extract_and_store(db, sources, _Dispatch(), now=NOW)
    assert (report.seen, report.stored, report.skipped, report.errored) == (3, 1, 1, 1)
    assert len(report.article_ids) == 1

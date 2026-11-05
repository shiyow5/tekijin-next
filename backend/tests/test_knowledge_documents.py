"""Procedure and fact drafts from documents, with a fake model (#113)."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select, text

from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import (
    SOURCE_CHANGED_NOTE,
    article_sources,
    verify_article,
)
from tekijin.data.sources import MANUAL, upsert_source_item
from tekijin.data.visibility import EVERYTHING
from tekijin.knowledge.documents import (
    DocumentDraftExtractor,
    document_item_ids,
    draft_document,
)
from tekijin.llm.schemas import (
    MAX_DRAFTS_PER_PASSAGE,
    DocumentKnowledgeItem,
    DocumentKnowledgeSchema,
)
from tekijin.models.tables import KnowledgeArticle, Organization, SourceItem

NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
LATER = NOW + dt.timedelta(days=1)

MANUAL_TEXT = (
    "部室の鍵\n部室の鍵は守衛室で借ります。返却は午後9時までです。\n\n"
    "故障したとき\n壊れた機材は赤いテープを貼って棚の一番下に置きます。"
)


def _item(issue: str, resolution: str = "答え", kind: str = "fact") -> DocumentKnowledgeItem:
    return DocumentKnowledgeItem(kind=kind, issue=issue, resolution=resolution)


class _FakeModel:
    """Answers by the passage's first line; records every prompt."""

    def __init__(self, by_heading: dict[str, list[DocumentKnowledgeItem]] | None = None) -> None:
        self.by_heading = by_heading or {}
        self.prompts: list[list[tuple[str, str]]] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        body = prompt[1][1].split("本文:\n", 1)[1]
        return DocumentKnowledgeSchema(items=self.by_heading.get(body.splitlines()[0], []))


class _Raising:
    def invoke(self, prompt):
        raise RuntimeError("model down")


class _Empty:
    def invoke(self, prompt):
        return None


@pytest.fixture
def org(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        row = Organization(name="サークル", discord_guild_id="113000")
        session.add(row)
        session.commit()
        org_id = row.id
    yield org_id
    with factory() as session:
        for table in ("knowledge_sources", "knowledge_articles", "chunks", "source_items"):
            session.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed names
        session.execute(text("DELETE FROM organizations"))
        session.commit()


@pytest.fixture
def db(engine, org):
    factory = get_sessionmaker(engine)
    with factory() as session:
        yield session
        session.rollback()


def _store(db, org, embedder, body: str = MANUAL_TEXT, now: dt.datetime = NOW) -> int:
    return upsert_source_item(
        db,
        embedder,
        org_id=org,
        provider=MANUAL,
        external_id="circle:manual",
        title="機材マニュアル",
        body=body,
        url=None,
        author_id=None,
        now=now,
    ).item_id


def _articles(db, org) -> list[KnowledgeArticle]:
    db.expire_all()
    return list(
        db.scalars(
            select(KnowledgeArticle)
            .where(KnowledgeArticle.org_id == org)
            .order_by(KnowledgeArticle.id)
        )
    )


def _long_manual() -> str:
    # Two passages: each over half the chunk limit, so they are not packed together.
    keys = "部室の鍵\n" + "鍵は守衛室で借ります。" * 30
    broken = "故障したとき\n" + "赤いテープを貼ります。" * 30
    return f"{keys}\n\n{broken}"


def test_prompt_carries_the_title_and_the_passage() -> None:
    system, human = DocumentDraftExtractor.prompt(title="マニュアル", passage="鍵は箱")
    assert "procedure" in system[1] and "fact" in system[1]
    assert human[1] == "資料の題名: マニュアル\n\n本文:\n鍵は箱"
    assert DocumentDraftExtractor.prompt(title=None, passage="x")[1][1] == "本文:\nx"


def test_an_empty_model_output_is_an_error() -> None:
    with pytest.raises(ValueError):
        DocumentDraftExtractor(model=_Empty()).extract_passage(title=None, passage="x")


def test_one_document_gives_several_drafts_each_citing_its_passage(db, org, fake_embedder):
    item = _store(db, org, fake_embedder, _long_manual())
    model = _FakeModel(
        {
            "部室の鍵": [
                _item("部室の鍵はどこで借りる", "守衛室で借りる", "procedure"),
                _item("鍵の返却はいつまで", "午後9時まで"),
            ],
            "故障したとき": [_item("壊れた機材はどうする", "赤いテープを貼る", "procedure")],
        }
    )
    report = draft_document(
        db, org_id=org, item_id=item, extractor=DocumentDraftExtractor(model=model), now=NOW
    )
    assert (report.passages, report.stored, report.replaced, report.errored) == (2, 3, 0, False)
    rows = _articles(db, org)
    assert [(r.kind, r.issue, r.state) for r in rows] == [
        ("procedure", "部室の鍵はどこで借りる", "draft"),
        ("fact", "鍵の返却はいつまで", "draft"),
        ("procedure", "壊れた機材はどうする", "draft"),
    ]
    sources = article_sources(db, [r.id for r in rows], org_id=org, scope=EVERYTHING)
    assert {s.source_item_id for group in sources.values() for s in group} == {item}
    assert sources[rows[0].id][0].quote.startswith("部室の鍵")
    assert sources[rows[2].id][0].quote.startswith("故障したとき")
    assert len(model.prompts) == 2


def test_items_without_an_answer_or_past_the_limit_are_dropped(db, org, fake_embedder):
    item = _store(db, org, fake_embedder)
    many = [_item(f"問い{i}") for i in range(MAX_DRAFTS_PER_PASSAGE + 2)]
    model = _FakeModel({"部室の鍵": [_item("空の答え", " "), _item("  ", "答え"), *many]})
    report = draft_document(
        db, org_id=org, item_id=item, extractor=DocumentDraftExtractor(model=model), now=NOW
    )
    assert report.stored == MAX_DRAFTS_PER_PASSAGE - 2


def test_a_failing_model_changes_nothing(db, org, fake_embedder) -> None:
    item = _store(db, org, fake_embedder)
    good = DocumentDraftExtractor(model=_FakeModel({"部室の鍵": [_item("鍵はどこ")]}))
    draft_document(db, org_id=org, item_id=item, extractor=good, now=NOW)
    report = draft_document(
        db, org_id=org, item_id=item, extractor=DocumentDraftExtractor(model=_Raising()), now=NOW
    )
    assert report.errored and report.stored == 0
    assert [r.issue for r in _articles(db, org)] == ["鍵はどこ"]


def test_an_updated_document_is_drafted_again(db, org, fake_embedder) -> None:
    item = _store(db, org, fake_embedder)
    first = DocumentDraftExtractor(
        model=_FakeModel({"部室の鍵": [_item("鍵はどこ", "守衛室"), _item("返却は", "9時")]})
    )
    report = draft_document(db, org_id=org, item_id=item, extractor=first, now=NOW)
    verified_id = report.article_ids[0]
    verify_article(db, verified_id, org_id=org, verifier_id=None, now=NOW)

    # The text changes: the verified one goes back with a note (upsert does that).
    _store(db, org, fake_embedder, MANUAL_TEXT.replace("守衛室", "事務室"), now=LATER)
    second = DocumentDraftExtractor(
        model=_FakeModel(
            {"部室の鍵": [_item("鍵はどこで借りる", "事務室"), _item("返却は", "9時")]}
        )
    )
    report = draft_document(db, org_id=org, item_id=item, extractor=second, now=LATER)
    # The unreviewed "返却は" draft went with the old text, when it was stored.
    assert (report.stored, report.replaced) == (2, 0)
    rows = _articles(db, org)
    assert [(r.issue, r.resolution, r.state, r.review_note) for r in rows] == [
        ("鍵はどこ", "守衛室", "draft", SOURCE_CHANGED_NOTE),
        ("鍵はどこで借りる", "事務室", "draft", None),
        ("返却は", "9時", "draft", None),
    ]
    # Drafting the same text again replaces the unreviewed drafts, never duplicates.
    again = draft_document(db, org_id=org, item_id=item, extractor=second, now=LATER)
    assert (again.stored, again.replaced) == (2, 2)
    assert len(_articles(db, org)) == 3


def test_an_issue_already_kept_is_not_drafted_twice(db, org, fake_embedder) -> None:
    item = _store(db, org, fake_embedder)
    model = DocumentDraftExtractor(model=_FakeModel({"部室の鍵": [_item("鍵は どこ")]}))
    report = draft_document(db, org_id=org, item_id=item, extractor=model, now=NOW)
    verify_article(db, report.article_ids[0], org_id=org, verifier_id=None, now=NOW)
    again = draft_document(db, org_id=org, item_id=item, extractor=model, now=NOW)
    assert (again.stored, again.replaced) == (0, 0)
    # Once its source changed, the same question is drafted again from the new text.
    _store(db, org, fake_embedder, MANUAL_TEXT + "\n\n追記", now=LATER)
    changed = draft_document(db, org_id=org, item_id=item, extractor=model, now=LATER)
    assert changed.stored == 1


def test_conversations_and_unknown_items_give_nothing(db, org, fake_embedder) -> None:
    conversation = SourceItem(
        org_id=org,
        provider="discord",
        external_id="thread:1",
        title="#機材",
        body="鍵は?",
        content_hash="h",
        updated_at=NOW,
    )
    db.add(conversation)
    db.flush()
    model = _FakeModel({"鍵は?": [_item("鍵")]})
    extractor = DocumentDraftExtractor(model=model)
    for item_id in (conversation.id, -1):
        report = draft_document(db, org_id=org, item_id=item_id, extractor=extractor, now=NOW)
        assert (report.passages, report.stored) == (0, 0)
    assert model.prompts == []
    document = _store(db, org, fake_embedder)
    assert document_item_ids(db, org_id=org) == [document]
    assert document_item_ids(db, org_id=org, item_ids=[conversation.id]) == []


def test_a_document_changed_while_it_was_read_is_not_written(db, org, fake_embedder) -> None:
    """The text read is checked again before the drafts are stored (Codex)."""

    item = _store(db, org, fake_embedder)

    class _ChangesTheDocument:
        def invoke(self, prompt):
            db.get(SourceItem, item).content_hash = "newer"
            db.flush()
            return DocumentKnowledgeSchema(items=[_item("鍵はどこ")])

    report = draft_document(
        db,
        org_id=org,
        item_id=item,
        extractor=DocumentDraftExtractor(model=_ChangesTheDocument()),
        now=NOW,
    )
    assert report.outdated and report.stored == 0
    assert _articles(db, org) == []

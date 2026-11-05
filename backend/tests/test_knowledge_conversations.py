"""Case drafts from solved Discord conversations, with a fake model (#113)."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select, text

from tekijin.data.conversations import IncomingMessage, delete_messages, ingest_messages
from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import SOURCE_EDITED_NOTE, article_sources, retire_article
from tekijin.data.visibility import EVERYTHING
from tekijin.knowledge.conversations import (
    WINDOW_MAX,
    capture_conversation_case,
    closing_candidates,
    draft_conversation_case,
)
from tekijin.knowledge.extract import DraftExtractor
from tekijin.llm.schemas import KnowledgeDraftSchema
from tekijin.models.tables import ChannelMessage, Employee, KnowledgeArticle, Organization

GUILD = "113777"
NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
THREAD = 555


class _FakeModel:
    def __init__(self, kind: str = "fact", extractable: bool = True) -> None:
        self.kind = kind
        self.extractable = extractable
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt[1][1])
        return KnowledgeDraftSchema(
            extractable=self.extractable,
            kind=self.kind,
            issue="20cmの反射でピントが合わない",
            resolution="延長筒を付ける",
        )


@pytest.fixture
def org(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        row = Organization(
            name="サークル", discord_guild_id=GUILD, knowledge_from_conversations=True
        )
        session.add(row)
        session.commit()
        org_id = row.id
    yield org_id
    with factory() as session:
        for table in (
            "knowledge_sources",
            "knowledge_articles",
            "chunks",
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


def _msg(message_id: int, author: str, content: str, *, thread: int | None = THREAD):
    return IncomingMessage(
        message_id=message_id,
        channel_id=10,
        channel_name="機材",
        thread_id=thread,
        thread_name="ピント" if thread else None,
        author_discord_id=author,
        author_name={"1": "相川", "2": "岩田", "3": "上野"}.get(author, author),
        content=content,
        created_at=NOW + dt.timedelta(minutes=message_id),
        edited_at=None,
    )


FOCUS = [
    _msg(1, "1", "20センチの反射でピントが合いません"),
    _msg(2, "2", "延長筒を付けましたか"),
    _msg(3, "1", "付けたら合いました、ありがとうございます"),
]


def _ingest(db, org, embedder, messages):
    return ingest_messages(
        db, embedder, org=db.get(Organization, org), guild_id=GUILD, messages=messages, now=NOW
    )


def _cases(db, org) -> list[KnowledgeArticle]:
    db.expire_all()
    return list(
        db.scalars(
            select(KnowledgeArticle)
            .where(KnowledgeArticle.org_id == org)
            .order_by(KnowledgeArticle.id)
        )
    )


def _capture(db, org, model, message_id, reactor=None, *, check_setting=True):
    fn = capture_conversation_case if check_setting else draft_conversation_case
    return fn(
        db,
        org_id=org,
        message_id=message_id,
        extractor=DraftExtractor(model=model),
        now=NOW,
        reactor_discord_id=reactor,
    )


def test_the_askers_thanks_makes_a_case_citing_the_stretch(db, org, fake_embedder) -> None:
    result = _ingest(db, org, fake_embedder, FOCUS)
    assert result.new_messages == (1, 2, 3)
    assert closing_candidates(db, org_id=org, message_ids=result.new_messages) == [3]
    model = _FakeModel(kind="fact")
    report = _capture(db, org, model, 3)
    assert report is not None and report.stored == 1
    [case] = _cases(db, org)
    # A solved conversation is always a case, whatever kind the model chose.
    assert (case.kind, case.state, case.issue) == ("case", "draft", "20cmの反射でピントが合わない")
    responder = db.scalar(select(Employee.id).where(Employee.name == "岩田"))
    assert case.author_id == responder
    sources = article_sources(db, [case.id], org_id=org, scope=EVERYTHING)[case.id]
    assert [s.quote for s in sources] == [m.content for m in FOCUS]
    assert "相川: 20センチの反射でピントが合いません" in model.prompts[0]
    assert "#機材 › ピント" in model.prompts[0]
    # The same signal again reads nothing new.
    assert _capture(db, org, model, 3) is None
    assert len(_cases(db, org)) == 1


def test_thanks_from_someone_else_is_not_a_signal(db, org, fake_embedder) -> None:
    _ingest(
        db, org, fake_embedder, [*FOCUS[:2], _msg(3, "2", "どういたしまして、ありがとうございます")]
    )
    assert _capture(db, org, _FakeModel(), 3) is None
    assert _cases(db, org) == []


def test_a_thanks_with_a_follow_up_question_is_not_a_signal(db, org, fake_embedder) -> None:
    _ingest(
        db, org, fake_embedder, [*FOCUS[:2], _msg(3, "1", "ありがとうございます。もう一つ質問です")]
    )
    assert closing_candidates(db, org_id=org, message_ids=[3]) == []
    assert _capture(db, org, _FakeModel(), 3) is None


def test_one_person_alone_is_not_a_case(db, org, fake_embedder) -> None:
    _ingest(db, org, fake_embedder, [_msg(1, "1", "ピントが合わない"), _msg(2, "1", "直りました")])
    assert _capture(db, org, _FakeModel(), 2) is None


def test_a_check_mark_from_a_participant_makes_a_case(db, org, fake_embedder) -> None:
    _ingest(db, org, fake_embedder, [*FOCUS[:2], _msg(3, "3", "見てます")])
    # A bystander who never wrote in the conversation, and an unknown user, are ignored.
    _ingest(db, org, fake_embedder, [_msg(9, "4", "別の話", thread=999)])
    assert _capture(db, org, _FakeModel(), 2, reactor="4") is None
    assert _capture(db, org, _FakeModel(), 2, reactor="404") is None
    report = _capture(db, org, _FakeModel(), 2, reactor="2")
    assert report is not None and report.stored == 1
    [case] = _cases(db, org)
    sources = article_sources(db, [case.id], org_id=org, scope=EVERYTHING)[case.id]
    assert len(sources) == 2  # up to the reacted message


def test_a_channel_day_holds_several_cases(db, org, fake_embedder) -> None:
    day = [
        _msg(1, "1", "鍵はどこで借りますか", thread=None),
        _msg(2, "2", "守衛室です", thread=None),
        _msg(3, "1", "ありがとうございます", thread=None),
        _msg(4, "3", "延長筒はどこですか", thread=None),
        _msg(5, "2", "ケースの右ポケットです", thread=None),
        _msg(6, "3", "ありました、ありがとうございます", thread=None),
    ]
    _ingest(db, org, fake_embedder, day)
    model = _FakeModel()
    assert _capture(db, org, model, 3).stored == 1
    assert _capture(db, org, model, 6).stored == 1
    second = _cases(db, org)[1]
    sources = article_sources(db, [second.id], org_id=org, scope=EVERYTHING)[second.id]
    assert [s.quote for s in sources] == [m.content for m in day[3:]]
    assert "鍵はどこ" not in model.prompts[1]


def test_a_discarded_case_is_not_read_again(db, org, fake_embedder) -> None:
    _ingest(db, org, fake_embedder, FOCUS)
    _capture(db, org, _FakeModel(), 3)
    [case] = _cases(db, org)
    retire_article(db, case.id, org_id=org, now=NOW)
    assert _capture(db, org, _FakeModel(), 3, reactor="2") is None


def test_nothing_to_keep_stores_nothing(db, org, fake_embedder) -> None:
    _ingest(db, org, fake_embedder, FOCUS)
    report = _capture(db, org, _FakeModel(extractable=False), 3)
    assert report is not None and (report.stored, report.skipped) == (0, 1)
    assert _cases(db, org) == []


def test_the_setting_is_off_by_default(db, org, fake_embedder) -> None:
    db.get(Organization, org).knowledge_from_conversations = False
    db.flush()
    _ingest(db, org, fake_embedder, FOCUS)
    assert _capture(db, org, _FakeModel(), 3) is None
    # The eval path drafts without looking at the setting.
    assert _capture(db, org, _FakeModel(), 3, check_setting=False).stored == 1
    assert Organization(name="x").knowledge_from_conversations in (None, False)


def test_unknown_or_deleted_messages_give_nothing(db, org, fake_embedder) -> None:
    _ingest(db, org, fake_embedder, FOCUS)
    assert _capture(db, org, _FakeModel(), 404) is None
    delete_messages(db, fake_embedder, org_id=org, guild_id=GUILD, message_ids=[3], now=NOW)
    assert _capture(db, org, _FakeModel(), 3) is None
    assert closing_candidates(db, org_id=org, message_ids=[]) == []


def test_a_long_conversation_is_cut_to_its_end(db, org, fake_embedder) -> None:
    chatter = [_msg(i, "2" if i % 2 else "1", f"話{i}") for i in range(1, WINDOW_MAX + 10)]
    last = WINDOW_MAX + 10
    _ingest(db, org, fake_embedder, [*chatter, _msg(last, "2", "見てます")])
    _capture(db, org, _FakeModel(), last, reactor="1")
    [case] = _cases(db, org)
    sources = article_sources(db, [case.id], org_id=org, scope=EVERYTHING)[case.id]
    assert len(sources) == WINDOW_MAX
    assert sources[-1].quote == "見てます"


def test_a_message_edited_while_the_model_reads_drops_the_case(db, org, fake_embedder) -> None:
    """The stretch is checked again before the case is stored (Codex)."""

    _ingest(db, org, fake_embedder, FOCUS)

    class _EditsTheAnswer(_FakeModel):
        def invoke(self, prompt):
            out = super().invoke(prompt)
            row = db.scalars(select(ChannelMessage).where(ChannelMessage.message_id == 2)).one()
            row.content = "延長筒ではなく接眼部を回して"
            db.flush()
            return out

    report = _capture(db, org, _EditsTheAnswer(), 3)
    assert report is not None and (report.stored, report.outdated) == (0, 1)
    assert _cases(db, org) == []


def test_an_edit_after_the_case_marks_it_for_review(db, org, fake_embedder) -> None:
    _ingest(db, org, fake_embedder, FOCUS)
    _capture(db, org, _FakeModel(), 3)
    _ingest(db, org, fake_embedder, [_msg(2, "2", "延長筒は右のポケットです")])
    [case] = _cases(db, org)
    assert (case.state, case.review_note) == ("draft", SOURCE_EDITED_NOTE)

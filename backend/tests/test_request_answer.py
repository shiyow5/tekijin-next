"""An answer written in a Discord request thread (#114, ``knowledge/request_answer.py``).

Each answer the requested member writes becomes an ``answers`` row, and the
question's case draft follows the answers until an admin has checked it. A checked
(verified) article is never rewritten from the thread.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pytest
from sqlalchemy import select

from tekijin.data.db import get_sessionmaker
from tekijin.data.knowledge import article_sources, articles_from_question, verify_article
from tekijin.data.visibility import EVERYTHING
from tekijin.knowledge.request_answer import record_thread_answer
from tekijin.models.tables import Answer, Employee, Organization, Question

NOW = dt.datetime(2026, 11, 5, 12, 0)


@pytest.fixture
def db(engine, seed_counts) -> Iterator:
    session = get_sessionmaker(engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def _parties(db) -> dict:
    org = Organization(name="test org", discord_guild_id="5150")
    db.add(org)
    db.flush()
    asker = Employee(name="質問者", org_id=org.id)
    responder = Employee(name="回答者", org_id=org.id)
    db.add_all([asker, responder])
    db.flush()
    db.add(
        Question(
            id="rq_test_1",
            asker_id=asker.id,
            body="VPNが切れる",
            topics=["network"],
            created_at=NOW,
        )
    )
    db.flush()
    return {
        "question_id": "rq_test_1",
        "asker_id": asker.id,
        "responder_id": responder.id,
        "org_id": org.id,
    }


def test_answers_are_stored_and_the_case_draft_follows_them(db) -> None:
    parties = _parties(db)
    first = record_thread_answer(
        db, parties=parties, body="ルーターを再起動してください", embedding=None, now=NOW
    )
    second = record_thread_answer(
        db, parties=parties, body="直らなければ回線を確認", embedding=None, now=NOW
    )

    answers = db.scalars(select(Answer).where(Answer.question_id == "rq_test_1")).all()
    assert sorted(a.body for a in answers) == [
        "ルーターを再起動してください",
        "直らなければ回線を確認",
    ]
    assert {a.responder_id for a in answers} == {parties["responder_id"]}
    assert {a.topic for a in answers} == {"network"}

    drafts = articles_from_question(db, "rq_test_1")
    assert len(drafts) == 1 and first == second == drafts[0].id
    draft = drafts[0]
    assert (draft.kind, draft.state, draft.issue) == ("case", "draft", "VPNが切れる")
    assert "ルーターを再起動してください" in draft.resolution
    assert "直らなければ回線を確認" in draft.resolution
    assert draft.author_id == parties["responder_id"]
    sources = article_sources(db, [draft.id], org_id=parties["org_id"], scope=EVERYTHING)
    assert [s.question_id for s in sources[draft.id]] == ["rq_test_1"]


def test_a_verified_article_is_left_alone(db) -> None:
    parties = _parties(db)
    article_id = record_thread_answer(db, parties=parties, body="答え", embedding=None, now=NOW)
    verify_article(db, article_id, org_id=parties["org_id"], verifier_id=None, now=NOW)

    again = record_thread_answer(db, parties=parties, body="補足", embedding=None, now=NOW)
    assert again is None
    assert articles_from_question(db, "rq_test_1")[0].resolution == "答え"
    # The answer itself is still kept.
    assert db.scalar(select(Answer.id).where(Answer.body == "補足")) is not None


def test_no_organization_keeps_only_the_answer(db, monkeypatch) -> None:
    monkeypatch.setattr("tekijin.knowledge.request_answer.thread_org_id", lambda *_: None)
    parties = _parties(db)
    assert record_thread_answer(db, parties=parties, body="答え", embedding=None, now=NOW) is None
    assert articles_from_question(db, "rq_test_1") == []
    assert db.scalar(select(Answer.id).where(Answer.body == "答え")) is not None

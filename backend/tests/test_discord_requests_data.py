"""The Discord request threads and the bot's outbox (#114, ``data/discord_requests.py``).

The API never talks to Discord: it queues what must appear in a request thread and
the bot delivers it. These tests pin the queue's rules without the bot: a request is
created once per recommendation, a post waits until its thread exists, a delivery
result fills in the thread, and an item that keeps failing stops being offered.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pytest

from tekijin.data.db import get_sessionmaker
from tekijin.data.discord_requests import (
    MAX_ATTEMPTS,
    POST,
    REQUEST,
    create_request,
    enqueue_post,
    pending_outbox,
    record_delivery,
    request_for_recommendation,
    request_for_thread,
)
from tekijin.models.tables import Question, Recommendation

NOW = dt.datetime(2026, 11, 5, 12, 0)


@pytest.fixture
def db(engine, seed_counts) -> Iterator:
    """A session whose writes are all rolled back."""

    session = get_sessionmaker(engine)()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def _rec(db, n: int = 1) -> int:
    db.add(Question(id=f"dq_test_{n}", asker_id=1, body="質問", created_at=NOW))
    db.flush()
    rec = Recommendation(question_id=f"dq_test_{n}", employee_id=2, rank=1)
    db.add(rec)
    db.flush()
    return rec.id


def test_a_request_is_created_once_and_queues_its_thread(db) -> None:
    rec = _rec(db)
    assert create_request(db, recommendation_id=rec, session_id="discord_x", payload={"a": 1})
    assert not create_request(db, recommendation_id=rec, session_id="discord_x", payload={"a": 2})

    items = pending_outbox(db, limit=10)
    assert [(i.kind, i.recommendation_id, i.thread_id, i.payload) for i in items] == [
        (REQUEST, rec, None, {"a": 1})
    ]
    assert request_for_recommendation(db, rec).session_id == "discord_x"


def test_a_post_waits_for_its_thread(db) -> None:
    rec = _rec(db)
    assert not enqueue_post(db, rec, "まだ依頼が無い")  # no request, nothing to post into
    create_request(db, recommendation_id=rec, session_id="discord_x", payload={})
    assert enqueue_post(db, rec, "承諾しました")
    assert [i.kind for i in pending_outbox(db, limit=10)] == [REQUEST]

    request_item = pending_outbox(db, limit=10)[0]
    assert record_delivery(db, request_item.id, ok=True, channel_id="77", thread_id="88", now=NOW)
    items = pending_outbox(db, limit=10)
    assert [(i.kind, i.thread_id, i.payload) for i in items] == [
        (POST, "88", {"text": "承諾しました"})
    ]
    assert request_for_thread(db, "88").recommendation_id == rec
    assert request_for_thread(db, "99") is None


def test_a_request_delivery_needs_its_thread(db) -> None:
    rec = _rec(db)
    create_request(db, recommendation_id=rec, session_id="discord_x", payload={})
    item = pending_outbox(db, limit=10)[0]
    assert not record_delivery(db, item.id, ok=True, channel_id=None, thread_id=None, now=NOW)
    assert [i.id for i in pending_outbox(db, limit=10)] == [item.id]


def test_failures_stop_after_the_limit(db) -> None:
    rec = _rec(db)
    create_request(db, recommendation_id=rec, session_id="discord_x", payload={})
    item = pending_outbox(db, limit=10)[0]
    for _ in range(MAX_ATTEMPTS):
        assert pending_outbox(db, limit=10)
        record_delivery(db, item.id, ok=False, now=NOW)
    assert pending_outbox(db, limit=10) == []


def test_a_delivered_item_is_not_offered_again(db) -> None:
    rec = _rec(db)
    create_request(db, recommendation_id=rec, session_id="discord_x", payload={})
    item = pending_outbox(db, limit=10)[0]
    record_delivery(db, item.id, ok=True, channel_id="77", thread_id="88", now=NOW)
    assert not record_delivery(db, item.id, ok=True, channel_id="77", thread_id="88", now=NOW)
    assert pending_outbox(db, limit=10) == []
    assert not record_delivery(db, 999999, ok=True, now=NOW)

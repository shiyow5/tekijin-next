"""Finding who can answer from the organization's evidence, on a real database (#107).

Every test runs inside one transaction that is rolled back, so nothing it writes
outlives it and no shared table needs cleaning.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid

import pytest
from sqlalchemy import update

from tekijin.api.events import node_event
from tekijin.channels.ask_summary import summarize_events
from tekijin.data.conversations import IncomingMessage, ingest_messages
from tekijin.data.db import get_sessionmaker
from tekijin.data.members import resolve_discord_member
from tekijin.data.self_intros import set_join_intro
from tekijin.data.sources import upsert_source_item
from tekijin.data.visibility import EVERYTHING, PUBLIC, SourceScope, record_visibility
from tekijin.discord_bot.format import format_reply
from tekijin.models.tables import Answer, Employee, Organization, Question
from tekijin.retrieval.bm25_cache import clear_bm25_cache
from tekijin.retrieval.experts import (
    ANSWER,
    ANSWER_REASON,
    ANSWER_TOP_K,
    CONVERSATION,
    DOCUMENT,
    FLOORS,
    INTRO_TOP_K,
    SELF_INTRO,
    DbExpertFinder,
)
from tekijin.retrieval.sources import ChunkSearch, source_citation_id

NOW = dt.datetime(2026, 11, 5, 12, 0, 0)
GUILD = "107107107"
OPEN_CHANNEL, SECRET_CHANNEL = 601, 602
QUERY = "望遠鏡 赤道儀 修理"
NO_FLOOR = dict.fromkeys(FLOORS, 0.0)


@pytest.fixture
def session(engine):
    clear_bm25_cache()
    factory = get_sessionmaker(engine)
    with factory() as session:
        yield session
        session.rollback()
    clear_bm25_cache()


@pytest.fixture
def org(session) -> Organization:
    org = Organization(name="推薦のテスト", discord_guild_id=GUILD)
    session.add(org)
    session.flush()
    return org


def _member(session, org: Organization, discord_id: str, name: str) -> Employee:
    return resolve_discord_member(session, org, discord_id, display_name=name, now=NOW)


def _find(session, embedder, org, *, scope=PUBLIC, hits_scope=None, asker=None, exclude=(), **kw):
    """C4's search, then the finder, as ``c4_retrieve`` runs them."""

    vector = embedder.encode([QUERY])[0]
    found = ChunkSearch(session).search(QUERY, org.id, vector, hits_scope or scope)
    finder = DbExpertFinder(session, floors=kw.pop("floors", NO_FLOOR), **kw)
    return finder.find(
        org_id=org.id,
        question=QUERY,
        query_vector=vector,
        scope=scope,
        source_hits=found.hits,
        asker_id=asker,
        exclude=exclude,
    )


def _doc(session, embedder, org, *, author, title="機材の使い方と管理マニュアル", body=QUERY):
    return upsert_source_item(
        session,
        embedder,
        org_id=org.id,
        provider="manual",
        external_id=f"note:{uuid.uuid4().hex}",
        title=title,
        body=body,
        url=None,
        author_id=author,
        now=NOW,
    )


def _msg(message_id: int, content: str, author: str, name: str, channel=OPEN_CHANNEL):
    return IncomingMessage(
        message_id=message_id,
        channel_id=channel,
        channel_name="機材" if channel == OPEN_CHANNEL else "幹部",
        thread_id=None,
        thread_name=None,
        author_discord_id=author,
        author_name=name,
        content=content,
        created_at=dt.datetime(2026, 11, 1, 3, 0) + dt.timedelta(minutes=message_id),
        edited_at=None,
    )


def _talk(session, embedder, org, channel=OPEN_CHANNEL):
    ingest_messages(
        session,
        embedder,
        org=org,
        guild_id=GUILD,
        messages=[
            _msg(1, "望遠鏡 赤道儀 修理 どうすれば", "1", "質問 する人", channel),
            _msg(
                2,
                "望遠鏡 赤道儀 修理 は 延長筒 を 付けて ネジ を 締めます",
                "2",
                "岩田 颯太",
                channel,
            ),
            _msg(3, "修理 ありがとう", "3", "小野 咲", channel),
        ],
        now=NOW,
    )
    record_visibility(
        session, org.id, channels={OPEN_CHANNEL: True, SECRET_CHANNEL: False}, threads={}, now=NOW
    )
    session.flush()


def _ids(result) -> list[int]:
    return [e.member_id for e in result.experts]


# --------------------------------------------------------------------------- #
# The four kinds of evidence
# --------------------------------------------------------------------------- #
def test_a_self_introduction_credits_its_member(session, fake_embedder, org) -> None:
    iwata = _member(session, org, "2", "岩田 颯太")
    _member(session, org, "3", "小野 咲")
    set_join_intro(session, fake_embedder, member=iwata, text=QUERY + " が 得意", now=NOW)
    session.flush()

    result = _find(session, fake_embedder, org)

    assert _ids(result) == [iwata.id]
    reason = result.experts[0].reasons[0]
    assert reason == {"type": SELF_INTRO, "detail": f"自己紹介: {QUERY} が 得意"}
    assert result.confidence > 0.5


def test_a_document_credits_its_author_with_a_cited_reason(session, fake_embedder, org) -> None:
    iwata = _member(session, org, "2", "岩田 颯太")
    added = _doc(session, fake_embedder, org, author=iwata.id)
    session.flush()

    result = _find(session, fake_embedder, org)

    assert _ids(result) == [iwata.id]
    assert result.experts[0].reasons[0] == {
        "type": DOCUMENT,
        "detail": "『機材の使い方と管理マニュアル』を書いた",
        "source_id": source_citation_id(added.item_id),
    }


def test_a_document_without_an_author_credits_nobody(session, fake_embedder, org) -> None:
    _member(session, org, "2", "岩田 颯太")
    _doc(session, fake_embedder, org, author=None)
    session.flush()

    assert _find(session, fake_embedder, org).experts == []


def test_the_intro_slides_credit_the_people_not_their_author(session, fake_embedder, org) -> None:
    author = _member(session, org, "1", "蒼井 陽菜")
    iwata = _member(session, org, "2", "岩田 颯太")
    _doc(
        session,
        fake_embedder,
        org,
        author=author.id,
        title="2026年度 自己紹介",
        body=f"スライド1 岩田 颯太\n{QUERY}\n",
    )
    session.flush()

    result = _find(session, fake_embedder, org)

    assert _ids(result) == [iwata.id]
    assert {r["type"] for r in result.experts[0].reasons} == {SELF_INTRO}


def test_conversations_count_only_when_the_organization_allows_it(
    session, fake_embedder, org
) -> None:
    _talk(session, fake_embedder, org)

    assert _find(session, fake_embedder, org).experts == []

    session.execute(
        update(Organization).where(Organization.id == org.id).values(expertise_from_messages=True)
    )
    result = _find(session, fake_embedder, org)

    iwata = _member(session, org, "2", "岩田 颯太")
    asker = _member(session, org, "1", "質問 する人")
    ono = _member(session, org, "3", "小野 咲")
    # Credit follows each speaker's share of the conversation's characters.
    assert _ids(result) == [iwata.id, asker.id, ono.id]
    reason = result.experts[0].reasons[0]
    assert reason["type"] == CONVERSATION
    assert reason["detail"].endswith(" の会話で発言した")
    assert reason["detail"].startswith("#機材")


def test_the_settings_override_is_honoured(session, fake_embedder, org) -> None:
    _talk(session, fake_embedder, org)

    assert _find(session, fake_embedder, org, expertise_from_messages=True).experts
    session.execute(
        update(Organization).where(Organization.id == org.id).values(expertise_from_messages=True)
    )
    assert _find(session, fake_embedder, org, expertise_from_messages=False).experts == []


def test_a_private_conversation_neither_credits_nor_is_cited(session, fake_embedder, org) -> None:
    _talk(session, fake_embedder, org, channel=SECRET_CHANNEL)
    session.execute(
        update(Organization).where(Organization.id == org.id).values(expertise_from_messages=True)
    )

    # Even if a caller passed hits found under a wider scope, the reader's scope wins.
    hidden = _find(session, fake_embedder, org, scope=PUBLIC, hits_scope=EVERYTHING)
    seen = _find(
        session, fake_embedder, org, scope=SourceScope(channels=frozenset({SECRET_CHANNEL}))
    )

    assert hidden.experts == []
    assert hidden.confidence == 0.0
    assert seen.experts


PRIVATE_QUESTION = "赤道儀が動かないのは私が壊したからです"


def _answer(session, embedder, *, asker: Employee, responder: Employee, body=QUERY) -> str:
    qid = f"q_{uuid.uuid4().hex}"
    session.add(Question(id=qid, asker_id=asker.id, body=PRIVATE_QUESTION, created_at=NOW))
    session.flush()
    session.add(
        Answer(
            id=f"a_{uuid.uuid4().hex}",
            question_id=qid,
            responder_id=responder.id,
            body=body,
            embedding=embedder.encode([body])[0],
        )
    )
    session.flush()
    return qid


def test_a_tekijin_answer_credits_its_responder(session, fake_embedder, org) -> None:
    asker = _member(session, org, "1", "質問 する人")
    iwata = _member(session, org, "2", "岩田 颯太")
    _answer(session, fake_embedder, asker=asker, responder=iwata)

    result = _find(session, fake_embedder, org)

    assert _ids(result) == [iwata.id]
    assert result.experts[0].reasons[0] == {"type": ANSWER, "detail": ANSWER_REASON}


def test_another_members_question_is_never_quoted(session, fake_embedder, org) -> None:
    """Questions come from another member's /ask and may be private: neither the
    question, the answer nor the question's id may reach a different asker."""

    asker = _member(session, org, "1", "質問 する人")
    iwata = _member(session, org, "2", "岩田 颯太")
    secret_answer = QUERY + " 内緒 の 回答"
    qid = _answer(session, fake_embedder, asker=asker, responder=iwata, body=secret_answer)
    _answer(session, fake_embedder, asker=asker, responder=iwata, body=QUERY)

    rec = _find(session, fake_embedder, org, asker=None).experts[0].as_recommendation()
    sse = node_event("c6_score", {"recommendations": [rec]})
    assert sse is not None
    payload = json.loads(sse.data or "{}")
    outcome = summarize_events([("recommend", payload)])
    reply = format_reply(outcome.model_dump(), web_url="https://tekijin.example/")

    # Two answers, one fixed reason.
    assert rec["reasons"] == [{"type": ANSWER, "detail": ANSWER_REASON}]
    for text in (json.dumps(rec, ensure_ascii=False), sse.data or "", reply):
        assert PRIVATE_QUESTION[:4] not in text
        assert "内緒" not in text
        assert qid not in text
    assert ANSWER_REASON in reply


def test_answers_by_people_left_out_do_not_crowd_out_an_eligible_one(
    session, fake_embedder, org
) -> None:
    asker = _member(session, org, "1", "質問 する人")
    gone = _member(session, org, "2", "去った 人")
    iwata = _member(session, org, "3", "岩田 颯太")
    qid = "q_window"
    session.add(Question(id=qid, asker_id=asker.id, body="赤道儀が動かない", created_at=NOW))
    session.flush()
    vector = fake_embedder.encode([QUERY])[0]
    # Same distance for all; ties break on the id, so the eligible one comes 21st.
    session.add_all(
        [
            Answer(id=f"a_{i:02d}", question_id=qid, responder_id=gone.id, embedding=vector)
            for i in range(ANSWER_TOP_K)
        ]
    )
    session.add(Answer(id="a_99", question_id=qid, responder_id=iwata.id, embedding=vector))
    gone.is_active = False
    session.flush()

    assert _ids(_find(session, fake_embedder, org)) == [iwata.id]


def test_answers_to_another_organization_do_not_count(session, fake_embedder, org) -> None:
    other = Organization(name="ほか", discord_guild_id="107000")
    session.add(other)
    session.flush()
    outsider = _member(session, other, "9", "外部 の人")
    iwata = _member(session, org, "2", "岩田 颯太")
    _answer(session, fake_embedder, asker=outsider, responder=iwata)

    assert _find(session, fake_embedder, org).experts == []


# --------------------------------------------------------------------------- #
# Who is never recommended
# --------------------------------------------------------------------------- #
def test_the_asker_opted_out_inactive_and_excluded_are_left_out(
    session, fake_embedder, org
) -> None:
    people = [_member(session, org, str(i), f"メンバー {i}") for i in range(1, 6)]
    for person in people:
        set_join_intro(session, fake_embedder, member=person, text=QUERY, now=NOW)
    asker, opted_out, gone, declined, kept = people
    session.flush()
    opted_out.opted_out_at = NOW
    gone.is_active = False
    session.flush()

    result = _find(session, fake_embedder, org, asker=asker.id, exclude=[declined.id])

    assert _ids(result) == [kept.id]


def test_the_askers_own_introductions_do_not_crowd_out_others(session, fake_embedder, org) -> None:
    """One member may have many introductions (one per Discord post): the asker's
    must not fill the window the others are read from."""

    asker = _member(session, org, "1", "質問 する人")
    iwata = _member(session, org, "2", "岩田 颯太")
    ingest_messages(
        session,
        fake_embedder,
        org=org,
        guild_id=GUILD,
        messages=[
            IncomingMessage(
                message_id=1000 + i,
                channel_id=OPEN_CHANNEL,
                channel_name="自己紹介",
                thread_id=None,
                thread_name=None,
                author_discord_id="1",
                author_name="質問 する人",
                content=f"{QUERY} {i}",
                created_at=dt.datetime(2026, 11, 1, 3, 0) + dt.timedelta(minutes=i),
                edited_at=None,
            )
            for i in range(INTRO_TOP_K + 1)
        ],
        now=NOW,
    )
    record_visibility(session, org.id, channels={OPEN_CHANNEL: True}, threads={}, now=NOW)
    set_join_intro(
        session, fake_embedder, member=iwata, text=f"{QUERY} が 得意 です 本当に", now=NOW
    )
    session.flush()

    assert _ids(_find(session, fake_embedder, org, asker=asker.id)) == [iwata.id]


def test_describe_only_returns_members_who_may_be_recommended(session, fake_embedder, org) -> None:
    kept = _member(session, org, "1", "残る 人")
    gone = _member(session, org, "2", "去った 人")
    gone.is_active = False
    session.flush()

    described = DbExpertFinder(session).describe(org.id, [kept.id, gone.id])

    assert [e.member_id for e in described] == [kept.id]
    assert described[0].as_recommendation()["reasons"] == []


def test_an_unrelated_question_recommends_nobody_with_the_floors(
    session, fake_embedder, org
) -> None:
    iwata = _member(session, org, "2", "岩田 颯太")
    set_join_intro(session, fake_embedder, member=iwata, text="合宿 バス 手配", now=NOW)
    session.flush()

    floors = dict.fromkeys(FLOORS, 0.3)
    assert _find(session, fake_embedder, org, floors=floors).experts == []

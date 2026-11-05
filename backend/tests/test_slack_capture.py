"""Solve-capture (#476 Screen 02): ✅ reaction on a hand-off thread → knowledge draft.

Unit tests drive the reaction dispatcher's gating with a fake ``schedule_solve_capture``
(no DB, no thread); integration tests seed a resolved hand-off and exercise the thread
assembler + the capture orchestration against the live database with a fake extractor.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

from tekijin.api import slack_routes
from tekijin.config import Settings
from tekijin.data.db import get_sessionmaker, session_scope
from tekijin.data.slack_channel_links import create_channel_link
from tekijin.data.slack_links import upsert_slack_link
from tekijin.knowledge.extract import DraftExtractor
from tekijin.knowledge.slack_thread import slack_thread_source
from tekijin.llm.schemas import KnowledgeDraftSchema
from tekijin.models.tables import Answer, Employee, Organization, Question, Recommendation
from tekijin.slack.capture import capture_resolved_thread, thread_draft

NOW = dt.datetime(2026, 8, 27, 12, 0, 0)  # noqa: DTZ001 - naive, matches created_at


# --------------------------------------------------------------------------- #
# reaction dispatcher gating (no DB, no network)
# --------------------------------------------------------------------------- #
def _capture_calls(monkeypatch, *, enabled: bool) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(slack_routes, "schedule_solve_capture", lambda _sf, **kw: calls.append(kw))
    monkeypatch.setattr(
        slack_routes, "get_settings", lambda: SimpleNamespace(slack_solve_capture_enabled=enabled)
    )
    return calls


def _reaction_event(reaction: str = "white_check_mark", **over) -> dict:
    event = {
        "type": "reaction_added",
        "reaction": reaction,
        "user": "U_REACTOR",
        "item": {"type": "message", "channel": "C_THREAD", "ts": "1.0"},
    }
    event.update(over)
    return event


def test_slack_thread_uses_the_knowledge_extraction_prompt() -> None:
    # A resolved thread is free-form conversational text, read with the same prompt
    # as every other record (#476 security review, #112).
    from tekijin.knowledge.extract import _SYSTEM_PROMPT
    from tekijin.knowledge.extract import ExtractionSource as _Source

    source = _Source(org_id=1, text="質問: a\n回答: b", sources=())
    assert DraftExtractor.prompt(source)[0][1] == _SYSTEM_PROMPT


def test_reaction_schedules_capture_when_enabled(monkeypatch) -> None:
    calls = _capture_calls(monkeypatch, enabled=True)
    slack_routes._handle_reaction_event(object(), _reaction_event())
    assert calls == [
        {"channel_id": "C_THREAD", "message_ts": "1.0", "reactor_slack_user_id": "U_REACTOR"}
    ]


def test_reaction_is_noop_when_flag_off(monkeypatch) -> None:
    calls = _capture_calls(monkeypatch, enabled=False)
    slack_routes._handle_reaction_event(object(), _reaction_event())
    assert calls == []


def test_reaction_ignores_non_solve_emoji(monkeypatch) -> None:
    calls = _capture_calls(monkeypatch, enabled=True)
    slack_routes._handle_reaction_event(object(), _reaction_event(reaction="tada"))
    assert calls == []


def test_reaction_ignores_other_event_types(monkeypatch) -> None:
    calls = _capture_calls(monkeypatch, enabled=True)
    slack_routes._handle_reaction_event(object(), _reaction_event(type="reaction_removed"))
    assert calls == []


def test_reaction_requires_channel_and_user(monkeypatch) -> None:
    calls = _capture_calls(monkeypatch, enabled=True)
    slack_routes._handle_reaction_event(object(), _reaction_event(item={"channel": ""}))
    slack_routes._handle_reaction_event(object(), _reaction_event(user=""))
    assert calls == []


# --------------------------------------------------------------------------- #
# thread assembler + capture orchestration (live DB)
# --------------------------------------------------------------------------- #
# Ids far outside any fixture range to avoid collisions in the shared DB.
ASKER, RESPONDER, BYSTANDER = 940_301, 940_302, 940_303
CHANNEL, TEAM = "C_SLKCAP", "T_SLKCAP"
TOPICS = ["ネットワーク・VPN"]


@pytest.fixture
def _resolved_thread(engine, seed_counts):
    """Seed one accepted hand-off with an answer + a pair-channel link; commit so a
    separate session (the capture's own) can read it. Returns (factory, thread_id)."""

    factory = get_sessionmaker(engine)
    with session_scope(factory) as session:
        org = Organization(name="捕捉の会", discord_guild_id="slkcap")
        session.add(org)
        session.flush()
        session.add_all(
            [
                Employee(id=ASKER, name="質問者", email="asker@slkcap", org_id=org.id),
                Employee(id=RESPONDER, name="回答者", email="responder@slkcap", org_id=org.id),
                Employee(id=BYSTANDER, name="第三者", email="bystander@slkcap", org_id=org.id),
            ]
        )
        session.add(
            Question(id="slkcap_q", asker_id=ASKER, body="VPNの拠点間接続が切れる", topics=TOPICS)
        )
        rec = Recommendation(
            question_id="slkcap_q", employee_id=RESPONDER, rank=1, outcome="accepted"
        )
        session.add(rec)
        session.flush()  # assign rec.id (the thread id)
        thread_id = rec.id
        session.add(
            Answer(
                id="slkcap_a",
                question_id="slkcap_q",
                responder_id=RESPONDER,
                body="MTUを1400に下げると安定します",
            )
        )
        create_channel_link(
            session,
            ASKER,
            RESPONDER,
            thread_id=thread_id,
            slack_channel_id=CHANNEL,
            slack_team_id=TEAM,
            now=NOW,
        )
        upsert_slack_link(session, ASKER, slack_user_id="U_ASKER", slack_team_id=TEAM, now=NOW)
        upsert_slack_link(session, RESPONDER, slack_user_id="U_RESP", slack_team_id=TEAM, now=NOW)
        upsert_slack_link(session, BYSTANDER, slack_user_id="U_BYSTD", slack_team_id=TEAM, now=NOW)
    yield factory, thread_id
    # Clean up so the committed rows do not leak into other tests on the shared DB.
    with session_scope(factory) as session:
        from tekijin.models.tables import (
            KnowledgeArticle,
            Message,
            SlackChannelLink,
            SlackLink,
            SlackMessageAnchor,
        )

        org_id = session.query(Organization.id).filter_by(discord_guild_id="slkcap").scalar()
        session.query(KnowledgeArticle).filter(KnowledgeArticle.org_id == org_id).delete()
        session.query(SlackMessageAnchor).filter_by(slack_channel_id=CHANNEL).delete()
        session.query(Message).filter(Message.recommendation_id == thread_id).delete()
        session.query(Answer).filter(Answer.id == "slkcap_a").delete()
        session.query(Recommendation).filter(Recommendation.question_id == "slkcap_q").delete()
        session.query(Question).filter(Question.id == "slkcap_q").delete()
        session.query(SlackChannelLink).filter(
            SlackChannelLink.slack_channel_id == CHANNEL
        ).delete()
        session.query(SlackLink).filter(
            SlackLink.employee_id.in_([ASKER, RESPONDER, BYSTANDER])
        ).delete()
        session.query(Employee).filter(Employee.id.in_([ASKER, RESPONDER, BYSTANDER])).delete()
        session.query(Organization).filter_by(discord_guild_id="slkcap").delete()


def _extractor(*, extractable: bool = True) -> DraftExtractor:
    schema = KnowledgeDraftSchema(
        extractable=extractable,
        kind="case",
        issue="VPNの拠点間接続が切れる" if extractable else "",
        resolution="MTUを1400に下げると安定する" if extractable else None,
    )

    class _FakeModel:
        def invoke(self, _prompt):
            return schema

    return DraftExtractor(model=_FakeModel())


def _settings() -> Settings:
    return Settings(_env_file=None, slack_solve_capture_enabled=True)  # type: ignore[call-arg]


def test_slack_thread_source_assembles_question_and_answer(_resolved_thread) -> None:
    factory, thread_id = _resolved_thread
    with session_scope(factory) as session:
        source = slack_thread_source(session, thread_id)
    assert source is not None
    assert source.org_id is not None  # the asker's organization
    assert [ref.question_id for ref in source.sources] == ["slkcap_q"]  # cites the question
    assert source.author_id == RESPONDER
    assert "VPNの拠点間接続が切れる" in source.text  # 質問
    assert "MTUを1400に下げると安定します" in source.text  # 回答 (captured answer body)


def test_slack_thread_source_none_for_unknown_thread(_resolved_thread) -> None:
    factory, _thread_id = _resolved_thread
    with session_scope(factory) as session:
        assert slack_thread_source(session, 9_999_999) is None


def test_capture_stores_unreviewed_draft(_resolved_thread) -> None:
    factory, thread_id = _resolved_thread
    stored = capture_resolved_thread(
        factory,
        message_ts=None,
        channel_id=CHANNEL,
        reactor_slack_user_id="U_RESP",
        extractor=_extractor(),
        settings=_settings(),
    )
    assert stored == f"slack_thread_{thread_id}"
    with session_scope(factory) as session:
        unit = thread_draft(session, thread_id)
    assert unit is not None
    assert unit.state == "draft"  # lands in the draft box
    assert unit.kind == "case"


def test_a_second_reaction_does_not_add_a_second_draft(_resolved_thread) -> None:
    from tekijin.data.knowledge import articles_from_question

    factory, thread_id = _resolved_thread
    for reactor in ("U_RESP", "U_ASKER"):
        capture_resolved_thread(
            factory,
            message_ts=None,
            channel_id=CHANNEL,
            reactor_slack_user_id=reactor,
            extractor=_extractor(),
            settings=_settings(),
        )
    with session_scope(factory) as session:
        assert len(articles_from_question(session, "slkcap_q")) == 1


def test_discarding_a_draft_without_an_answer_retires_it(_resolved_thread) -> None:
    factory, thread_id = _resolved_thread
    with session_scope(factory) as session:
        from tekijin.knowledge.extract import DraftExtractor, extract_and_store
        from tekijin.knowledge.slack_thread import slack_thread_source

        schema = KnowledgeDraftSchema(extractable=True, kind="case", issue="VPNが切れる")

        class _NoAnswer:
            def invoke(self, _prompt):
                return schema

        extract_and_store(
            session,
            [slack_thread_source(session, thread_id)],
            DraftExtractor(model=_NoAnswer()),
            now=NOW,
        )
    with session_scope(factory) as session:
        from tekijin.slack.capture import discard_thread_draft

        assert discard_thread_draft(session, thread_id) is True
    with session_scope(factory) as session:
        assert thread_draft(session, thread_id).state == "retired"


def test_capture_noop_when_flag_off(_resolved_thread) -> None:
    factory, thread_id = _resolved_thread
    disabled = Settings(_env_file=None, slack_solve_capture_enabled=False)  # type: ignore[call-arg]
    stored = capture_resolved_thread(
        factory,
        message_ts=None,
        channel_id=CHANNEL,
        reactor_slack_user_id="U_RESP",
        extractor=_extractor(),
        settings=disabled,
    )
    assert stored is None
    with session_scope(factory) as session:
        assert thread_draft(session, thread_id) is None


def test_capture_rejects_non_participant_reactor(_resolved_thread) -> None:
    factory, thread_id = _resolved_thread
    # U_BYSTD is a linked employee but not the asker/responder of this thread.
    stored = capture_resolved_thread(
        factory,
        message_ts=None,
        channel_id=CHANNEL,
        reactor_slack_user_id="U_BYSTD",
        extractor=_extractor(),
        settings=_settings(),
    )
    assert stored is None
    with session_scope(factory) as session:
        assert thread_draft(session, thread_id) is None


def test_capture_noop_for_unknown_channel(_resolved_thread) -> None:
    factory, _thread_id = _resolved_thread
    stored = capture_resolved_thread(
        factory,
        message_ts=None,
        channel_id="C_NOT_OURS",
        reactor_slack_user_id="U_RESP",
        extractor=_extractor(),
        settings=_settings(),
    )
    assert stored is None


def test_capture_skips_when_model_declines(_resolved_thread) -> None:
    factory, thread_id = _resolved_thread
    stored = capture_resolved_thread(
        factory,
        message_ts=None,
        channel_id=CHANNEL,
        reactor_slack_user_id="U_RESP",
        extractor=_extractor(extractable=False),
        settings=_settings(),
    )
    assert stored is None  # model said "not a case" -> nothing stored
    with session_scope(factory) as session:
        assert thread_draft(session, thread_id) is None


# --------------------------------------------------------------------------- #
# per-message anchor + #508 reaction→thread attribution (live DB)
# --------------------------------------------------------------------------- #
def test_message_anchor_records_and_resolves(session) -> None:
    from tekijin.data.slack_message_anchors import record_message_anchor, thread_for_message

    assert thread_for_message(session, "C_A", "111.1") is None
    record_message_anchor(session, slack_channel_id="C_A", slack_ts="111.1", thread_id=7, now=NOW)
    session.flush()
    assert thread_for_message(session, "C_A", "111.1") == 7
    # Idempotent upsert: re-recording the same message updates in place.
    record_message_anchor(session, slack_channel_id="C_A", slack_ts="111.1", thread_id=9, now=NOW)
    session.flush()
    assert thread_for_message(session, "C_A", "111.1") == 9
    session.query(_anchor_model()).filter_by(slack_channel_id="C_A").delete()


def _anchor_model():
    from tekijin.models.tables import SlackMessageAnchor

    return SlackMessageAnchor


# Second pair for the channel-reuse scenario.
A2, R2 = 940_311, 940_312
CH2, TEAM2 = "C_SLKREUSE", "T_SLKREUSE"


@pytest.fixture
def _two_threads_one_channel(engine, seed_counts):
    """Same pair, TWO accepted hand-offs sharing one reused channel. Anchors an old
    message to thread #1 while current_thread_id points at #2 — the #508 setup.
    Returns (factory, thread1_id, thread2_id, old_message_ts)."""

    from tekijin.data.slack_channel_links import get_channel_link
    from tekijin.data.slack_message_anchors import record_message_anchor

    factory = get_sessionmaker(engine)
    old_ts = "1000.0001"
    with session_scope(factory) as session:
        org = Organization(name="再利用の会", discord_guild_id="slkreuse")
        session.add(org)
        session.flush()
        session.add_all(
            [
                Employee(id=A2, name="質問者2", email="asker2@slkreuse", org_id=org.id),
                Employee(id=R2, name="回答者2", email="responder2@slkreuse", org_id=org.id),
            ]
        )
        # Thread #1 + its answer (the genuinely-resolved conversation).
        session.add(
            Question(id="reuse_q1", asker_id=A2, body="質問1: DNSが引けない", topics=TOPICS)
        )
        rec1 = Recommendation(question_id="reuse_q1", employee_id=R2, rank=1, outcome="accepted")
        session.add(rec1)
        session.flush()
        t1 = rec1.id
        session.add(
            Answer(id="reuse_a1", question_id="reuse_q1", responder_id=R2, body="resolv.confを直す")
        )
        # Thread #2 (a LATER hand-off) reuses the channel; current_thread_id -> #2.
        session.add(
            Question(id="reuse_q2", asker_id=A2, body="質問2: 証明書が期限切れ", topics=TOPICS)
        )
        rec2 = Recommendation(question_id="reuse_q2", employee_id=R2, rank=1, outcome="accepted")
        session.add(rec2)
        session.flush()
        t2 = rec2.id
        session.add(
            Answer(id="reuse_a2", question_id="reuse_q2", responder_id=R2, body="証明書を再発行")
        )
        create_channel_link(
            session, A2, R2, thread_id=t1, slack_channel_id=CH2, slack_team_id=TEAM2, now=NOW
        )
        # Reuse: the channel now points at thread #2 (the pair's latest hand-off).
        get_channel_link(session, A2, R2).current_thread_id = t2
        # But an OLD message from thread #1 was mirrored back then, anchored to #1.
        record_message_anchor(session, slack_channel_id=CH2, slack_ts=old_ts, thread_id=t1, now=NOW)
        upsert_slack_link(session, R2, slack_user_id="U_R2", slack_team_id=TEAM2, now=NOW)
    yield factory, t1, t2, old_ts
    with session_scope(factory) as session:
        from tekijin.models.tables import (
            KnowledgeArticle,
            SlackChannelLink,
            SlackLink,
            SlackMessageAnchor,
        )

        org_id = session.query(Organization.id).filter_by(discord_guild_id="slkreuse").scalar()
        session.query(KnowledgeArticle).filter(KnowledgeArticle.org_id == org_id).delete()
        session.query(SlackMessageAnchor).filter_by(slack_channel_id=CH2).delete()
        session.query(Answer).filter(Answer.id.in_(["reuse_a1", "reuse_a2"])).delete()
        session.query(Recommendation).filter(
            Recommendation.question_id.in_(["reuse_q1", "reuse_q2"])
        ).delete()
        session.query(Question).filter(Question.id.in_(["reuse_q1", "reuse_q2"])).delete()
        session.query(SlackChannelLink).filter(SlackChannelLink.slack_channel_id == CH2).delete()
        session.query(SlackLink).filter(SlackLink.employee_id == R2).delete()
        session.query(Employee).filter(Employee.id.in_([A2, R2])).delete()
        session.query(Organization).filter_by(discord_guild_id="slkreuse").delete()


def test_capture_uses_message_anchor_not_current_thread(_two_threads_one_channel) -> None:
    # #508: reacting ✅ on thread #1's OLD message must capture thread #1, even though
    # the channel's current_thread_id now points at the later thread #2.
    factory, t1, t2, old_ts = _two_threads_one_channel
    stored = capture_resolved_thread(
        factory,
        message_ts=old_ts,
        channel_id=CH2,
        reactor_slack_user_id="U_R2",
        extractor=_extractor(),
        settings=_settings(),
    )
    assert stored == f"slack_thread_{t1}"  # the reacted-on thread, NOT t2
    with session_scope(factory) as session:
        assert thread_draft(session, t1) is not None
        assert thread_draft(session, t2) is None


def test_capture_falls_back_to_current_thread_without_anchor(_two_threads_one_channel) -> None:
    # A ✅ on a message with NO anchor (e.g. posted before capture was on) falls back
    # to current_thread_id = thread #2 (best-effort) rather than doing nothing.
    factory, _t1, t2, _old_ts = _two_threads_one_channel
    stored = capture_resolved_thread(
        factory,
        message_ts="9999.0000",  # unknown ts -> no anchor
        channel_id=CH2,
        reactor_slack_user_id="U_R2",
        extractor=_extractor(),
        settings=_settings(),
    )
    assert stored == f"slack_thread_{t2}"


# --------------------------------------------------------------------------- #
# message mirroring records a per-message anchor (wiring, live DB)
# --------------------------------------------------------------------------- #
def _message_event(
    ts: str, *, user: str = "U_RESP", text: str = "解決しました、ありがとうございます"
) -> dict:
    return {
        "type": "message",
        "channel": CHANNEL,
        "user": user,
        "text": text,
        "ts": ts,
    }


def test_message_event_records_anchor_when_capture_enabled(_resolved_thread, monkeypatch) -> None:
    from tekijin.data.slack_message_anchors import thread_for_message

    factory, thread_id = _resolved_thread
    monkeypatch.setattr(slack_routes, "get_settings", _settings)
    slack_routes._handle_message_event(factory, _message_event("ts_anchor_on"))
    with session_scope(factory) as session:
        assert thread_for_message(session, CHANNEL, "ts_anchor_on") == thread_id


def test_message_event_records_no_anchor_when_capture_disabled(
    _resolved_thread, monkeypatch
) -> None:
    from tekijin.data.slack_message_anchors import thread_for_message

    factory, _thread_id = _resolved_thread
    monkeypatch.setattr(
        slack_routes,
        "get_settings",
        lambda: Settings(_env_file=None, slack_solve_capture_enabled=False),  # type: ignore[call-arg]
    )
    slack_routes._handle_message_event(factory, _message_event("ts_anchor_off"))
    with session_scope(factory) as session:
        # The message is still mirrored, but no anchor is written (flag-off inert).
        assert thread_for_message(session, CHANNEL, "ts_anchor_off") is None


# --------------------------------------------------------------------------- #
# Slice B2: utterance detection + in-thread prompt + keep/discard (live DB)
# --------------------------------------------------------------------------- #
def test_is_solve_utterance_matches_resolution_not_generic_completion() -> None:
    from tekijin.knowledge.solve_signals import is_solve_utterance, is_thanks_utterance

    assert is_solve_utterance("おかげさまで解決しました")
    assert is_solve_utterance("MTUを下げたら動きました")
    assert is_solve_utterance("設定を直したらできるようになりました")
    # Generic completion remains excluded. Thanks has its own classifier because
    # only the Slack event handler knows whether the sender is the asker (#537).
    assert not is_solve_utterance("資料ができました")
    assert not is_solve_utterance("ありがとうございました")
    assert not is_solve_utterance("ありがとうございます")
    assert not is_solve_utterance("確認してみます")
    assert not is_solve_utterance("")
    assert is_thanks_utterance("ご回答ありがとうございました。")
    assert is_thanks_utterance("ありがとうございます！")
    assert not is_thanks_utterance("ありがとうございます。追加で質問があります")
    assert not is_thanks_utterance("承知しました")


def test_asker_thanks_schedules_prompt_but_responder_thanks_does_not(
    _resolved_thread, monkeypatch
) -> None:
    factory, thread_id = _resolved_thread
    calls: list[dict] = []
    monkeypatch.setattr(slack_routes, "get_settings", _settings)
    monkeypatch.setattr(slack_routes, "schedule_solve_prompt", lambda _sf, **kw: calls.append(kw))

    slack_routes._handle_message_event(
        factory,
        _message_event("ts_asker_thanks", user="U_ASKER", text="ありがとうございました"),
    )
    slack_routes._handle_message_event(
        factory,
        _message_event("ts_responder_thanks", user="U_RESP", text="ありがとうございました"),
    )

    assert calls == [{"channel_id": CHANNEL, "thread_id": thread_id}]


def test_capture_and_prompt_creates_draft_and_posts_prompt(_resolved_thread, monkeypatch) -> None:
    from tekijin.slack import capture as capture_mod

    factory, thread_id = _resolved_thread
    capture_mod._capture_progress_notified.discard(thread_id)
    posts: list[dict] = []
    monkeypatch.setattr(capture_mod, "post_message", lambda **kw: posts.append(kw))

    stored = capture_mod.capture_and_prompt(
        factory,
        channel_id=CHANNEL,
        thread_id=thread_id,
        extractor=_extractor(),
        settings=_settings(),
    )
    assert stored == f"slack_thread_{thread_id}"
    with session_scope(factory) as session:
        unit = thread_draft(session, thread_id)
    assert unit is not None and unit.state == "draft"
    # The immediate progress post precedes the prompt carrying the keep/discard actions.
    assert len(posts) == 2
    assert posts[0]["text"] == capture_mod.CAPTURE_STARTED_TEXT
    assert posts[1]["text"] == capture_mod.CAPTURE_READY_TEXT
    action_ids = [
        e["action_id"] for b in posts[1]["blocks"] if b["type"] == "actions" for e in b["elements"]
    ]
    assert set(action_ids) == {"tekijin_knowledge_keep", "tekijin_knowledge_discard"}


def test_capture_and_prompt_dedups_when_draft_exists(_resolved_thread, monkeypatch) -> None:
    from tekijin.slack import capture as capture_mod

    factory, thread_id = _resolved_thread
    capture_mod._capture_progress_notified.discard(thread_id)
    posts: list[dict] = []
    monkeypatch.setattr(capture_mod, "post_message", lambda **kw: posts.append(kw))
    capture_mod.capture_and_prompt(
        factory,
        channel_id=CHANNEL,
        thread_id=thread_id,
        extractor=_extractor(),
        settings=_settings(),
    )
    # A second solve utterance must NOT re-capture or re-prompt (draft already exists).
    again = capture_mod.capture_and_prompt(
        factory,
        channel_id=CHANNEL,
        thread_id=thread_id,
        extractor=_extractor(),
        settings=_settings(),
    )
    assert again is None
    assert len(posts) == 2  # only the first attempt's progress + prompt


def test_capture_and_prompt_notifies_when_no_case(_resolved_thread, monkeypatch) -> None:
    """#525: an explicit "解決" that yields no capturable case must not be silent.

    The model declines the record (``extractable=false``) so nothing is stored, but
    the solver still gets a short notice — never dead air — telling them why nothing
    was saved. It is a plain message, NOT the keep/discard prompt (there is no draft
    to keep)."""

    from tekijin.slack import capture as capture_mod

    factory, thread_id = _resolved_thread
    capture_mod._no_case_notified.discard(thread_id)  # isolate from other tests' state
    capture_mod._capture_progress_notified.discard(thread_id)
    posts: list[dict] = []
    monkeypatch.setattr(capture_mod, "post_message", lambda **kw: posts.append(kw))

    stored = capture_mod.capture_and_prompt(
        factory,
        channel_id=CHANNEL,
        thread_id=thread_id,
        extractor=_extractor(extractable=False),
        settings=_settings(),
    )
    assert stored is None
    with session_scope(factory) as session:
        unit = thread_draft(session, thread_id)
    assert unit is None  # nothing stored
    # Progress is closed by the plain no-case notice, which has no action buttons.
    assert len(posts) == 2
    assert posts[0]["text"] == capture_mod.CAPTURE_STARTED_TEXT
    assert not posts[1].get("blocks")
    assert posts[1]["text"] == capture_mod._NO_CASE_NOTICE


def test_capture_and_prompt_no_case_notice_dedups_per_thread(_resolved_thread, monkeypatch) -> None:
    """#525 review: repeated solve-utterances on a thread that never yields a case must
    not spam the no-case notice — at most one per thread, like the success prompt."""

    from tekijin.slack import capture as capture_mod

    factory, thread_id = _resolved_thread
    capture_mod._no_case_notified.discard(thread_id)
    capture_mod._capture_progress_notified.discard(thread_id)
    posts: list[dict] = []
    monkeypatch.setattr(capture_mod, "post_message", lambda **kw: posts.append(kw))

    for _ in range(3):
        again = capture_mod.capture_and_prompt(
            factory,
            channel_id=CHANNEL,
            thread_id=thread_id,
            extractor=_extractor(extractable=False),
            settings=_settings(),
        )
        assert again is None
    assert len(posts) == 2  # one progress + one no-case notice, then silent


def test_capture_and_prompt_notifies_on_extraction_error(_resolved_thread, monkeypatch) -> None:
    """#529: an extraction error closes progress with a retry notice, never the
    factually-wrong no-case notice, and a later utterance may retry."""

    from tekijin.slack import capture as capture_mod

    factory, thread_id = _resolved_thread
    capture_mod._no_case_notified.discard(thread_id)
    capture_mod._capture_progress_notified.discard(thread_id)
    posts: list[dict] = []
    monkeypatch.setattr(capture_mod, "post_message", lambda **kw: posts.append(kw))

    class _EmptyOutputModel:  # empty structured output -> extract() raises -> errored
        def invoke(self, _prompt):
            return None

    stored = capture_mod.capture_and_prompt(
        factory,
        channel_id=CHANNEL,
        thread_id=thread_id,
        extractor=DraftExtractor(model=_EmptyOutputModel()),
        settings=_settings(),
    )
    assert stored is None
    assert [post["text"] for post in posts] == [
        capture_mod.CAPTURE_STARTED_TEXT,
        capture_mod.CAPTURE_FAILED_TEXT,
    ]
    assert all(not post.get("blocks") for post in posts)
    assert thread_id not in capture_mod._no_case_notified  # not consumed by an error
    assert thread_id not in capture_mod._capture_progress_notified  # retry can narrate again


def test_knowledge_discard_marks_draft_rejected(_resolved_thread) -> None:
    factory, thread_id = _resolved_thread
    # Seed a draft first.
    with session_scope(factory) as session:
        from tekijin.knowledge.extract import extract_and_store
        from tekijin.knowledge.slack_thread import slack_thread_source

        extract_and_store(session, [slack_thread_source(session, thread_id)], _extractor(), now=NOW)

    resp = slack_routes._handle_knowledge_action(
        SimpleNamespace(session_factory=factory),
        "tekijin_knowledge_discard",
        {"thread_id": thread_id},
        "U_RESP",  # a party (the responder)
        None,
        [],
    )
    assert resp.status_code == 200
    with session_scope(factory) as session:
        unit = thread_draft(session, thread_id)
    assert unit is not None and unit.state == "retired"


def test_knowledge_keep_leaves_draft_unreviewed(_resolved_thread) -> None:
    factory, thread_id = _resolved_thread
    with session_scope(factory) as session:
        from tekijin.knowledge.extract import extract_and_store
        from tekijin.knowledge.slack_thread import slack_thread_source

        extract_and_store(session, [slack_thread_source(session, thread_id)], _extractor(), now=NOW)

    slack_routes._handle_knowledge_action(
        SimpleNamespace(session_factory=factory),
        "tekijin_knowledge_keep",
        {"thread_id": thread_id},
        "U_RESP",
        None,
        [],
    )
    with session_scope(factory) as session:
        unit = thread_draft(session, thread_id)
    assert unit is not None and unit.state == "draft"


def test_knowledge_keep_shows_review_blocks_in_slack(_resolved_thread, monkeypatch) -> None:
    """#527: 残す replaces the prompt with the draft's CONTENT + 承認する/破棄する — the
    review happens in Slack. Keep itself does not approve (status stays unreviewed)."""

    factory, thread_id = _resolved_thread
    with session_scope(factory) as session:
        from tekijin.knowledge.extract import extract_and_store
        from tekijin.knowledge.slack_thread import slack_thread_source

        extract_and_store(session, [slack_thread_source(session, thread_id)], _extractor(), now=NOW)

    sent: list[dict] = []
    monkeypatch.setattr(
        slack_routes, "respond_to_response_url", lambda _url, payload: sent.append(payload)
    )

    slack_routes._handle_knowledge_action(
        SimpleNamespace(session_factory=factory),
        "tekijin_knowledge_keep",
        {"thread_id": thread_id},
        "U_RESP",
        "https://hooks.slack.test/response",
        [],
    )
    assert len(sent) == 1
    blocks = sent[0]["blocks"]
    action_ids = [e["action_id"] for b in blocks if b["type"] == "actions" for e in b["elements"]]
    assert set(action_ids) == {"tekijin_knowledge_approve", "tekijin_knowledge_discard"}
    fields_text = " ".join(
        f["text"]
        for b in blocks
        if b.get("type") == "section" and "fields" in b
        for f in b["fields"]
    )
    assert "VPNの拠点間接続が切れる" in fields_text  # the extracted problem is shown
    with session_scope(factory) as session:
        unit = thread_draft(session, thread_id)
    assert unit.state == "draft"  # keep shows; approve is the next step


def test_knowledge_approve_marks_draft_approved_and_schedules_embed(_resolved_thread) -> None:
    """#527/#531: 承認する flips the draft to approved AND schedules an embed so the
    unit is immediately searchable by knowledge_answer (nothing else indexes it)."""

    factory, thread_id = _resolved_thread
    with session_scope(factory) as session:
        from tekijin.knowledge.extract import extract_and_store
        from tekijin.knowledge.slack_thread import slack_thread_source

        extract_and_store(session, [slack_thread_source(session, thread_id)], _extractor(), now=NOW)

    embeds: list[int] = []
    service = SimpleNamespace(
        session_factory=factory, schedule_knowledge_embed=lambda: embeds.append(1)
    )
    resp = slack_routes._handle_knowledge_action(
        service, "tekijin_knowledge_approve", {"thread_id": thread_id}, "U_RESP", None, []
    )
    assert resp.status_code == 200
    with session_scope(factory) as session:
        unit = thread_draft(session, thread_id)
    assert unit is not None and unit.state == "verified"
    assert embeds == [1]  # embed scheduled exactly once on approve


def test_knowledge_keep_and_discard_do_not_schedule_embed(_resolved_thread) -> None:
    """#531: only approve indexes; keep (just shows) and discard must NOT embed."""

    factory, thread_id = _resolved_thread
    with session_scope(factory) as session:
        from tekijin.knowledge.extract import extract_and_store
        from tekijin.knowledge.slack_thread import slack_thread_source

        extract_and_store(session, [slack_thread_source(session, thread_id)], _extractor(), now=NOW)

    embeds: list[int] = []
    service = SimpleNamespace(
        session_factory=factory, schedule_knowledge_embed=lambda: embeds.append(1)
    )
    for action in ("tekijin_knowledge_keep", "tekijin_knowledge_discard"):
        slack_routes._handle_knowledge_action(
            service, action, {"thread_id": thread_id}, "U_RESP", None, []
        )
    assert embeds == []


def test_knowledge_action_rejects_non_party(_resolved_thread) -> None:
    factory, thread_id = _resolved_thread
    with session_scope(factory) as session:
        from tekijin.knowledge.extract import extract_and_store
        from tekijin.knowledge.slack_thread import slack_thread_source

        extract_and_store(session, [slack_thread_source(session, thread_id)], _extractor(), now=NOW)

    # U_BYSTD is linked but not a party of this thread -> refused, draft untouched.
    resp = slack_routes._handle_knowledge_action(
        SimpleNamespace(session_factory=factory),
        "tekijin_knowledge_discard",
        {"thread_id": thread_id},
        "U_BYSTD",
        None,
        [],
    )
    assert resp.status_code == 200
    with session_scope(factory) as session:
        unit = thread_draft(session, thread_id)
    assert unit is not None and unit.state == "draft"  # NOT rejected


def test_message_event_solve_utterance_schedules_prompt(_resolved_thread, monkeypatch) -> None:
    factory, thread_id = _resolved_thread
    calls: list[dict] = []
    monkeypatch.setattr(slack_routes, "schedule_solve_prompt", lambda _sf, **kw: calls.append(kw))
    monkeypatch.setattr(slack_routes, "get_settings", _settings)
    event = {
        "type": "message",
        "channel": CHANNEL,
        "user": "U_RESP",
        "text": "おかげさまで解決しました！",
        "ts": "ts_solve_1",
    }
    slack_routes._handle_message_event(factory, event)
    assert calls == [{"channel_id": CHANNEL, "thread_id": thread_id}]


def test_message_event_non_solve_does_not_schedule_prompt(_resolved_thread, monkeypatch) -> None:
    factory, _thread_id = _resolved_thread
    calls: list[dict] = []
    monkeypatch.setattr(slack_routes, "schedule_solve_prompt", lambda _sf, **kw: calls.append(kw))
    monkeypatch.setattr(slack_routes, "get_settings", _settings)
    event = {
        "type": "message",
        "channel": CHANNEL,
        "user": "U_RESP",
        "text": "確認してみます",
        "ts": "ts_nonsolve_1",
    }
    slack_routes._handle_message_event(factory, event)
    assert calls == []


def test_interactivity_dispatch_routes_knowledge_button(_resolved_thread) -> None:
    import json as _json

    factory, thread_id = _resolved_thread
    with session_scope(factory) as session:
        from tekijin.knowledge.extract import extract_and_store
        from tekijin.knowledge.slack_thread import slack_thread_source

        extract_and_store(session, [slack_thread_source(session, thread_id)], _extractor(), now=NOW)

    raw = _json.dumps(
        {
            "actions": [
                {
                    "action_id": "tekijin_knowledge_discard",
                    "value": _json.dumps({"thread_id": thread_id}),
                }
            ],
            "user": {"id": "U_RESP"},
        }
    )
    resp = slack_routes._handle_interactivity_action(SimpleNamespace(session_factory=factory), raw)
    assert resp.status_code == 200
    with session_scope(factory) as session:
        unit = thread_draft(session, thread_id)
    assert unit is not None and unit.state == "retired"

"""The Discord notifier (#114): it queues posts only when Discord output is on.

The notifier never talks to Discord; it queues work for the bot. Off by default
(no output channel), it must not even open a database session. A question asked on
the web must never be posted to Discord.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tekijin.channels import discord_notifier
from tekijin.channels.discord_notifier import DiscordNotifier, accepted_text, relay_text
from tekijin.config import get_settings

_PARTIES = {"asker_id": 1, "responder_id": 2, "responder_name": "回答者"}


@pytest.fixture
def output_on(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEKIJIN_DISCORD_BOT_SECRET", "s")
    monkeypatch.setenv("TEKIJIN_DISCORD_GUILD_ID", "1")
    monkeypatch.setenv("TEKIJIN_DISCORD_OUTPUT_CHANNEL_ID", "2")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def output_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TEKIJIN_DISCORD_BOT_SECRET", "s")
    monkeypatch.setenv("TEKIJIN_DISCORD_GUILD_ID", "1")
    monkeypatch.delenv("TEKIJIN_DISCORD_OUTPUT_CHANNEL_ID", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _all(notifier: DiscordNotifier, factory, session_id: str = "discord_abc") -> None:
    notifier.relay_message(
        factory, employee_a=1, employee_b=2, sender_name="A", body="hi", thread_id=7
    )
    notifier.announce_handoff_request(
        factory,
        session_id=session_id,
        recommendation_id=7,
        thread_id=7,
        parties=_PARTIES,
        draft="d",
    )
    notifier.announce_handoff_accepted(factory, thread_id=7, parties=_PARTIES, draft="d")


def test_without_an_output_channel_nothing_is_touched(output_off) -> None:
    factory = MagicMock(side_effect=AssertionError("no session may be opened"))
    _all(DiscordNotifier(), factory)


def test_with_an_output_channel_posts_are_queued(output_on, monkeypatch) -> None:
    posts, requests = [], []
    monkeypatch.setattr(
        discord_notifier, "enqueue_post", lambda s, rec, text: posts.append((rec, text))
    )
    monkeypatch.setattr(
        discord_notifier, "send_request", lambda s, **kw: requests.append(kw) or True
    )
    _all(DiscordNotifier(), MagicMock())
    assert posts == [(7, relay_text("A", "hi")), (7, accepted_text("回答者"))]
    assert requests == [{"session_id": "discord_abc", "recommendation_id": 7, "draft": "d"}]


def test_a_web_question_is_never_sent_to_discord(output_on, monkeypatch) -> None:
    session = MagicMock()
    session.get.side_effect = AssertionError("a web hand-off must not be looked up")
    assert not discord_notifier.send_request(
        session, session_id="web-session-1", recommendation_id=7, draft="d"
    )
    monkeypatch.setattr(discord_notifier, "enqueue_post", lambda *a: None)
    sent = MagicMock()
    monkeypatch.setattr(discord_notifier, "send_request", sent)
    _all(DiscordNotifier(), MagicMock(), session_id="web-session-1")
    sent.assert_not_called()


def test_relay_without_a_thread_is_ignored(output_on) -> None:
    factory = MagicMock(side_effect=AssertionError("no session may be opened"))
    DiscordNotifier().relay_message(factory, employee_a=1, employee_b=2, sender_name="A", body="x")


def test_relay_text_is_bounded() -> None:
    assert len(relay_text("A", "x" * 5000)) <= 1900

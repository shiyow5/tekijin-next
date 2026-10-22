"""Unit tests for the chat-channel notifier seam (``tekijin.channels``, Discord D0).

The domain layer (``api/routes.py``, ``api/service.py``) calls a ``ChannelNotifier``
instead of ``tekijin.slack.notify`` directly, so a Discord implementation can be added
next to Slack. These tests pin the two things the seam must guarantee: the Slack
implementation forwards to the existing ``tekijin.slack.notify`` functions unchanged,
and the fan-out delivers to every channel even when one of them raises.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tekijin.channels.notifier import FanoutNotifier, SlackNotifier, get_notifier

_FACTORY = object()
_PARTIES = {"asker_id": 1, "responder_id": 2}


def _call_all(notifier) -> None:
    notifier.relay_message(_FACTORY, employee_a=1, employee_b=2, sender_name="A", body="hi")
    notifier.announce_handoff_request(
        _FACTORY,
        session_id="s1",
        recommendation_id=7,
        thread_id=7,
        parties=_PARTIES,
        draft="draft",
    )
    notifier.announce_handoff_accepted(_FACTORY, thread_id=7, parties=_PARTIES, draft="draft")


def test_slack_notifier_forwards_to_slack_notify(monkeypatch: pytest.MonkeyPatch) -> None:
    relay, pending, accepted = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr("tekijin.slack.notify.relay_to_channel", relay)
    monkeypatch.setattr("tekijin.slack.notify.schedule_pending_handoff", pending)
    monkeypatch.setattr("tekijin.slack.notify.schedule_channel_setup_and_draft", accepted)

    _call_all(SlackNotifier())

    relay.assert_called_once_with(_FACTORY, employee_a=1, employee_b=2, sender_name="A", body="hi")
    pending.assert_called_once_with(
        _FACTORY,
        session_id="s1",
        recommendation_id=7,
        thread_id=7,
        parties=_PARTIES,
        draft="draft",
    )
    accepted.assert_called_once_with(_FACTORY, thread_id=7, parties=_PARTIES, draft="draft")


def test_fanout_reaches_every_channel_even_if_one_raises() -> None:
    broken, healthy = MagicMock(), MagicMock()
    broken.relay_message.side_effect = RuntimeError("down")
    broken.announce_handoff_request.side_effect = RuntimeError("down")
    broken.announce_handoff_accepted.side_effect = RuntimeError("down")

    _call_all(FanoutNotifier((broken, healthy)))

    healthy.relay_message.assert_called_once()
    healthy.announce_handoff_request.assert_called_once()
    healthy.announce_handoff_accepted.assert_called_once()


def test_default_notifier_is_slack_only() -> None:
    notifier = get_notifier()
    assert isinstance(notifier, FanoutNotifier)
    assert [type(n) for n in notifier.notifiers] == [SlackNotifier]
    assert get_notifier() is notifier  # one shared instance


def test_domain_layer_no_longer_binds_slack_notify() -> None:
    from tekijin.api import routes, service

    names = ("relay_to_channel", "schedule_pending_handoff", "schedule_channel_setup_and_draft")
    for module in (routes, service):
        for name in names:
            assert not hasattr(module, name), f"{module.__name__} still binds {name}"

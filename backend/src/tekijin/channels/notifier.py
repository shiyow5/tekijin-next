"""The seam between the domain layer and chat channels (Discord D0, #58).

``api/routes.py`` and ``api/service.py`` used to import ``tekijin.slack.notify``
directly. They now call :func:`get_notifier`, so a Discord implementation can sit
next to Slack without the domain layer knowing which channels a user is linked to.

Every method is best-effort and returns immediately or soon after: the Slack
implementation keeps its existing contract (``relay_message`` runs as a FastAPI
background task, the two ``announce_*`` methods start their own daemon thread).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from tekijin.slack import notify as slack_notify

logger = logging.getLogger(__name__)


class ChannelNotifier(Protocol):
    def relay_message(
        self,
        session_factory: sessionmaker[Session],
        *,
        employee_a: int,
        employee_b: int,
        sender_name: str,
        body: str,
    ) -> None:
        """Mirror a TEKIJIN chat message into the pair's channel, if one exists."""

    def announce_handoff_request(
        self,
        session_factory: sessionmaker[Session],
        *,
        session_id: str,
        recommendation_id: int,
        thread_id: int,
        parties: dict,
        draft: str,
    ) -> None:
        """Post a sent chat hand-off with its accept/decline actions."""

    def announce_handoff_accepted(
        self,
        session_factory: sessionmaker[Session],
        *,
        thread_id: int,
        parties: dict,
        draft: str,
    ) -> None:
        """Tell both parties the hand-off was accepted (posting the draft if new)."""


class SlackNotifier:
    """Forwards to ``tekijin.slack.notify`` unchanged.

    The functions are looked up on the module at call time, so tests that patch
    ``tekijin.slack.notify.*`` keep working.
    """

    def relay_message(
        self,
        session_factory: sessionmaker[Session],
        *,
        employee_a: int,
        employee_b: int,
        sender_name: str,
        body: str,
    ) -> None:
        slack_notify.relay_to_channel(
            session_factory,
            employee_a=employee_a,
            employee_b=employee_b,
            sender_name=sender_name,
            body=body,
        )

    def announce_handoff_request(
        self,
        session_factory: sessionmaker[Session],
        *,
        session_id: str,
        recommendation_id: int,
        thread_id: int,
        parties: dict,
        draft: str,
    ) -> None:
        slack_notify.schedule_pending_handoff(
            session_factory,
            session_id=session_id,
            recommendation_id=recommendation_id,
            thread_id=thread_id,
            parties=parties,
            draft=draft,
        )

    def announce_handoff_accepted(
        self,
        session_factory: sessionmaker[Session],
        *,
        thread_id: int,
        parties: dict,
        draft: str,
    ) -> None:
        slack_notify.schedule_channel_setup_and_draft(
            session_factory, thread_id=thread_id, parties=parties, draft=draft
        )


class FanoutNotifier:
    """Delivers to every channel; one channel failing never blocks the others."""

    def __init__(self, notifiers: tuple[ChannelNotifier, ...]) -> None:
        self.notifiers = notifiers

    def _each(self, what: str, call: Callable[[ChannelNotifier], None]) -> None:
        for notifier in self.notifiers:
            try:
                call(notifier)
            except Exception:  # noqa: BLE001 - one channel must not break the rest
                logger.warning("%s failed on %s", what, type(notifier).__name__, exc_info=True)

    def relay_message(
        self,
        session_factory: sessionmaker[Session],
        *,
        employee_a: int,
        employee_b: int,
        sender_name: str,
        body: str,
    ) -> None:
        self._each(
            "relay_message",
            lambda n: n.relay_message(
                session_factory,
                employee_a=employee_a,
                employee_b=employee_b,
                sender_name=sender_name,
                body=body,
            ),
        )

    def announce_handoff_request(
        self,
        session_factory: sessionmaker[Session],
        *,
        session_id: str,
        recommendation_id: int,
        thread_id: int,
        parties: dict,
        draft: str,
    ) -> None:
        self._each(
            "announce_handoff_request",
            lambda n: n.announce_handoff_request(
                session_factory,
                session_id=session_id,
                recommendation_id=recommendation_id,
                thread_id=thread_id,
                parties=parties,
                draft=draft,
            ),
        )

    def announce_handoff_accepted(
        self,
        session_factory: sessionmaker[Session],
        *,
        thread_id: int,
        parties: dict,
        draft: str,
    ) -> None:
        self._each(
            "announce_handoff_accepted",
            lambda n: n.announce_handoff_accepted(
                session_factory, thread_id=thread_id, parties=parties, draft=draft
            ),
        )


@lru_cache(maxsize=1)
def get_notifier() -> FanoutNotifier:
    """The process-wide notifier. Slack only until the Discord adapter lands (D1+)."""

    return FanoutNotifier((SlackNotifier(),))

"""Deliver check-mark reactions after the message they are on (#113).

A check mark can mark a conversation solved, but the backend can only read the
conversation once the reacted message is stored. Messages reach the backend in
batches (``ingest.Batcher``), and not at all while the bot is reading history or
the backend is away. So a reaction is queued here and sent only after a
successful flush of the queued messages. The backend answers ``pending`` when it
does not have the message yet (still in history being read); such a reaction, and
one that could not be sent, is kept and tried again, up to :data:`MAX_TRIES`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

#: How many times one reaction is tried before it is dropped.
MAX_TRIES = 20
#: The most reactions kept waiting; the oldest go first when more arrive.
MAX_PENDING = 500

Send = Callable[[dict[str, Any]], Awaitable[str]]


class ReactionRelay:
    """Queues reactions and sends them once the messages before them are stored.

    ``send`` returns the backend's status: ``ok`` / ``ignored`` / ``unlinked`` are
    final, ``pending`` and anything else (``error``) are tried again.
    """

    def __init__(
        self, send: Send, flush: Callable[[], Awaitable[bool]], *, ready: asyncio.Event
    ) -> None:
        self._send = send
        self._flush = flush
        self._ready = ready
        self._pending: list[tuple[dict[str, Any], int]] = []
        self._lock = asyncio.Lock()

    @property
    def pending(self) -> int:
        return len(self._pending)

    def add(self, reaction: dict[str, Any]) -> None:
        self._pending.append((reaction, 0))
        if len(self._pending) > MAX_PENDING:
            dropped, _ = self._pending.pop(0)
            logger.warning("dropped a reaction on message %s", dropped.get("message_id"))

    async def deliver(self) -> None:
        """Send what is queued, if the queued messages could be flushed first."""

        async with self._lock:
            if not self._pending or not self._ready.is_set():
                return
            if not await self._flush():
                return
            batch, self._pending = self._pending, []
            kept: list[tuple[dict[str, Any], int]] = []
            for reaction, tries in batch:
                status = await self._send(reaction)
                if status in ("ok", "ignored", "unlinked"):
                    continue
                if tries + 1 < MAX_TRIES:
                    kept.append((reaction, tries + 1))
                else:
                    logger.warning("gave up a reaction on message %s", reaction.get("message_id"))
            # Reactions that arrived while these were being sent stay after them.
            self._pending = kept + self._pending

    async def run(self, interval: float) -> None:
        """Try again every ``interval`` seconds, forever."""

        while True:
            await asyncio.sleep(interval)
            await self.deliver()

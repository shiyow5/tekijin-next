"""The bot's side of channel ingestion (#76, R2), kept free of the Discord client so
it can be tested: what a message becomes, how messages are batched to the backend,
and how history is read after a restart.

Only conversation is forwarded — messages people wrote in the configured server.
Bots, DMs, system messages (pins, joins, …) and messages without text are skipped.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

# Message types that are someone talking; everything else is a system notice.
_CONVERSATION_TYPES = ("default", "reply")

SendMessages = Callable[[list[dict[str, Any]]], Awaitable[bool]]
SendDeletes = Callable[[list[int]], Awaitable[bool]]


def parse_ids(value: str) -> set[int]:
    """Comma-separated ids from a setting; anything not a number is ignored."""

    return {int(part) for part in (p.strip() for p in value.split(",")) if part.isdigit()}


def message_payload(message: Any) -> dict[str, Any] | None:
    """What the backend stores for one Discord message, or ``None`` to skip it."""

    if message.guild is None or message.author.bot:
        return None
    if message.type.name not in _CONVERSATION_TYPES:
        return None
    content = message.content.strip()
    if not content:
        return None
    channel = message.channel
    parent = getattr(channel, "parent", None)
    if parent is not None:  # a thread (or a forum post) inside a channel
        channel_id, channel_name = parent.id, parent.name
        thread_id, thread_name = channel.id, channel.name
    else:
        channel_id, channel_name = channel.id, channel.name
        thread_id, thread_name = None, None
    return {
        "message_id": message.id,
        "channel_id": channel_id,
        "channel_name": channel_name,
        "thread_id": thread_id,
        "thread_name": thread_name,
        "author_id": str(message.author.id),
        "author_name": message.author.display_name,
        "content": content,
        "created_at": message.created_at.isoformat(),
        "edited_at": message.edited_at.isoformat() if message.edited_at else None,
    }


def is_excluded(payload: dict[str, Any], excluded: set[int]) -> bool:
    return payload["channel_id"] in excluded or payload.get("thread_id") in excluded


class Batcher:
    """Collects messages and deletions and sends them to the backend in batches.

    Sending one request per message would rebuild a conversation for every line
    typed; batching every few seconds rebuilds it once. What fails to send stays
    queued for the next flush. The queue is bounded: if the backend is down for
    long, the oldest messages are dropped (a restart reads them back from history).
    """

    def __init__(
        self,
        send_messages: SendMessages,
        send_deletes: SendDeletes,
        *,
        max_batch: int = 50,
        max_pending: int = 5000,
    ) -> None:
        self._send_messages = send_messages
        self._send_deletes = send_deletes
        self._max_batch = max_batch
        self._messages: deque[dict[str, Any]] = deque(maxlen=max_pending)
        self._deletes: deque[int] = deque(maxlen=max_pending)
        self._lock = asyncio.Lock()

    @property
    def pending(self) -> int:
        return len(self._messages) + len(self._deletes)

    @property
    def max_batch(self) -> int:
        return self._max_batch

    def add(self, payload: dict[str, Any]) -> None:
        self._messages.append(payload)

    def delete(self, message_ids: Iterable[int]) -> None:
        self._deletes.extend(message_ids)

    async def flush(self) -> bool:
        """Send everything queued. False when the backend refused or was away.

        Deletions go after the messages, so a message posted and deleted within
        one interval is stored and then removed, not the other way round.
        """

        async with self._lock:
            while self._messages:
                batch = [
                    self._messages[i] for i in range(min(self._max_batch, len(self._messages)))
                ]
                if not await self._send_messages(batch):
                    return False
                for _ in batch:
                    self._messages.popleft()
            while self._deletes:
                ids = [self._deletes[i] for i in range(min(self._max_batch, len(self._deletes)))]
                if not await self._send_deletes(ids):
                    return False
                for _ in ids:
                    self._deletes.popleft()
            return True

    async def run(self, interval: float) -> None:
        """Flush every ``interval`` seconds, forever."""

        while True:
            await asyncio.sleep(interval)
            if not await self.flush():
                logger.warning("could not forward %d message change(s); will retry", self.pending)


async def backfill(
    channels: Sequence[Any],
    *,
    cursors: dict[str, int],
    batcher: Batcher,
    excluded: set[int],
    after: Callable[[int], Any] | None = None,
) -> int:
    """Read each channel's history after where ingestion left off.

    ``cursors`` maps a channel or thread id (as a string) to the last stored
    message id there; a channel without one is read from the beginning.
    ``after`` turns an id into what ``history(after=...)`` takes (a
    ``discord.Object``). A channel the bot cannot read is skipped. Returns how
    many messages were queued.
    """

    queued = 0
    for channel in channels:
        parent = getattr(channel, "parent", None)
        if channel.id in excluded or (parent is not None and parent.id in excluded):
            continue
        cursor = cursors.get(str(channel.id))
        start = after(cursor) if (cursor is not None and after is not None) else cursor
        try:
            async for message in channel.history(limit=None, after=start, oldest_first=True):
                payload = message_payload(message)
                if payload is None or is_excluded(payload, excluded):
                    continue
                batcher.add(payload)
                queued += 1
                if batcher.pending >= batcher.max_batch:
                    await batcher.flush()
        except Exception:  # noqa: BLE001 - one unreadable channel must not stop the rest
            logger.warning("could not read the history of channel %s", channel.id, exc_info=True)
    return queued

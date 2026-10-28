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


def stream_id(payload: dict[str, Any]) -> int:
    """The channel or thread whose history a message belongs to."""

    thread_id = payload.get("thread_id")
    return int(thread_id) if thread_id is not None else int(payload["channel_id"])


def edit_action(message: Any, excluded: set[int]) -> tuple[str, Any] | None:
    """What an edited message means for the backend.

    An edit that leaves text is stored again. An edit that removes all the text
    (an attachment kept, say) must remove what was stored, or the old words stay
    searchable; deleting an id that was never stored is harmless.
    """

    payload = message_payload(message)
    if payload is not None:
        return None if is_excluded(payload, excluded) else ("upsert", payload)
    if message.guild is None:
        return None
    return ("delete", message.id)


def merge_cursors(stored: dict[str, int], rewinds: dict[int, int]) -> dict[str, int]:
    """The backend's cursors, moved back where queued messages were dropped.

    A stream with no stored cursor is read from its beginning anyway.
    """

    merged = dict(stored)
    for stream, rewind in rewinds.items():
        key = str(stream)
        if key in merged:
            merged[key] = min(merged[key], rewind)
    return merged


class Batcher:
    """Collects messages and deletions and sends them to the backend in batches.

    Sending one request per message would rebuild a conversation for every line
    typed; batching every few seconds rebuilds it once. Each batch is taken off
    the queue before it is sent and put back if sending fails, so messages that
    arrive meanwhile are never mistaken for the batch.

    The message queue is bounded. When the backend is down long enough to fill it,
    new messages are refused and the batcher remembers, per channel or thread, to
    read history again from just before the first refused one (:meth:`take_rewinds`).
    The backend's cursor alone cannot bring them back: a later message that did get
    stored moves the cursor past them. Deletions are kept unbounded — they are only
    ids, and history cannot replay a deletion.
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
        self._max_pending = max_pending
        self._messages: deque[dict[str, Any]] = deque()
        self._deletes: deque[int] = deque()
        self._rewinds: dict[int, int] = {}
        self._lock = asyncio.Lock()

    @property
    def pending(self) -> int:
        return len(self._messages) + len(self._deletes)

    @property
    def max_batch(self) -> int:
        return self._max_batch

    @property
    def needs_rewind(self) -> bool:
        return bool(self._rewinds)

    def add(self, payload: dict[str, Any]) -> None:
        # The latest change to a message wins: a pending deletion of it is
        # cancelled (the text came back), and a newer version replaces an older one.
        message_id = payload["message_id"]
        _remove(self._deletes, lambda queued: queued == message_id)
        _remove(self._messages, lambda queued: queued["message_id"] == message_id)
        if len(self._messages) >= self._max_pending:
            stream = stream_id(payload)
            before = int(payload["message_id"]) - 1
            self._rewinds[stream] = min(self._rewinds.get(stream, before), before)
            return
        self._messages.append(payload)

    def delete(self, message_ids: Iterable[int]) -> None:
        ids = list(dict.fromkeys(message_ids))  # in order, without repeats
        wanted = set(ids)
        _remove(self._messages, lambda queued: queued["message_id"] in wanted)
        _remove(self._deletes, lambda queued: queued in wanted)
        self._deletes.extend(ids)

    def take_rewinds(self) -> dict[int, int]:
        """Where history must be read again, cleared once taken."""

        rewinds, self._rewinds = self._rewinds, {}
        return rewinds

    async def flush(self) -> bool:
        """Send everything queued. False when the backend refused or was away.

        Deletions go after the messages, so a message posted and deleted within
        one interval is stored and then removed, not the other way round.
        """

        async with self._lock:
            if not await _drain(self._messages, self._send_messages, self._max_batch):
                return False
            return await _drain(self._deletes, self._send_deletes, self._max_batch)

    async def run(
        self,
        interval: float,
        on_rewind: Callable[[], None] | None = None,
        ready: asyncio.Event | None = None,
    ) -> None:
        """Flush every ``interval`` seconds, forever. After a successful flush with
        refused messages to recover, ``on_rewind`` is called to read history again.

        Nothing is sent while ``ready`` is clear. The bot clears it until it has
        read where history left off: storing a new message first would move that
        point past the messages posted while the bot was away.
        """

        while True:
            await asyncio.sleep(interval)
            if ready is not None and not ready.is_set():
                continue
            if not await self.flush():
                logger.warning("could not forward %d message change(s); will retry", self.pending)
            elif self._rewinds and on_rewind is not None:
                on_rewind()


def _remove(queue: deque, matches: Callable[[Any], bool]) -> None:
    kept = [item for item in queue if not matches(item)]
    if len(kept) != len(queue):
        queue.clear()
        queue.extend(kept)


async def fetch_until_ready(
    fetch: Callable[[], Awaitable[dict[str, int] | None]], *, retry_after: float
) -> dict[str, int]:
    """Ask the backend where history left off until it answers.

    Giving up would let live messages move that point forward, and the messages
    posted while the bot was away would never be read.
    """

    while True:
        cursors = await fetch()
        if cursors is not None:
            return cursors
        logger.warning("backend did not say where history left off; retrying")
        await asyncio.sleep(retry_after)


async def _drain(queue: deque, send: Callable[[list], Awaitable[bool]], size: int) -> bool:
    while queue:
        batch = [queue.popleft() for _ in range(min(size, len(queue)))]
        if not await send(batch):
            queue.extendleft(reversed(batch))
            return False
    return True


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

"""The bot's side of channel ingestion (#76, R2), kept free of the Discord client so
it can be tested: what a message becomes, how messages are batched to the backend,
and how history is read after a restart.

Only conversation is forwarded — messages people wrote in the configured server.
Bots, DMs, system messages (pins, joins, …) and messages without text are skipped.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

logger = logging.getLogger(__name__)

# Message types that are someone talking; everything else is a system notice.
_CONVERSATION_TYPES = ("default", "reply")

# How many message ids seen live (posted, edited, deleted) the batcher remembers, so
# that history read afterwards cannot bring back an older version of them.
_LIVE_MEMORY = 20000

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
    """Collects message changes and sends them to the backend in batches.

    Sending one request per message would rebuild a conversation for every line
    typed; batching every few seconds rebuilds it once.

    Every live change to a message (posted, edited, deleted) gets a new generation
    number, and each queued operation carries the generation it was made at. An
    operation older than its message's latest generation is never sent again:
    a failed batch is put back without the operations overtaken while it was in
    flight, and the result of fetching an edit (:meth:`begin_edit`) is dropped if
    a newer change came first. A message read from history has no generation; it
    never overrides a message seen live.

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
        send_stream_deletes: SendDeletes | None = None,
        max_batch: int = 50,
        max_pending: int = 5000,
    ) -> None:
        self._send_messages = send_messages
        self._send_deletes = send_deletes
        self._send_stream_deletes = send_stream_deletes
        self._max_batch = max_batch
        self._max_pending = max_pending
        self._messages: deque[tuple[int, dict[str, Any]]] = deque()
        self._deletes: deque[tuple[int, int]] = deque()
        self._streams: deque[tuple[int, int]] = deque()
        self._rewinds: dict[int, int] = {}
        # message id -> its latest live generation (0: never seen live).
        self._generations: OrderedDict[int, int] = OrderedDict()
        self._counter = 0
        self._lock = asyncio.Lock()

    @property
    def pending(self) -> int:
        return len(self._messages) + len(self._deletes) + len(self._streams)

    @property
    def max_batch(self) -> int:
        return self._max_batch

    @property
    def needs_rewind(self) -> bool:
        return bool(self._rewinds)

    def begin_edit(self, message_id: int) -> int:
        """Start fetching an edited message; pass the result to :meth:`add` or
        :meth:`delete` with this token, which drops it if a newer change came."""

        return self._bump(message_id)

    def add(
        self, payload: dict[str, Any], *, history: bool = False, token: int | None = None
    ) -> None:
        """Queue a message: a live change, a fetched edit (``token``), or one read
        from history (``history``)."""

        message_id = int(payload["message_id"])
        if history and message_id in self._generations:
            return  # a snapshot older than what was seen live
        if token is not None and self._generation(message_id) != token:
            return  # a newer change overtook this fetch
        if len(self._messages) >= self._max_pending:
            # Read history again from just before it: what history then holds is
            # the latest version, including an edit refused here.
            self.reread(stream_id(payload), message_id)
            return
        if history:
            generation = 0
        elif token is not None:
            generation = token
        else:
            generation = self._bump(message_id)
        self._discard(message_id)
        self._messages.append((generation, payload))

    def delete(self, message_ids: Iterable[int], *, token: int | None = None) -> None:
        for message_id in dict.fromkeys(message_ids):  # in order, without repeats
            if token is not None and self._generation(message_id) != token:
                continue
            generation = token if token is not None else self._bump(message_id)
            self._discard(message_id)
            self._deletes.append((generation, message_id))

    def delete_streams(self, stream_ids: Iterable[int]) -> None:
        """A channel or thread was deleted: forget everything stored from it."""

        ids = set(stream_ids)
        _keep(
            self._messages,
            lambda item: stream_id(item[1]) not in ids and item[1]["channel_id"] not in ids,
        )
        self._streams.extend((0, stream) for stream in ids)
        for stream in ids:
            self._rewinds.pop(stream, None)

    def reread(self, stream: int, message_id: int) -> None:
        """A live change to ``message_id`` could not be taken: get it from history.

        The message is forgotten as seen live, or history's version — newer than
        what was stored — would be ignored as a stale snapshot.
        """

        self._generations.pop(message_id, None)
        self.rewind(stream, message_id - 1)

    def rewind(self, stream: int, after: int) -> None:
        """Read ``stream``'s history again from after message ``after`` (0: from the start)."""

        self._rewinds[stream] = min(self._rewinds.get(stream, after), after)

    def take_rewinds(self) -> dict[int, int]:
        """Where history must be read again, cleared once taken."""

        rewinds, self._rewinds = self._rewinds, {}
        return rewinds

    async def flush(self) -> bool:
        """Send everything queued. False when the backend refused or was away.

        Messages go first, then deletions, then deleted channels, so a message
        posted and deleted within one interval is stored and then removed.
        """

        async with self._lock:
            if not await self._drain(
                self._messages, self._send_messages, lambda p: int(p["message_id"])
            ):
                return False
            if not await self._drain(self._deletes, self._send_deletes, int):
                return False
            if self._send_stream_deletes is None:
                self._streams.clear()
                return True
            return await self._drain(self._streams, self._send_stream_deletes, None)

    async def _drain(
        self,
        queue: deque,
        send: Callable[[list], Awaitable[bool]],
        message_id_of: Callable[[Any], int] | None,
    ) -> bool:
        while queue:
            batch = [queue.popleft() for _ in range(min(self._max_batch, len(queue)))]
            if await send([item for _, item in batch]):
                continue
            # Put back only what was not overtaken while in flight.
            current = [
                (generation, item)
                for generation, item in batch
                if message_id_of is None or self._generation(message_id_of(item)) == generation
            ]
            queue.extendleft(reversed(current))
            return False
        return True

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

    def _generation(self, message_id: int) -> int:
        return self._generations.get(message_id, 0)

    def _bump(self, message_id: int) -> int:
        self._counter += 1
        self._generations[message_id] = self._counter
        self._generations.move_to_end(message_id)
        while len(self._generations) > _LIVE_MEMORY:
            self._generations.popitem(last=False)
        return self._counter

    def _discard(self, message_id: int) -> None:
        """Drop queued operations on ``message_id``; the newest one replaces them."""

        _keep(self._messages, lambda item: int(item[1]["message_id"]) != message_id)
        _keep(self._deletes, lambda item: item[1] != message_id)


def _keep(queue: deque, wanted: Callable[[Any], bool]) -> None:
    """Filter ``queue`` in place: a flush in flight holds this same object."""

    kept = [item for item in queue if wanted(item)]
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


async def backfill(
    channels: Sequence[Any],
    *,
    cursors: dict[str, int],
    batcher: Batcher,
    excluded: set[int],
    after: Callable[[int], Any] | None = None,
    is_permanent: Callable[[Exception], bool] = lambda exc: False,
    flush: Callable[[], Awaitable[Any]] | None = None,
) -> int:
    """Read each channel's history after where ingestion left off.

    ``cursors`` maps a channel or thread id (as a string) to the last stored
    message id there; a channel without one is read from the beginning.
    ``after`` turns an id into what ``history(after=...)`` takes (a
    ``discord.Object``). A channel the bot may not read (``is_permanent``) is
    skipped. Any other failure leaves a rewind for that channel, so it is read
    again from the same point once sending works: giving up would let live
    messages move its cursor past the unread part. ``flush`` sends what has been
    queued so far (default: the batcher's flush); the bot passes one that holds
    back while sending is paused. Returns how many messages were queued.
    """

    send = flush if flush is not None else batcher.flush

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
                batcher.add(payload, history=True)
                queued += 1
                if batcher.pending >= batcher.max_batch:
                    await send()
        except Exception as exc:  # noqa: BLE001 - one channel must not stop the rest
            if is_permanent(exc):
                logger.info("cannot read the history of channel %s; skipped", channel.id)
            else:
                logger.warning("reading channel %s failed; will retry", channel.id, exc_info=True)
                batcher.rewind(channel.id, cursor if cursor is not None else 0)
    return queued

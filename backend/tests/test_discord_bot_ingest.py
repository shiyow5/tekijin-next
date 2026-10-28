"""Unit tests for the bot's side of channel ingestion (#76): turning a Discord
message into a payload, batching what goes to the backend, and reading history."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from types import SimpleNamespace

import httpx

from tekijin.discord_bot.backend import (
    SECRET_HEADER,
    get_cursors,
    post_deletes,
    post_messages,
    post_stream_deletes,
)
from tekijin.discord_bot.ingest import (
    Batcher,
    backfill,
    edit_action,
    fetch_until_ready,
    is_excluded,
    merge_cursors,
    message_payload,
    parse_ids,
)

AT = dt.datetime(2026, 10, 28, 3, 0, tzinfo=dt.UTC)


def _channel(channel_id: int = 10, name: str = "雑談", parent=None):
    return SimpleNamespace(id=channel_id, name=name, parent=parent)


def _message(message_id: int = 1, content: str = "部室の鍵は？", **kw):
    author = SimpleNamespace(
        id=kw.get("author_id", 501), display_name="アリス", bot=kw.get("bot", False)
    )
    return SimpleNamespace(
        id=message_id,
        guild=kw.get("guild", SimpleNamespace(id=42)),
        channel=kw.get("channel", _channel()),
        author=author,
        type=SimpleNamespace(name=kw.get("type", "default")),
        content=content,
        created_at=AT,
        edited_at=kw.get("edited_at"),
    )


def test_a_plain_message_becomes_a_payload() -> None:
    assert message_payload(_message()) == {
        "message_id": 1,
        "channel_id": 10,
        "channel_name": "雑談",
        "thread_id": None,
        "thread_name": None,
        "author_id": "501",
        "author_name": "アリス",
        "content": "部室の鍵は？",
        "created_at": "2026-10-28T03:00:00+00:00",
        "edited_at": None,
    }


def test_a_thread_message_keeps_its_parent_channel() -> None:
    thread = _channel(77, "合宿の宿", parent=_channel(10, "雑談"))
    payload = message_payload(_message(channel=thread, edited_at=AT))
    assert payload is not None
    assert (payload["channel_id"], payload["channel_name"]) == (10, "雑談")
    assert (payload["thread_id"], payload["thread_name"]) == (77, "合宿の宿")
    assert payload["edited_at"] == "2026-10-28T03:00:00+00:00"


def test_messages_that_are_not_conversation_are_skipped() -> None:
    assert message_payload(_message(bot=True)) is None
    assert message_payload(_message(guild=None)) is None
    assert message_payload(_message(type="pins_add")) is None
    assert message_payload(_message(content="  ")) is None
    assert message_payload(_message(type="reply")) is not None


def test_exclusion_covers_threads_of_an_excluded_channel() -> None:
    excluded = parse_ids(" 10, 12 ,,x")
    assert excluded == {10, 12}
    assert is_excluded({"channel_id": 10, "thread_id": None}, excluded)
    assert is_excluded({"channel_id": 10, "thread_id": 77}, excluded)
    assert is_excluded({"channel_id": 11, "thread_id": 12}, excluded)
    assert not is_excluded({"channel_id": 11, "thread_id": None}, excluded)


class _Recorder:
    def __init__(self, fail: int = 0) -> None:
        self.calls: list[list] = []
        self.fail = fail

    async def __call__(self, items: list) -> bool:
        if self.fail:
            self.fail -= 1
            return False
        self.calls.append(list(items))
        return True


def test_batcher_sends_in_batches_and_deletes_after_messages() -> None:
    sent, deleted = _Recorder(), _Recorder()
    batcher = Batcher(sent, deleted, max_batch=2)
    for n in range(5):
        batcher.add({"message_id": n, "channel_id": 10, "thread_id": None})
    batcher.delete([9, 8])

    assert asyncio.run(batcher.flush()) is True
    assert [[m["message_id"] for m in call] for call in sent.calls] == [[0, 1], [2, 3], [4]]
    assert deleted.calls == [[9, 8]]
    assert batcher.pending == 0


MSG1 = {"message_id": 1, "channel_id": 10, "thread_id": None}


def test_batcher_keeps_what_failed_for_the_next_flush() -> None:
    sent, deleted = _Recorder(fail=1), _Recorder()
    batcher = Batcher(sent, deleted, max_batch=10)
    batcher.add(MSG1)
    batcher.delete([2])

    assert asyncio.run(batcher.flush()) is False
    assert batcher.pending == 2
    assert deleted.calls == []  # deletions wait until the messages before them are in
    assert asyncio.run(batcher.flush()) is True
    assert sent.calls == [[MSG1]] and deleted.calls == [[2]]


def test_a_full_queue_refuses_new_messages_and_remembers_where_to_reread() -> None:
    """Dropping a message must stay recoverable from history (Codex #77)."""

    batcher = Batcher(_Recorder(), _Recorder(), max_batch=10, max_pending=2)
    for n in (5, 6):
        batcher.add({"message_id": n, "channel_id": 10, "thread_id": None})
    batcher.add({"message_id": 9, "channel_id": 10, "thread_id": None})
    batcher.add({"message_id": 8, "channel_id": 10, "thread_id": None})
    batcher.add({"message_id": 30, "channel_id": 10, "thread_id": 77})
    assert batcher.pending == 2 and batcher.needs_rewind
    assert batcher.take_rewinds() == {10: 7, 77: 29}
    assert not batcher.needs_rewind
    # Deletions are never refused.
    batcher.delete(range(100, 200))
    assert batcher.pending == 102


def test_messages_added_while_sending_are_not_lost() -> None:
    """A batch is taken off the queue before it is sent (Codex #77)."""

    holder: dict = {}

    async def send(items: list) -> bool:
        holder.setdefault("calls", []).append([m["message_id"] for m in items])
        if len(holder["calls"]) == 1:
            holder["batcher"].add({"message_id": 4, "channel_id": 10, "thread_id": None})
        return True

    batcher = Batcher(send, _Recorder(), max_batch=2, max_pending=3)
    holder["batcher"] = batcher
    for n in (1, 2, 3):
        batcher.add({"message_id": n, "channel_id": 10, "thread_id": None})

    assert asyncio.run(batcher.flush()) is True
    assert holder["calls"] == [[1, 2], [3, 4]]


def test_run_flushes_and_asks_to_reread_after_refusals() -> None:
    sent = _Recorder()
    batcher = Batcher(sent, _Recorder(), max_pending=1)
    batcher.add({"message_id": 1, "channel_id": 10, "thread_id": None})
    batcher.add({"message_id": 2, "channel_id": 10, "thread_id": None})
    rereads: list = []

    async def run_once() -> None:
        task = asyncio.create_task(batcher.run(0.01, lambda: rereads.append(True)))
        await asyncio.sleep(0.05)
        task.cancel()

    asyncio.run(run_once())
    assert sent.calls[0] == [{"message_id": 1, "channel_id": 10, "thread_id": None}]
    assert rereads


def test_merge_cursors_moves_back_only_stored_streams() -> None:
    assert merge_cursors({"10": 9, "11": 3}, {10: 4, 11: 7, 12: 1}) == {"10": 4, "11": 3}


def test_edit_action() -> None:
    assert edit_action(_message(content="直した"), set())[0] == "upsert"
    # Text removed by the edit: what was stored must go.
    assert edit_action(_message(7, content=""), set()) == ("delete", 7)
    assert edit_action(_message(content="x"), {10}) is None
    assert edit_action(_message(content="", guild=None), set()) is None


class _History:
    def __init__(self, channel_id: int, messages: list) -> None:
        self.id = channel_id
        self.messages = messages
        self.after: list = []

    def history(self, *, limit, after, oldest_first):
        assert limit is None and oldest_first is True
        self.after.append(after)

        async def gen():
            for message in self.messages:
                yield message

        return gen()


def test_backfill_resumes_after_each_cursor_and_skips_excluded_channels() -> None:
    general = _History(10, [_message(5, channel=_channel(10)), _message(6, channel=_channel(10))])
    fresh = _History(11, [_message(7, channel=_channel(11, "運営"))])
    secret = _History(12, [_message(8, channel=_channel(12, "秘密"))])
    sent = _Recorder()
    batcher = Batcher(sent, _Recorder(), max_batch=1)

    count = asyncio.run(
        backfill(
            [general, fresh, secret],
            cursors={"10": 4},
            batcher=batcher,
            excluded={12},
            after=lambda message_id: f"after:{message_id}",
        )
    )

    assert count == 3
    assert general.after == ["after:4"] and fresh.after == [None]
    assert secret.after == []
    asyncio.run(batcher.flush())
    assert [call[0]["message_id"] for call in sent.calls] == [5, 6, 7]


def test_backfill_survives_a_channel_it_cannot_read() -> None:
    class Forbidden(_History):
        def history(self, **kwargs):
            async def gen():
                raise PermissionError("no access")
                yield  # pragma: no cover

            return gen()

    batcher = Batcher(_Recorder(), _Recorder())
    readable = _History(11, [_message(7, channel=_channel(11))])
    count = asyncio.run(
        backfill(
            [Forbidden(10, []), readable],
            cursors={},
            batcher=batcher,
            excluded=set(),
            is_permanent=lambda exc: isinstance(exc, PermissionError),
        )
    )
    assert count == 1
    # Not allowed to read is final: no rewind, so no endless retries.
    assert not batcher.needs_rewind


def test_backfill_rereads_a_channel_after_a_passing_failure() -> None:
    """A transient error must keep the channel's starting point (Codex #77)."""

    class Flaky(_History):
        def history(self, **kwargs):
            async def gen():
                raise RuntimeError("503")
                yield  # pragma: no cover

            return gen()

    batcher = Batcher(_Recorder(), _Recorder())
    asyncio.run(
        backfill(
            [Flaky(10, []), Flaky(11, [])], cursors={"10": 40}, batcher=batcher, excluded=set()
        )
    )
    assert batcher.take_rewinds() == {10: 40, 11: 0}
    # Rewound to 0 = read from the start, even if a live message stored a cursor since.
    assert merge_cursors({"11": 90}, {11: 0}) == {"11": 0}


def _call(handler, func, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await func(http, base_url="http://api/", secret="k", guild_id=42, **kwargs)

    return asyncio.run(run())


def test_post_messages_and_deletes_report_success() -> None:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content), request.headers[SECRET_HEADER]))
        return httpx.Response(200, json={"status": "ok"})

    assert _call(handler, post_messages, messages=[{"message_id": 1}]) is True
    assert _call(handler, post_deletes, message_ids=[1]) is True
    assert seen[0] == (
        "http://api/internal/discord/messages",
        {"guild_id": "42", "messages": [{"message_id": 1}]},
        "k",
    )
    assert seen[1][0] == "http://api/internal/discord/messages/delete"
    assert seen[1][1] == {"guild_id": "42", "message_ids": [1]}


def test_post_messages_fails_when_not_stored() -> None:
    def unlinked(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "unlinked"})

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert _call(unlinked, post_messages, messages=[]) is False
    assert _call(down, post_deletes, message_ids=[1]) is False


def test_get_cursors() -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        assert request.url.params["guild_id"] == "42"
        return httpx.Response(200, json={"status": "ok", "cursors": {"10": 5}})

    def unlinked(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "unlinked", "cursors": {}})

    def error(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert _call(ok, get_cursors) == {"10": 5}
    assert _call(unlinked, get_cursors) is None
    assert _call(error, get_cursors) is None
    assert _call(down, get_cursors) is None


def test_the_latest_change_to_a_message_wins() -> None:
    """Text removed and then restored before a flush must end up stored (Codex #77)."""

    sent, deleted = _Recorder(), _Recorder()
    batcher = Batcher(sent, deleted)
    first = {"message_id": 1, "channel_id": 10, "thread_id": None, "content": "a"}
    batcher.add(first)
    batcher.delete([1])
    batcher.add({**first, "content": "b"})
    batcher.add({**first, "content": "c"})
    batcher.add({"message_id": 2, "channel_id": 10, "thread_id": None})
    batcher.delete([2])

    asyncio.run(batcher.flush())
    assert sent.calls == [[{**first, "content": "c"}]]
    assert deleted.calls == [[2]]


def test_nothing_is_sent_until_ready() -> None:
    sent = _Recorder()
    batcher = Batcher(sent, _Recorder())
    batcher.add({"message_id": 1, "channel_id": 10, "thread_id": None})

    async def scenario() -> tuple[int, int]:
        ready = asyncio.Event()
        task = asyncio.create_task(batcher.run(0.01, ready=ready))
        await asyncio.sleep(0.05)
        before = len(sent.calls)
        ready.set()
        await asyncio.sleep(0.05)
        task.cancel()
        return before, len(sent.calls)

    assert asyncio.run(scenario()) == (0, 1)


def test_fetch_until_ready_retries() -> None:
    answers = [None, None, {"10": 5}]

    async def fetch():
        return answers.pop(0)

    assert asyncio.run(fetch_until_ready(fetch, retry_after=0)) == {"10": 5}
    assert answers == []


def test_history_never_overrides_what_was_seen_live() -> None:
    """A snapshot read from history may predate a live edit or deletion (Codex #77)."""

    sent, deleted = _Recorder(), _Recorder()
    batcher = Batcher(sent, deleted)
    batcher.delete([2])
    asyncio.run(batcher.flush())  # the deletion already reached the backend
    batcher.add({"message_id": 3, "channel_id": 10, "thread_id": None, "content": "新"})

    batcher.add({"message_id": 2, "channel_id": 10, "thread_id": None}, history=True)
    batcher.add(
        {"message_id": 3, "channel_id": 10, "thread_id": None, "content": "旧"}, history=True
    )
    batcher.add({"message_id": 4, "channel_id": 10, "thread_id": None}, history=True)
    asyncio.run(batcher.flush())

    assert [m["message_id"] for m in sent.calls[0]] == [3, 4]
    assert sent.calls[0][0]["content"] == "新"


def test_a_refused_live_message_can_come_back_from_history() -> None:
    batcher = Batcher(_Recorder(), _Recorder(), max_pending=1)
    batcher.add({"message_id": 1, "channel_id": 10, "thread_id": None})
    batcher.add({"message_id": 2, "channel_id": 10, "thread_id": None})  # refused
    asyncio.run(batcher.flush())
    batcher.add({"message_id": 2, "channel_id": 10, "thread_id": None}, history=True)
    assert batcher.pending == 1


def test_a_refused_edit_comes_back_from_history() -> None:
    """The edited message was seen live before; history must still update it (Codex #77)."""

    sent = _Recorder()
    batcher = Batcher(sent, _Recorder(), max_pending=1)
    old = {"message_id": 1, "channel_id": 10, "thread_id": None, "content": "旧"}
    batcher.add(old)
    asyncio.run(batcher.flush())
    batcher.add({"message_id": 2, "channel_id": 10, "thread_id": None})
    batcher.add({**old, "content": "新"})  # refused: the queue is full
    assert batcher.take_rewinds() == {10: 0}
    asyncio.run(batcher.flush())

    batcher.add({**old, "content": "新"}, history=True)
    asyncio.run(batcher.flush())
    assert sent.calls[-1] == [{**old, "content": "新"}]


def test_reread_forgets_the_message_and_rewinds_its_stream() -> None:
    batcher = Batcher(_Recorder(), _Recorder())
    batcher.add({"message_id": 50, "channel_id": 10, "thread_id": 77})
    batcher.reread(77, 50)
    assert batcher.take_rewinds() == {77: 49}
    batcher.add({"message_id": 50, "channel_id": 10, "thread_id": 77}, history=True)
    assert batcher.pending == 1


def test_backfill_flushes_through_the_given_function() -> None:
    """The bot holds back sending while paused after a reconnect (Codex #77)."""

    calls: list = []

    async def paused_flush() -> None:
        calls.append("held")

    sent = _Recorder()
    batcher = Batcher(sent, _Recorder(), max_batch=1)
    channel = _History(10, [_message(5, channel=_channel(10)), _message(6, channel=_channel(10))])
    asyncio.run(
        backfill([channel], cursors={}, batcher=batcher, excluded=set(), flush=paused_flush)
    )
    assert calls == ["held", "held"]
    assert sent.calls == [] and batcher.pending == 2


def test_a_failed_operation_overtaken_in_flight_is_not_put_back() -> None:
    """Text removed, the deletion fails in flight, the text comes back (Codex #77)."""

    holder: dict = {}

    async def failing_delete(ids: list) -> bool:
        holder["batcher"].add({"message_id": 1, "channel_id": 10, "thread_id": None, "v": 2})
        return False

    sent = _Recorder()
    batcher = Batcher(sent, failing_delete)
    holder["batcher"] = batcher
    batcher.delete([1])
    assert asyncio.run(batcher.flush()) is False
    # The overtaken deletion is gone; only the restored text remains.
    assert batcher.pending == 1
    asyncio.run(batcher.flush())
    assert sent.calls[-1] == [{"message_id": 1, "channel_id": 10, "thread_id": None, "v": 2}]


def test_a_fetched_edit_overtaken_by_a_deletion_is_dropped() -> None:
    """The edit fetch returns after the message was deleted (Codex #77)."""

    batcher = Batcher(_Recorder(), _Recorder())
    token = batcher.begin_edit(1)
    batcher.delete([1])  # arrives while the fetch is in flight
    batcher.add({"message_id": 1, "channel_id": 10, "thread_id": None}, token=token)
    batcher.delete([1], token=token)
    assert batcher.pending == 1  # just the newer deletion

    fresh = batcher.begin_edit(2)
    batcher.add({"message_id": 2, "channel_id": 10, "thread_id": None}, token=fresh)
    assert batcher.pending == 2


def test_deleted_streams_drop_their_queue_and_are_sent_last() -> None:
    sent, deleted, streams = _Recorder(), _Recorder(), _Recorder()
    batcher = Batcher(sent, deleted, send_stream_deletes=streams)
    batcher.add({"message_id": 1, "channel_id": 10, "thread_id": 77})
    batcher.add({"message_id": 2, "channel_id": 11, "thread_id": None})
    batcher.add({"message_id": 3, "channel_id": 77, "thread_id": None})
    batcher.delete_streams([77])
    asyncio.run(batcher.flush())
    assert [m["message_id"] for m in sent.calls[0]] == [2]
    assert streams.calls == [[77]]


def test_post_stream_deletes() -> None:
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"status": "ok"})

    assert _call(handler, post_stream_deletes, stream_ids=[77]) is True
    assert seen == [
        ("http://api/internal/discord/streams/delete", {"guild_id": "42", "stream_ids": [77]})
    ]


def test_nothing_comes_back_into_a_deleted_stream() -> None:
    """A late edit fetch must not recreate a deleted thread (Codex #77)."""

    sent = _Recorder()
    batcher = Batcher(sent, _Recorder(), send_stream_deletes=_Recorder())
    token = batcher.begin_edit(5)
    batcher.delete_streams([77, 11])
    asyncio.run(batcher.flush())
    batcher.add({"message_id": 5, "channel_id": 10, "thread_id": 77}, token=token)
    batcher.add({"message_id": 6, "channel_id": 11, "thread_id": 90}, history=True)
    batcher.add({"message_id": 7, "channel_id": 12, "thread_id": None})
    asyncio.run(batcher.flush())
    assert [m["message_id"] for call in sent.calls for m in call] == [7]


def test_a_pause_stops_a_flush_between_batches() -> None:
    """A reconnect clears the gate while a flush is under way (Codex #77)."""

    gate = asyncio.Event()
    gate.set()

    async def send(items: list) -> bool:
        gate.clear()  # the Gateway reconnects while this batch is in flight
        return True

    batcher = Batcher(send, _Recorder(), max_batch=1, gate=gate)
    for n in (1, 2):
        batcher.add({"message_id": n, "channel_id": 10, "thread_id": None})
    assert asyncio.run(batcher.flush()) is False
    assert batcher.pending == 1  # the second batch waits for the gate


def test_backfill_removes_text_an_edit_took_away() -> None:
    """Recovery reads the message after its text was removed (Codex #77)."""

    deleted = _Recorder()
    batcher = Batcher(_Recorder(), deleted)
    batcher.add({"message_id": 9, "channel_id": 10, "thread_id": None})
    batcher.reread(10, 9)  # the edit could not be fetched
    asyncio.run(batcher.flush())
    channel = _History(10, [_message(9, content="", channel=_channel(10)), _message(8, bot=True)])
    asyncio.run(backfill([channel], cursors={}, batcher=batcher, excluded=set()))
    asyncio.run(batcher.flush())
    assert deleted.calls == [[9]]
    # A message seen live is not overridden by such a snapshot.
    batcher.add({"message_id": 20, "channel_id": 10, "thread_id": None})
    batcher.delete_from_history(20)
    assert batcher.pending == 1

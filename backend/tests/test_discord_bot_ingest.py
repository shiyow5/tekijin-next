"""Unit tests for the bot's side of channel ingestion (#76): turning a Discord
message into a payload, batching what goes to the backend, and reading history."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from types import SimpleNamespace

import httpx

from tekijin.discord_bot.backend import SECRET_HEADER, get_cursors, post_deletes, post_messages
from tekijin.discord_bot.ingest import Batcher, backfill, is_excluded, message_payload, parse_ids

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
        batcher.add({"message_id": n})
    batcher.delete([9, 8])

    assert asyncio.run(batcher.flush()) is True
    assert [[m["message_id"] for m in call] for call in sent.calls] == [[0, 1], [2, 3], [4]]
    assert deleted.calls == [[9, 8]]
    assert batcher.pending == 0


def test_batcher_keeps_what_failed_for_the_next_flush() -> None:
    sent, deleted = _Recorder(fail=1), _Recorder()
    batcher = Batcher(sent, deleted, max_batch=10)
    batcher.add({"message_id": 1})
    batcher.delete([2])

    assert asyncio.run(batcher.flush()) is False
    assert batcher.pending == 2
    assert deleted.calls == []  # deletions wait until the messages before them are in
    assert asyncio.run(batcher.flush()) is True
    assert sent.calls == [[{"message_id": 1}]] and deleted.calls == [[2]]


def test_batcher_drops_the_oldest_beyond_its_limit() -> None:
    batcher = Batcher(_Recorder(), _Recorder(), max_batch=10, max_pending=3)
    for n in range(5):
        batcher.add({"message_id": n})
    assert batcher.pending == 3


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
        backfill([Forbidden(10, []), readable], cursors={}, batcher=batcher, excluded=set())
    )
    assert count == 1


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

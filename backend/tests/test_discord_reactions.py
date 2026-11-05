"""The bot's check-mark relay: sent after the messages, kept until taken (#113)."""

from __future__ import annotations

import asyncio

from tekijin.discord_bot import reactions as module
from tekijin.discord_bot.reactions import MAX_TRIES, ReactionRelay


class _Backend:
    def __init__(self, statuses: list[str], flushes: list[bool] | None = None) -> None:
        self.statuses = statuses
        self.flushes = flushes or []
        self.sent: list[dict] = []
        self.flushed = 0

    async def send(self, reaction: dict) -> str:
        self.sent.append(reaction)
        return self.statuses.pop(0) if self.statuses else "ok"

    async def flush(self) -> bool:
        self.flushed += 1
        return self.flushes.pop(0) if self.flushes else True


def _relay(backend: _Backend, *, ready: bool = True) -> tuple[ReactionRelay, asyncio.Event]:
    event = asyncio.Event()
    if ready:
        event.set()
    return ReactionRelay(backend.send, backend.flush, ready=event), event


def test_a_reaction_waits_for_ready_and_a_successful_flush() -> None:
    async def scenario() -> None:
        backend = _Backend([], flushes=[False])
        relay, ready = _relay(backend, ready=False)
        relay.add({"message_id": 1})
        await relay.deliver()  # reading history: nothing is sent
        assert backend.sent == [] and backend.flushed == 0
        ready.set()
        await relay.deliver()  # the messages could not be stored: wait
        assert backend.sent == [] and relay.pending == 1
        await relay.deliver()
        assert backend.sent == [{"message_id": 1}] and relay.pending == 0

    asyncio.run(scenario())


def test_pending_and_failed_reactions_are_tried_again_then_dropped() -> None:
    async def scenario() -> None:
        backend = _Backend(["pending", "error", "ignored"])
        relay, _ = _relay(backend)
        relay.add({"message_id": 1})
        await relay.deliver()
        await relay.deliver()
        assert relay.pending == 1
        await relay.deliver()
        assert relay.pending == 0 and len(backend.sent) == 3

        stubborn = _Backend(["pending"] * MAX_TRIES)
        relay, _ = _relay(stubborn)
        relay.add({"message_id": 2})
        for _ in range(MAX_TRIES):
            await relay.deliver()
        assert relay.pending == 0 and len(stubborn.sent) == MAX_TRIES

    asyncio.run(scenario())


def test_the_queue_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(module, "MAX_PENDING", 2)
    relay, _ = _relay(_Backend([]))
    for i in range(3):
        relay.add({"message_id": i})
    assert relay.pending == 2


def test_run_retries_on_its_interval() -> None:
    async def scenario() -> None:
        backend = _Backend(["pending"])
        relay, _ = _relay(backend)
        relay.add({"message_id": 1})
        task = asyncio.create_task(relay.run(0.01))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if relay.pending == 0:
                break
        task.cancel()
        assert relay.pending == 0 and len(backend.sent) == 2

    asyncio.run(scenario())

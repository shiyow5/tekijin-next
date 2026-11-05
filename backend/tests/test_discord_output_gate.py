"""The bot's one place to post (#114, ``discord_bot/output.py``).

The bot runs in a server shared with other people, so it may post only in the
configured output channel and the threads under it, and it pings no one unless
``discord_mention_responders`` is on. Unset, it posts nothing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import discord
import pytest

from tekijin.discord_bot.output import OutputGate, OutputRefused

GUILD = 10
OUTPUT = 20


class _Channel:
    def __init__(self, cid: int, *, parent_id: int | None = None, guild: int = GUILD) -> None:
        self.id = cid
        self.parent_id = parent_id
        self.guild = SimpleNamespace(id=guild)
        self.sent: list[tuple[str, dict]] = []
        self.threads: list[_Channel] = []

    async def send(self, content: str, **kwargs):
        self.sent.append((content, kwargs))
        return SimpleNamespace(id=999, channel=self)

    async def create_thread(self, **kwargs):
        thread = _Channel(self.id * 100 + len(self.threads), parent_id=self.id, guild=self.guild.id)
        thread.kwargs = kwargs
        self.threads.append(thread)
        return thread


class _Client:
    def __init__(self, *channels: _Channel) -> None:
        self._channels = {c.id: c for c in channels}

    def get_channel(self, cid: int):
        return self._channels.get(cid)

    async def fetch_channel(self, cid: int):
        raise discord.NotFound(SimpleNamespace(status=404, reason="nf"), "missing")


def _gate(channel: int | None = OUTPUT, *, mention: bool = False) -> OutputGate:
    return OutputGate(channel, guild_id=GUILD, mention=mention)


def test_unset_gate_posts_nothing() -> None:
    gate = _gate(None)
    output = _Channel(OUTPUT)
    assert not gate.enabled
    assert not gate.owns(output)
    with pytest.raises(OutputRefused):
        asyncio.run(gate.open_thread(_Client(output), title="t", content="c"))
    with pytest.raises(OutputRefused):
        asyncio.run(gate.post(_Client(output), OUTPUT, "c"))
    assert output.sent == [] and output.threads == []


def test_only_the_output_channel_and_its_threads_are_owned() -> None:
    gate = _gate()
    assert gate.owns(_Channel(OUTPUT))
    assert gate.owns(_Channel(21, parent_id=OUTPUT))
    assert not gate.owns(_Channel(21))
    assert not gate.owns(_Channel(22, parent_id=21))
    assert not gate.owns(_Channel(OUTPUT, guild=11))  # same id, another server
    assert gate.is_request_thread(_Channel(21, parent_id=OUTPUT))
    assert not gate.is_request_thread(_Channel(OUTPUT))
    assert not gate.owns(None)


def test_posting_elsewhere_is_refused() -> None:
    other = _Channel(30)
    other_thread = _Channel(31, parent_id=30)
    client = _Client(other, other_thread)
    gate = _gate()
    with pytest.raises(OutputRefused):
        asyncio.run(gate.post(client, 30, "c"))
    with pytest.raises(OutputRefused):
        asyncio.run(gate.post(client, 31, "c"))
    with pytest.raises(OutputRefused):
        asyncio.run(gate.post(client, 32, "c"))  # unknown channel
    assert other.sent == [] and other_thread.sent == []


def test_a_request_thread_pings_no_one_by_default() -> None:
    output = _Channel(OUTPUT)
    gate = _gate()
    thread = asyncio.run(
        gate.open_thread(_Client(output), title="依頼", content="本文", mention_ids=["123"])
    )
    assert thread.parent_id == OUTPUT
    ((content, kwargs),) = thread.sent
    mentions = kwargs["allowed_mentions"]
    assert content == "本文"
    assert mentions.everyone is False and mentions.users is False and mentions.roles is False
    assert output.sent == []  # the thread is opened without a message in the channel


def test_every_post_pings_no_one_by_default() -> None:
    thread = _Channel(21, parent_id=OUTPUT)
    asyncio.run(_gate().post(_Client(thread), 21, "@everyone hi", mention_ids=["123"]))
    ((_, kwargs),) = thread.sent
    mentions = kwargs["allowed_mentions"]
    assert mentions.everyone is False and mentions.users is False and mentions.roles is False


def test_mentions_reach_only_the_named_member_when_on() -> None:
    gate = _gate(mention=True)
    mentions = gate.allowed_mentions(["123"])
    assert mentions.everyone is False and mentions.roles is False
    assert [u.id for u in mentions.users] == [123]
    assert gate.label("たろう", "123") == "<@123>"
    # Nothing to mention: still no one.
    assert gate.allowed_mentions([]).users is False


def test_labels_are_plain_names_when_off() -> None:
    gate = _gate()
    assert gate.label("たろう", "123") == "たろう"
    assert "@everyone" not in gate.label("@everyone", None)
    assert gate.allowed_mentions(["123"]).users is False

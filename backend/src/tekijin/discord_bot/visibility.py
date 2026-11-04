"""Channel permissions, as the backend needs them (#85).

The backend never talks to Discord, so the bot tells it two things: which
channels everyone in the server can read and which threads are private, and, with
each ``/ask``, which channels the asker can read. Pure functions over what
discord.py hands the bot, so they are tested without Discord.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# How many entries one report carries; the backend accepts up to 1000 channels and
# 20000 threads per call.
REPORT_BATCH = 1000


def readable_channels(guild: Any) -> list[Any]:
    """The channels a conversation can be said in (a forum's posts are threads)."""

    return [
        *guild.text_channels,
        *guild.voice_channels,
        *guild.stage_channels,
        *guild.forums,
    ]


def _can_read(permissions: Any) -> bool:
    return bool(permissions.view_channel and permissions.read_message_history)


def reader_scope(member: Any, channels: Iterable[Any]) -> dict[str, list[int]] | None:
    """What ``member`` may see, for ``/ask``. ``None`` when the asker is not a server
    member (the backend then shows only what everyone can read)."""

    if not hasattr(member, "roles"):
        return None
    readable: list[int] = []
    managers: list[int] = []
    for channel in channels:
        permissions = channel.permissions_for(member)
        if not _can_read(permissions):
            continue
        readable.append(channel.id)
        if permissions.manage_threads:
            managers.append(channel.id)
    return {"readable_channel_ids": readable, "thread_manager_channel_ids": managers}


def channel_report(channels: Iterable[Any], everyone: Any) -> list[dict[str, Any]]:
    """Whether ``everyone`` (the server's @everyone role) can read each channel."""

    return [{"id": c.id, "public": _can_read(c.permissions_for(everyone))} for c in channels]


def thread_report(threads: Iterable[Any]) -> list[dict[str, Any]]:
    return [
        {"id": t.id, "private": getattr(t.type, "name", "") == "private_thread"} for t in threads
    ]


def batches(
    channels: list[dict[str, Any]], threads: list[dict[str, Any]]
) -> list[dict[str, list[dict[str, Any]]]]:
    """``channels`` and ``threads`` cut into reports the backend accepts."""

    reports = [
        {"channels": channels[i : i + REPORT_BATCH], "threads": []}
        for i in range(0, len(channels), REPORT_BATCH)
    ]
    reports += [
        {"channels": [], "threads": threads[i : i + REPORT_BATCH]}
        for i in range(0, len(threads), REPORT_BATCH)
    ]
    return reports

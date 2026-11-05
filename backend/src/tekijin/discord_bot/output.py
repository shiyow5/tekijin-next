"""The one place the bot posts to a channel (#114).

The bot runs in a server shared with other people. Apart from ephemeral replies to
the person who typed a command or pressed a button, everything it writes goes
through :class:`OutputGate`, which allows only the configured output channel
(``discord_output_channel_id``) and the threads under it. Unset, the gate refuses
every post. It also decides who may be pinged: nobody, unless
``discord_mention_responders`` is on, and then only the requested member.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import Any

import discord

#: A request thread archives itself after a week without messages.
_THREAD_ARCHIVE_MINUTES = 10080


class OutputRefused(Exception):
    """The gate does not allow posting there."""


class OutputGate:
    def __init__(self, channel_id: int | None, *, guild_id: int, mention: bool) -> None:
        self.channel_id = channel_id
        self.guild_id = guild_id
        self.mention = mention

    @property
    def enabled(self) -> bool:
        return self.channel_id is not None

    def owns(self, channel: Any) -> bool:
        """True for the output channel itself and the threads under it."""

        if not self.enabled or channel is None:
            return False
        guild = getattr(getattr(channel, "guild", None), "id", None)
        if guild != self.guild_id:
            return False
        return channel.id == self.channel_id or self.is_request_thread(channel)

    def is_request_thread(self, channel: Any) -> bool:
        return (
            self.enabled
            and channel is not None
            and getattr(channel, "parent_id", None) == self.channel_id
            and getattr(getattr(channel, "guild", None), "id", None) == self.guild_id
        )

    def allowed_mentions(self, user_ids: Iterable[str | int] = ()) -> discord.AllowedMentions:
        """Nobody, unless mentions are on; then only ``user_ids``."""

        ids = [int(u) for u in user_ids if u]
        if not self.mention or not ids:
            return discord.AllowedMentions.none()
        return discord.AllowedMentions(
            everyone=False,
            roles=False,
            replied_user=False,
            users=[discord.Object(id=i) for i in ids],
        )

    def label(self, name: str, discord_user_id: str | None) -> str:
        """How a member is named in a post: a mention when mentions are on."""

        if self.mention and discord_user_id:
            return f"<@{int(discord_user_id)}>"
        return discord.utils.escape_mentions(name or "")

    async def _channel(self, client: Any, channel_id: int) -> Any:
        channel = client.get_channel(channel_id)
        if channel is None:
            try:
                channel = await client.fetch_channel(channel_id)
            except discord.DiscordException as exc:
                raise OutputRefused(f"channel {channel_id} is not reachable") from exc
        if not self.owns(channel):
            raise OutputRefused(f"channel {channel_id} is not the output channel")
        return channel

    async def open_thread(
        self,
        client: Any,
        *,
        title: str,
        content: str,
        view: discord.ui.View | None = None,
        mention_ids: Iterable[str | int] = (),
    ) -> Any:
        """Open a public thread in the output channel and write ``content`` in it."""

        if not self.enabled:
            raise OutputRefused("no output channel is configured")
        channel = await self._channel(client, self.channel_id)  # type: ignore[arg-type]
        if channel.id != self.channel_id:  # pragma: no cover - owns() already checked
            raise OutputRefused("threads are opened in the output channel only")
        thread = await channel.create_thread(
            name=title[:100],
            type=discord.ChannelType.public_thread,
            auto_archive_duration=_THREAD_ARCHIVE_MINUTES,
        )
        kwargs: dict[str, Any] = {"allowed_mentions": self.allowed_mentions(mention_ids)}
        if view is not None:
            kwargs["view"] = view
        await thread.send(content, **kwargs)
        return thread

    async def post(
        self, client: Any, channel_id: int, content: str, *, mention_ids: Iterable[str | int] = ()
    ) -> Any:
        """Write ``content`` in the output channel or one of its threads."""

        if not self.enabled:
            raise OutputRefused("no output channel is configured")
        channel = await self._channel(client, channel_id)
        return await channel.send(content, allowed_mentions=self.allowed_mentions(mention_ids))

    async def clear_buttons(self, message: Any) -> None:
        """Take the buttons off one of the bot's own posts in the output channel."""

        if message is None or not self.owns(getattr(message, "channel", None)):
            return
        # On failure the buttons stay; pressing them again is answered as handled.
        with contextlib.suppress(discord.DiscordException):
            await message.edit(view=None)

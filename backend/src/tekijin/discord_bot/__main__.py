"""Run the TEKIJIN Discord bot: ``python -m tekijin.discord_bot`` (Discord D1, #60).

Registers ``/ask`` and the ``/tekijin`` group (``join``, ``optout``, ``optin``, ``add``),
and forwards the server's conversation to the backend (#76): history after where
ingestion left off when it starts, then new, edited and deleted messages as they
happen. Reading message text needs the MESSAGE CONTENT privileged intent, which must
also be switched on for the app in the Discord Developer Portal.

Everything arrives over the Gateway connection, so no public URL is needed. The reply
is ephemeral (only the asker sees it): a question can name customers or internal
trouble, and the channel it was typed in may be wide.
"""

from __future__ import annotations

import asyncio
import logging

import discord
import httpx
from discord import app_commands

from tekijin.config import get_settings
from tekijin.discord_bot.attachments import (
    AddInputError,
    check_attachment,
    compose_add,
    decode_attachment,
)
from tekijin.discord_bot.backend import (
    add_backend,
    ask_backend,
    discord_user,
    get_cursors,
    member_backend,
    post_deletes,
    post_messages,
)
from tekijin.discord_bot.format import format_add_reply, format_member_reply, format_reply
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

logger = logging.getLogger("tekijin.discord_bot")

# A graph run makes several LLM calls in sequence (C1, C2, the additive answer, the C7
# draft), each allowed its own timeout, and the endpoint answers only at the end.
# Discord accepts a follow-up for 15 minutes after ``defer``; wait almost that long.
_BACKEND_TIMEOUT_S = 14 * 60.0
# How often collected message changes are forwarded (#76).
_FLUSH_INTERVAL_S = 10.0
# How long to wait before asking the backend again where history left off.
_CURSOR_RETRY_S = 30.0
# How long to wait before reading history again when part of it could not be listed.
_HISTORY_RETRY_S = 60.0


def _user(interaction: discord.Interaction) -> dict:
    return discord_user(interaction.user.id, interaction.guild_id, interaction.user.display_name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    if (
        not settings.discord_bot_token
        or not settings.discord_bot_secret
        or not settings.discord_guild_id
    ):
        raise SystemExit(
            "TEKIJIN_DISCORD_BOT_TOKEN・TEKIJIN_DISCORD_BOT_SECRET・"
            "TEKIJIN_DISCORD_GUILD_ID を設定してください"
        )

    guild_id = settings.discord_guild_id
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)
    http = httpx.AsyncClient(timeout=_BACKEND_TIMEOUT_S)
    excluded = parse_ids(settings.discord_excluded_channel_ids)
    backend = {"base_url": settings.discord_backend_url, "secret": settings.discord_bot_secret}

    async def send_messages(messages: list[dict]) -> bool:
        return await post_messages(http, guild_id=guild_id, messages=messages, **backend)

    async def send_deletes(message_ids: list[int]) -> bool:
        return await post_deletes(http, guild_id=guild_id, message_ids=message_ids, **backend)

    batcher = Batcher(send_messages, send_deletes)
    # Strong references: the event loop keeps only weak ones to running tasks.
    tasks: dict[str, asyncio.Task] = {}
    # Clear until the bot knows where history left off; nothing is stored meanwhile.
    ready = asyncio.Event()

    @tree.command(name="ask", description="TEKIJINに質問する")
    @app_commands.describe(question="質問の内容")
    async def ask(interaction: discord.Interaction, question: app_commands.Range[str, 1, 2000]):
        await interaction.response.defer(thinking=True, ephemeral=True)
        outcome = await ask_backend(
            http,
            base_url=settings.discord_backend_url,
            secret=settings.discord_bot_secret,
            user=_user(interaction),
            question=question,
        )
        reply = format_reply(outcome, web_url=outcome.get("web_url") or settings.discord_web_url)
        await interaction.followup.send(reply, ephemeral=True)

    group = app_commands.Group(name="tekijin", description="TEKIJINへの参加と設定")

    async def member_action(interaction: discord.Interaction, action: str) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        result = await member_backend(
            http,
            base_url=settings.discord_backend_url,
            secret=settings.discord_bot_secret,
            user=_user(interaction),
            action=action,
        )
        await interaction.followup.send(format_member_reply(result), ephemeral=True)

    @group.command(name="join", description="TEKIJINに参加し、Webのログイン用URLを受け取る")
    async def join(interaction: discord.Interaction) -> None:
        await member_action(interaction, "join")

    @group.command(name="optout", description="詳しい人として推薦される対象から外れる")
    async def optout(interaction: discord.Interaction) -> None:
        await member_action(interaction, "optout")

    @group.command(name="optin", description="詳しい人として推薦される対象に戻る")
    async def optin(interaction: discord.Interaction) -> None:
        await member_action(interaction, "optin")

    @group.command(
        name="add", description="文章やファイルをTEKIJINの知識に追加する(参加した人のみ)"
    )
    @app_commands.describe(
        text="追加する文章(4000字まで。ファイルだけでもよい)",
        file="テキスト(.txt)かMarkdown(.md)のファイル(200KBまで)",
        title="題名(任意。無ければファイル名)",
        url="元の文書の置き場所のURL(任意。同じURLを送り直すと更新になる)",
    )
    async def add(
        interaction: discord.Interaction,
        text: app_commands.Range[str, 1, 4000] | None = None,
        file: discord.Attachment | None = None,
        title: app_commands.Range[str, 1, 200] | None = None,
        url: app_commands.Range[str, 1, 2000] | None = None,
    ) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        try:
            file_text = None
            if file is not None:
                check_attachment(file.filename, file.size)
                file_text = decode_attachment(await file.read())
            body, note_title = compose_add(
                text=text,
                title=title,
                file_name=file.filename if file is not None else None,
                file_text=file_text,
            )
        except AddInputError as exc:
            await interaction.followup.send(str(exc), ephemeral=True)
            return
        except discord.HTTPException:
            logger.warning("could not download the attachment", exc_info=True)
            await interaction.followup.send(
                "ファイルを読み込めませんでした。もう一度お試しください。", ephemeral=True
            )
            return
        result = await add_backend(
            http,
            base_url=settings.discord_backend_url,
            secret=settings.discord_bot_secret,
            user=_user(interaction),
            text=body,
            title=note_title,
            url=url,
        )
        await interaction.followup.send(format_add_reply(result), ephemeral=True)

    tree.add_command(group)

    def forward(message: discord.Message) -> None:
        if message.guild is None or message.guild.id != guild_id:
            return
        payload = message_payload(message)
        if payload is not None and not is_excluded(payload, excluded):
            batcher.add(payload)

    @client.event
    async def on_message(message: discord.Message) -> None:
        forward(message)

    @client.event
    async def on_raw_message_edit(payload: discord.RawMessageUpdateEvent) -> None:
        if payload.guild_id != guild_id:
            return
        # The edit event may carry only part of the message; read it whole.
        try:
            channel = client.get_channel(payload.channel_id) or await client.fetch_channel(
                payload.channel_id
            )
            message = await channel.fetch_message(payload.message_id)  # type: ignore[union-attr]
        except discord.NotFound:
            batcher.delete([payload.message_id])  # gone since the edit
            return
        except discord.Forbidden:
            return  # the bot may not read this channel
        except discord.DiscordException:
            # Passing trouble: read it from history once sending works again, or
            # the text before the edit would stay searchable.
            logger.warning("could not read edited message %s", payload.message_id, exc_info=True)
            batcher.reread(payload.channel_id, payload.message_id)
            return
        action = edit_action(message, excluded)
        if action is None:
            return
        kind, value = action
        if kind == "upsert":
            batcher.add(value)
        else:
            batcher.delete([value])

    @client.event
    async def on_raw_message_delete(payload: discord.RawMessageDeleteEvent) -> None:
        if payload.guild_id == guild_id:
            batcher.delete([payload.message_id])

    @client.event
    async def on_raw_bulk_message_delete(payload: discord.RawBulkMessageDeleteEvent) -> None:
        if payload.guild_id == guild_id:
            batcher.delete(payload.message_ids)

    # Set when the Gateway reconnects while history is being read: another pass
    # must follow, from the points where history left off as of the reconnect.
    rerun = asyncio.Event()

    async def flush_if_ready() -> None:
        if ready.is_set():
            await batcher.flush()

    async def history_pass() -> bool:
        """One pass over the server's history. False when part of it could not be
        listed for a passing reason and the pass should be repeated."""

        stored = await fetch_until_ready(
            lambda: get_cursors(http, guild_id=guild_id, **backend), retry_after=_CURSOR_RETRY_S
        )
        ready.set()
        guild = client.get_guild(guild_id)
        if guild is None:
            logger.warning("skipped reading history: the bot is not in the configured server")
            return True
        # Where queued messages had to be refused, read again from before them.
        cursors = merge_cursors(stored, batcher.take_rewinds())
        me = guild.me
        channels: list = [
            c
            for c in guild.text_channels
            if c.permissions_for(me).read_message_history and c.permissions_for(me).view_channel
        ]
        channels.extend(guild.threads)
        parents: list[discord.TextChannel | discord.ForumChannel] = [
            *guild.text_channels,
            *guild.forums,
        ]
        seen = {c.id for c in channels}
        complete = True
        for channel in parents:
            # Public archives, then private ones: all of them when the bot may
            # manage threads, otherwise those it has joined.
            listings = [channel.archived_threads(limit=None)]
            if isinstance(channel, discord.TextChannel):  # forum posts are never private
                listings.append(channel.archived_threads(limit=None, private=True))
                listings.append(channel.archived_threads(limit=None, private=True, joined=True))
            for listing in listings:
                try:
                    async for thread in listing:
                        if thread.id not in seen:
                            seen.add(thread.id)
                            channels.append(thread)
                except discord.Forbidden:
                    continue  # no permission for this listing
                except discord.DiscordException:
                    logger.warning("listing archived threads failed; will retry", exc_info=True)
                    complete = False
        queued = await backfill(
            channels,
            cursors=cursors,
            batcher=batcher,
            excluded=excluded,
            after=discord.Object,
            # No permission is final; anything else (a timeout, a 5xx) is retried.
            is_permanent=lambda exc: isinstance(exc, discord.Forbidden),
            flush=flush_if_ready,
        )
        await flush_if_ready()
        logger.info("read history: %d message(s) from %d channel(s)", queued, len(channels))
        return complete

    async def read_history() -> None:
        while True:
            complete = await history_pass()
            if rerun.is_set():
                rerun.clear()
                continue
            if complete:
                return
            await asyncio.sleep(_HISTORY_RETRY_S)

    @client.event
    async def on_ready() -> None:
        # Only the configured server is served (#66), so sync there only: a
        # guild-scoped sync also shows the commands immediately.
        guild = discord.Object(id=guild_id)
        tree.copy_global_to(guild=guild)
        synced = await tree.sync(guild=guild)
        logger.info("logged in as %s; synced %d command(s)", client.user, len(synced))
        # on_ready fires again after a reconnect. The flush loop runs once; history
        # is read again, which costs only what was missed meanwhile.
        if "flush" not in tasks:
            tasks["flush"] = asyncio.create_task(
                batcher.run(_FLUSH_INTERVAL_S, start_history, ready=ready)
            )
        start_history(reconnected=True)

    def start_history(reconnected: bool = False) -> None:
        """Read history, or — after a reconnect — make sure it is read again.

        After a reconnect nothing is sent until the points where history left off
        have been read again, so messages arriving now cannot move them past what
        was posted while the bot was away. If a pass is already running, another
        one follows it rather than the request being dropped.
        """

        if reconnected:
            ready.clear()
        if "history" in tasks and not tasks["history"].done():
            if reconnected:
                rerun.set()
            return
        tasks["history"] = asyncio.create_task(read_history())

    client.run(settings.discord_bot_token, log_handler=None)


if __name__ == "__main__":
    main()

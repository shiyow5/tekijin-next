"""Run the TEKIJIN Discord bot: ``python -m tekijin.discord_bot`` (Discord D1, #60).

Registers ``/ask`` and the ``/tekijin`` group (``join``, ``intro``, ``optout``,
``optin``, ``add``, ``review``), sends hand-offs into request threads of the one
output channel (#114, ``requests.py``; nothing is posted while it is unset), and
forwards the server's conversation to the backend (#76):
history after where
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
    post_reaction,
    post_stream_deletes,
    post_visibility,
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
from tekijin.discord_bot.output import OutputGate
from tekijin.discord_bot.reactions import ReactionRelay
from tekijin.discord_bot.requests import RequestUI, install, send_view
from tekijin.discord_bot.visibility import (
    VisibilityReporter,
    channel_report,
    readable_channels,
    reader_scope,
    thread_report,
)
from tekijin.knowledge.solve_signals import is_solve_emoji

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
# How often to send again what the backend did not take of a visibility report.
_VISIBILITY_RETRY_S = 30.0
# How often to send again the check marks the backend could not take yet (#113).
_REACTION_RETRY_S = 30.0


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
    # #114: the only channel the bot posts to. Its request threads are not read
    # into TEKIJIN's sources: what is answered there is kept as answers instead.
    gate = OutputGate(
        settings.discord_output_channel_id,
        guild_id=guild_id,
        mention=settings.discord_mention_responders,
    )
    if gate.channel_id is not None:
        excluded.add(gate.channel_id)
    requests_ui = RequestUI(http=http, gate=gate, **backend)

    async def send_messages(messages: list[dict]) -> bool:
        return await post_messages(http, guild_id=guild_id, messages=messages, **backend)

    async def send_deletes(message_ids: list[int]) -> bool:
        return await post_deletes(http, guild_id=guild_id, message_ids=message_ids, **backend)

    async def send_stream_deletes(stream_ids: list[int]) -> bool:
        return await post_stream_deletes(http, guild_id=guild_id, stream_ids=stream_ids, **backend)

    async def send_visibility(part: dict) -> bool:
        return await post_visibility(http, guild_id=guild_id, report=part, **backend)

    # #85: what could not be sent is kept and sent again (see retry_visibility).
    reporter = VisibilityReporter(send_visibility)

    async def report(channels: list[dict], threads: list[dict]) -> None:
        if not await reporter.report(channels, threads):
            logger.warning("could not report channel visibility; will retry")

    async def retry_visibility() -> None:
        while True:
            await asyncio.sleep(_VISIBILITY_RETRY_S)
            if reporter.pending and not await reporter.flush():
                logger.warning("could not report channel visibility; will retry")

    async def report_channels(guild: discord.Guild) -> None:
        await report(channel_report(readable_channels(guild), guild.default_role), [])

    # Clear until the bot knows where history left off; nothing is stored meanwhile.
    ready = asyncio.Event()
    batcher = Batcher(
        send_messages, send_deletes, send_stream_deletes=send_stream_deletes, gate=ready
    )
    # Strong references: the event loop keeps only weak ones to running tasks.
    tasks: dict[str, asyncio.Task] = {}

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
            # #85: only conversations in channels this member can read.
            scope=reader_scope(interaction.user, readable_channels(interaction.guild))
            if interaction.guild is not None
            else None,
        )
        # #114: a hand-off that can be sent from Discord gets the 「依頼する」 button.
        session_id = outcome.get("session_id") if gate.enabled else None
        reply = format_reply(
            outcome,
            web_url=outcome.get("web_url") or settings.discord_web_url,
            send_channel_id=gate.channel_id if session_id else None,
        )
        if session_id:
            await interaction.followup.send(reply, ephemeral=True, view=send_view(session_id))
        else:
            await interaction.followup.send(reply, ephemeral=True)

    group = app_commands.Group(name="tekijin", description="TEKIJINへの参加と設定")

    async def member_action(
        interaction: discord.Interaction, action: str, intro: str | None = None
    ) -> None:
        await interaction.response.defer(thinking=True, ephemeral=True)
        result = await member_backend(
            http,
            base_url=settings.discord_backend_url,
            secret=settings.discord_bot_secret,
            user=_user(interaction),
            action=action,
            intro=intro,
        )
        await interaction.followup.send(format_member_reply(result), ephemeral=True)

    @group.command(name="join", description="TEKIJINに参加し、Webのログイン用URLを受け取る")
    @app_commands.describe(
        intro="自己紹介(任意。1000字まで。得意なことや担当を書くと推薦に使われる)"
    )
    async def join(
        interaction: discord.Interaction,
        intro: app_commands.Range[str, 1, 1000] | None = None,
    ) -> None:
        await member_action(interaction, "join", intro)

    @group.command(name="intro", description="自己紹介を登録する(前に登録したものは置き換わる)")
    @app_commands.describe(text="自己紹介(1000字まで。得意なことや担当を書くと推薦に使われる)")
    async def intro(
        interaction: discord.Interaction, text: app_commands.Range[str, 1, 1000]
    ) -> None:
        await member_action(interaction, "intro", text)

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

    install(client, group, requests_ui)
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
        if message.guild is not None and message.guild.id == guild_id:
            await requests_ui.on_thread_message(message)

    @client.event
    async def on_raw_message_edit(payload: discord.RawMessageUpdateEvent) -> None:
        if payload.guild_id != guild_id:
            return
        # #114: an edit in a request thread corrects the answer it became; these
        # threads are not ingested, so nothing below applies to them.
        if gate.is_request_thread(client.get_channel(payload.channel_id)):
            await requests_ui.on_thread_edit(client, payload.channel_id, payload.message_id)
            return
        # Taken before fetching: a deletion or a newer edit arriving while the
        # fetch is in flight makes its result stale, and the batcher drops it.
        token = batcher.begin_edit(payload.message_id)
        # The edit event may carry only part of the message; read it whole.
        try:
            channel = client.get_channel(payload.channel_id) or await client.fetch_channel(
                payload.channel_id
            )
            message = await channel.fetch_message(payload.message_id)  # type: ignore[union-attr]
        except discord.NotFound:
            batcher.delete([payload.message_id], token=token)  # gone since the edit
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
            batcher.add(value, token=token)
        else:
            batcher.delete([value], token=token)

    async def forget_answers(channel_id: int, message_ids: list[int]) -> None:
        """#114: a deletion in a request thread removes the answers it became. An
        uncached channel may be one; the backend ignores ids it never kept."""

        channel = client.get_channel(channel_id)
        if gate.enabled and (channel is None or gate.is_request_thread(channel)):
            await requests_ui.on_thread_delete(message_ids, thread_id=channel_id)

    @client.event
    async def on_raw_message_delete(payload: discord.RawMessageDeleteEvent) -> None:
        if payload.guild_id == guild_id:
            batcher.delete([payload.message_id])
            await forget_answers(payload.channel_id, [payload.message_id])

    @client.event
    async def on_raw_bulk_message_delete(payload: discord.RawBulkMessageDeleteEvent) -> None:
        if payload.guild_id == guild_id:
            batcher.delete(payload.message_ids)
            await forget_answers(payload.channel_id, list(payload.message_ids))

    # #113: a check mark may mark a conversation solved. Sent only after the
    # queued messages were stored, and kept to try again (discord_bot/reactions.py).
    async def send_reaction(reaction: dict) -> str:
        return await post_reaction(
            http,
            guild_id=guild_id,
            channel_id=int(reaction["channel_id"]),
            message_id=int(reaction["message_id"]),
            user_id=int(reaction["user_id"]),
            emoji=str(reaction["emoji"]),
            **backend,
        )

    reactions = ReactionRelay(send_reaction, batcher.flush, ready=ready)

    @client.event
    async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
        if payload.guild_id != guild_id or not is_solve_emoji(str(payload.emoji)):
            return
        reactions.add(
            {
                "channel_id": payload.channel_id,
                "message_id": payload.message_id,
                "user_id": payload.user_id,
                "emoji": str(payload.emoji),
            }
        )
        await reactions.deliver()

    # Deleting a channel or a thread reports no per-message deletions.
    @client.event
    async def on_guild_channel_delete(channel: discord.abc.GuildChannel) -> None:
        if channel.guild.id == guild_id:
            batcher.delete_streams([channel.id])

    @client.event
    async def on_raw_thread_delete(payload: discord.RawThreadDeleteEvent) -> None:
        if payload.guild_id == guild_id:
            batcher.delete_streams([payload.thread_id])

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
        # Text channels, and the text chat every voice and stage channel has.
        candidates: list[discord.abc.GuildChannel] = [
            *guild.text_channels,
            *guild.voice_channels,
            *guild.stage_channels,
        ]
        channels: list = [
            c
            for c in candidates
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
        await report(
            channel_report(readable_channels(guild), guild.default_role),
            thread_report(c for c in channels if isinstance(c, discord.Thread)),
        )
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
    async def on_guild_channel_create(channel: discord.abc.GuildChannel) -> None:
        if channel.guild.id == guild_id:
            await report_channels(channel.guild)

    @client.event
    async def on_guild_channel_update(
        before: discord.abc.GuildChannel, after: discord.abc.GuildChannel
    ) -> None:
        # A category's permissions reach its channels, so all of them are reported.
        if after.guild.id == guild_id:
            await report_channels(after.guild)

    @client.event
    async def on_guild_role_update(before: discord.Role, after: discord.Role) -> None:
        if after.guild.id == guild_id and after.is_default():
            await report_channels(after.guild)

    @client.event
    async def on_thread_create(thread: discord.Thread) -> None:
        if thread.guild.id == guild_id:
            await report([], thread_report([thread]))

    @client.event
    async def on_thread_join(thread: discord.Thread) -> None:
        # An existing thread the bot can newly see (thread_create is not sent).
        if thread.guild.id == guild_id:
            await report([], thread_report([thread]))

    @client.event
    async def on_thread_update(before: discord.Thread, after: discord.Thread) -> None:
        if after.guild.id == guild_id:
            await report([], thread_report([after]))

    @client.event
    async def on_ready() -> None:
        # Pause sending before anything else: during the awaits below, a new
        # message could otherwise be stored and move the point where history
        # left off past what was posted while the bot was away.
        ready.clear()
        # Only the configured server is served (#66), so sync there only: a
        # guild-scoped sync also shows the commands immediately.
        guild = discord.Object(id=guild_id)
        tree.copy_global_to(guild=guild)
        try:
            synced = await tree.sync(guild=guild)
            logger.info("logged in as %s; synced %d command(s)", client.user, len(synced))
        except discord.DiscordException:
            # The commands stay as last synced; ingestion must still resume below.
            logger.warning("could not sync the commands", exc_info=True)
        # on_ready fires again after a reconnect. The flush loop runs once; history
        # is read again, which costs only what was missed meanwhile.
        if "visibility" not in tasks:
            tasks["visibility"] = asyncio.create_task(retry_visibility())
        if "reactions" not in tasks:
            tasks["reactions"] = asyncio.create_task(reactions.run(_REACTION_RETRY_S))
        if "outbox" not in tasks and gate.enabled:
            tasks["outbox"] = asyncio.create_task(requests_ui.run_outbox(client))
        guild_now = client.get_guild(guild_id)
        if (
            gate.enabled
            and guild_now is not None
            and ("replay" not in tasks or tasks["replay"].done())
        ):
            tasks["replay"] = asyncio.create_task(requests_ui.replay_all(client, guild_now))
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

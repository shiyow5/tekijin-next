"""Run the TEKIJIN Discord bot: ``python -m tekijin.discord_bot`` (Discord D1, #60).

Registers one slash command, ``/ask``. Slash commands arrive over the Gateway as
interactions, so no privileged intent (MESSAGE_CONTENT) and no public URL are needed.
The reply is ephemeral (only the asker sees it): a question can name customers or
internal trouble, and the channel it was typed in may be wide.
"""

from __future__ import annotations

import logging

import discord
import httpx
from discord import app_commands

from tekijin.config import get_settings
from tekijin.discord_bot.backend import ask_backend
from tekijin.discord_bot.format import format_reply

logger = logging.getLogger("tekijin.discord_bot")

# A graph run makes several LLM calls in sequence (C1, C2, the additive answer, the C7
# draft), each allowed its own timeout, and the endpoint answers only at the end.
# Discord accepts a follow-up for 15 minutes after ``defer``; wait almost that long.
_BACKEND_TIMEOUT_S = 14 * 60.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()
    if not settings.discord_bot_token or not settings.discord_bot_secret:
        raise SystemExit(
            "TEKIJIN_DISCORD_BOT_TOKEN と TEKIJIN_DISCORD_BOT_SECRET を設定してください"
        )

    client = discord.Client(intents=discord.Intents.default())
    tree = app_commands.CommandTree(client)
    http = httpx.AsyncClient(timeout=_BACKEND_TIMEOUT_S)

    @tree.command(name="ask", description="TEKIJINに質問する")
    @app_commands.describe(question="質問の内容")
    async def ask(interaction: discord.Interaction, question: app_commands.Range[str, 1, 2000]):
        await interaction.response.defer(thinking=True, ephemeral=True)
        outcome = await ask_backend(
            http,
            base_url=settings.discord_backend_url,
            secret=settings.discord_bot_secret,
            discord_user_id=interaction.user.id,
            question=question,
        )
        reply = format_reply(outcome, web_url=outcome.get("web_url") or settings.discord_web_url)
        await interaction.followup.send(reply, ephemeral=True)

    @client.event
    async def on_ready() -> None:
        # A guild-scoped sync shows the command immediately; a global one can take
        # up to an hour to appear, so use the guild id while developing.
        if settings.discord_guild_id:
            guild = discord.Object(id=settings.discord_guild_id)
            tree.copy_global_to(guild=guild)
            synced = await tree.sync(guild=guild)
        else:
            synced = await tree.sync()
        logger.info("logged in as %s; synced %d command(s)", client.user, len(synced))

    client.run(settings.discord_bot_token, log_handler=None)


if __name__ == "__main__":
    main()

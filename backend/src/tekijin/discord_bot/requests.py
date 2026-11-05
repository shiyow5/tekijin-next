"""Buttons, request threads and ``/tekijin review`` in the bot (#114).

* 「依頼する」 on the ephemeral ``/ask`` reply sends the hand-off.
* The backend queues the request thread; :meth:`RequestUI.deliver` posts it through
  the output gate with 承諾 / 辞退 / 「自分より適任がいる」.
* What the two parties write in a request thread goes to the backend.
* ``/tekijin review`` lists the drafts waiting for review with 確認 / 差し戻し.

Every button is a :class:`discord.ui.DynamicItem` whose ``custom_id`` carries what
it acts on, so buttons keep working after the bot restarts. A ``custom_id`` comes
from the Discord client and is only a claim: the backend checks who pressed it and
whether the action still applies. Replies to the presser are ephemeral; anything
else is posted only through :class:`~tekijin.discord_bot.output.OutputGate`.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from typing import Any

import discord
import httpx
from discord import app_commands

from tekijin.discord_bot.backend import discord_user, get_outbox, internal_post
from tekijin.discord_bot.output import OutputGate, OutputRefused
from tekijin.discord_bot.request_format import (
    clip,
    format_action_reply,
    format_request,
    format_review,
    format_review_action_reply,
    format_send_reply,
    request_title,
)

logger = logging.getLogger(__name__)

SEND_TEMPLATE = r"tk:send:(?P<session_id>discord_[0-9a-f]{32})"
ACTION_TEMPLATE = r"tk:req:(?P<action>accept|decline|refer):(?P<rec>[0-9]{1,12})"
REFER_TEMPLATE = r"tk:ref:(?P<rec>[0-9]{1,12})"
REVIEW_TEMPLATE = r"tk:rev:(?P<action>verify|back):(?P<article>[0-9]{1,12})"
REVIEW_PAGE_TEMPLATE = (
    r"tk:revn:(?P<offset>[0-9]{1,6}):(?P<shown>[0-9]{1,12}(?:\.[0-9]{1,12}){0,3})"
)
#: Drafts per ``/tekijin review`` page (the backend's page size): one row of
#: buttons each, and the fifth row for 「次へ」.
REVIEW_PAGE = 4

#: How often the outbox is read when nothing asks for it sooner.
OUTBOX_INTERVAL_S = 5.0
_THREAD_MESSAGE_MAX = 4000
#: Thread messages kept for sending again while the backend is away.
_RETRY_MAX = 1000
#: How far back each open request thread is read again after a (re)connect.
_REPLAY_LIMIT = 200
_UNAVAILABLE = "いまは処理できません。しばらくしてから、もう一度お試しください。"


def send_id(session_id: str) -> str:
    return f"tk:send:{session_id}"


def action_id(action: str, recommendation_id: int) -> str:
    return f"tk:req:{action}:{recommendation_id}"


def refer_id(recommendation_id: int) -> str:
    return f"tk:ref:{recommendation_id}"


def review_id(action: str, article_id: int) -> str:
    return f"tk:rev:{action}:{article_id}"


def review_page_id(offset: int, shown: list[int]) -> str:
    """「次へ」: where the next page starts, and the drafts the current page showed
    (any of them checked meanwhile has left the queue, so the start moves back)."""

    return f"tk:revn:{offset}:{'.'.join(str(i) for i in shown)}"


def _button(label: str, custom_id: str, style: discord.ButtonStyle) -> discord.ui.Button:
    return discord.ui.Button(label=label, custom_id=custom_id, style=style)


def send_view(session_id: str) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    view.add_item(_button("依頼する", send_id(session_id), discord.ButtonStyle.primary))
    return view


def request_view(recommendation_id: int) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    view.add_item(
        _button("承諾する", action_id("accept", recommendation_id), discord.ButtonStyle.success)
    )
    view.add_item(
        _button("辞退する", action_id("decline", recommendation_id), discord.ButtonStyle.secondary)
    )
    view.add_item(
        _button(
            "自分より適任がいる",
            action_id("refer", recommendation_id),
            discord.ButtonStyle.secondary,
        )
    )
    return view


def refer_view(recommendation_id: int) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    view.add_item(
        discord.ui.UserSelect(
            custom_id=refer_id(recommendation_id), placeholder="より適任の人を選ぶ", max_values=1
        )
    )
    return view


def review_view(
    articles: list[dict[str, Any]], *, next_offset: int | None = None
) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    for row, article in enumerate(articles[:REVIEW_PAGE]):
        aid = int(article["id"])
        view.add_item(
            discord.ui.Button(
                label=f"#{aid} を確認済みにする",
                custom_id=review_id("verify", aid),
                style=discord.ButtonStyle.success,
                row=row,
            )
        )
        view.add_item(
            discord.ui.Button(
                label=f"#{aid} を差し戻す",
                custom_id=review_id("back", aid),
                style=discord.ButtonStyle.secondary,
                row=row,
            )
        )
    if next_offset is not None:
        view.add_item(
            discord.ui.Button(
                label="次へ",
                custom_id=review_page_id(
                    next_offset, [int(a["id"]) for a in articles[:REVIEW_PAGE]]
                ),
                style=discord.ButtonStyle.primary,
                row=4,
            )
        )
    return view


def _who(interaction: Any) -> dict[str, Any]:
    return discord_user(interaction.user.id, interaction.guild_id, interaction.user.display_name)


class RequestUI:
    """What the bot does for requests and review. One per bot process."""

    def __init__(
        self, *, http: httpx.AsyncClient, base_url: str, secret: str, gate: OutputGate
    ) -> None:
        self.http = http
        self.base_url = base_url
        self.secret = secret
        self.gate = gate
        self.kick = asyncio.Event()
        self._lock = asyncio.Lock()
        # Thread messages the backend did not take; sent again with the outbox.
        self._retry: dict[str, dict[str, Any]] = {}
        # Deleted request-thread messages the backend has not heard about yet.
        self._deletes: set[str] = set()
        # Posted outbox items whose result the backend has not taken yet.
        self._acks: dict[int, dict[str, Any]] = {}

    async def _call(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return await internal_post(
            self.http, base_url=self.base_url, secret=self.secret, path=path, body=body
        )

    async def _reply(self, interaction: Any, text: str, **kwargs: Any) -> None:
        await interaction.followup.send(
            text, ephemeral=True, allowed_mentions=discord.AllowedMentions.none(), **kwargs
        )

    # -- buttons ---------------------------------------------------------- #
    async def on_send(self, interaction: Any, session_id: str) -> None:
        """「依頼する」 on the ``/ask`` reply."""

        await interaction.response.defer(ephemeral=True, thinking=True)
        if not self.gate.enabled:
            await self._reply(interaction, _UNAVAILABLE)
            return
        result = await self._call("request", {**_who(interaction), "session_id": session_id})
        await self._reply(interaction, format_send_reply(result, channel_id=self.gate.channel_id))
        if result.get("status") == "requested":
            self.kick.set()

    async def on_action(self, interaction: Any, recommendation_id: int, action: str) -> None:
        """承諾 / 辞退 / 「自分より適任がいる」 in a request thread."""

        if action == "refer":
            # Pick the person first; the backend checks the presser when it is sent.
            await interaction.response.send_message(
                "この依頼を回したい人を選んでください。",
                view=refer_view(recommendation_id),
                ephemeral=True,
            )
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await self._call(
            "request/action",
            {**_who(interaction), "recommendation_id": recommendation_id, "action": action},
        )
        await self._reply(interaction, format_action_reply(result))
        if result.get("status") in ("accepted", "declined", "stale", "gone"):
            await self.gate.clear_buttons(getattr(interaction, "message", None))
        self.kick.set()

    async def on_refer(self, interaction: Any, recommendation_id: int) -> None:
        """The person picked after 「自分より適任がいる」."""

        await interaction.response.defer(ephemeral=True, thinking=True)
        values = (getattr(interaction, "data", None) or {}).get("values") or []
        chosen = str(values[0]) if values else ""
        body: dict[str, Any] = {
            **_who(interaction),
            "recommendation_id": recommendation_id,
            "action": "refer",
        }
        if re.fullmatch(r"[0-9]{1,20}", chosen):
            body["referred_discord_user_id"] = chosen
        result = await self._call("request/action", body)
        await self._reply(interaction, format_action_reply(result))
        self.kick.set()

    async def on_review(
        self, interaction: Any, offset: int = 0, shown: list[int] | None = None
    ) -> None:
        """``/tekijin review``, or 「次へ」 from its page at ``offset``."""

        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await self._call(
            "review", {**_who(interaction), "offset": offset, "shown": shown or []}
        )
        articles = (result.get("articles") or []) if result.get("status") == "ok" else []
        following = int(result.get("offset") or 0) + len(articles)
        more = following < int(result.get("total") or 0)
        kwargs = (
            {"view": review_view(articles, next_offset=following if more else None)}
            if articles
            else {}
        )
        await self._reply(interaction, format_review(result), **kwargs)

    async def on_review_action(self, interaction: Any, article_id: int, action: str) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await self._call(
            "review/action",
            {
                **_who(interaction),
                "article_id": article_id,
                "action": "verify" if action == "verify" else "send_back",
            },
        )
        await self._reply(interaction, format_review_action_reply(result))

    # -- messages in request threads -------------------------------------- #
    async def on_thread_message(self, message: Any, *, edited: bool = False) -> None:
        """Forward what someone wrote (or edited) in a request thread."""

        if message.guild is None or message.author.bot:
            return
        if not self.gate.is_request_thread(message.channel):
            return
        content = (message.content or "").strip()
        if not content:
            return
        body = {
            **discord_user(message.author.id, message.guild.id, message.author.display_name),
            "thread_id": str(message.channel.id),
            "message_id": str(message.id),
            "body": content[:_THREAD_MESSAGE_MAX],
        }
        if edited:
            body["edited"] = True
        await self._forward(body)

    async def on_thread_delete(self, message_ids: list[int]) -> None:
        """Messages deleted from a request thread: the backend removes the answers
        they became. Kept and sent again while the backend is away."""

        self._deletes.update(str(i) for i in message_ids)
        await self.retry_deletes()

    async def retry_deletes(self) -> None:
        pending = sorted(self._deletes)
        for start in range(0, len(pending), 100):
            chunk = pending[start : start + 100]
            result = await self._call(
                "thread-message/delete",
                {"guild_id": str(self.gate.guild_id), "message_ids": chunk},
            )
            if "status" not in result:
                logger.warning("could not forward %d deletion(s); will retry", len(chunk))
                return
            self._deletes.difference_update(chunk)

    async def on_thread_edit(self, client: Any, channel_id: int, message_id: int) -> None:
        """An edit in a request thread: the backend corrects the kept answer."""

        try:
            channel = client.get_channel(channel_id) or await client.fetch_channel(channel_id)
            if not self.gate.is_request_thread(channel):
                return
            message = await channel.fetch_message(message_id)
        except discord.DiscordException:
            logger.warning("could not read edited request thread message %s", message_id)
            return
        await self.on_thread_message(message, edited=True)

    async def _forward(self, body: dict[str, Any]) -> bool:
        """Send one thread message; keep it for later when the backend is away.

        The backend remembers every message id it handled, so sending one again
        (a retry, or a replay after a restart) never makes a second answer."""

        result = await self._call("thread-message", body)
        if "status" in result:
            self._retry.pop(body["message_id"], None)
            return True
        logger.warning(
            "could not forward request thread message %s; will retry", body["message_id"]
        )
        if len(self._retry) < _RETRY_MAX or body["message_id"] in self._retry:
            self._retry[body["message_id"]] = body
        return False

    async def retry_messages(self) -> None:
        for body in list(self._retry.values()):
            if not await self._forward(body):
                return  # still away; try the rest next time

    async def replay(self, threads: Any) -> None:
        """Read the open request threads again, so messages written while the bot
        was away (or before a failed send) still reach the backend."""

        for thread in threads:
            if not self.gate.is_request_thread(thread):
                continue
            try:
                async for message in thread.history(limit=_REPLAY_LIMIT, oldest_first=True):
                    await self.on_thread_message(message)
            except discord.DiscordException:
                logger.warning("could not read request thread %s again", thread.id, exc_info=True)

    # -- the outbox -------------------------------------------------------- #
    async def deliver(self, client: Any) -> int:
        """Post what the backend queued. Returns how many items were posted."""

        if not self.gate.enabled:
            return 0
        async with self._lock:
            await self.retry_messages()
            await self.retry_deletes()
            await self._retry_acks()
            items = await get_outbox(self.http, base_url=self.base_url, secret=self.secret)
            posted = 0
            for item in items:
                outbox_id = int(item["id"])
                if outbox_id in self._acks:
                    continue  # posted already; only the backend has not heard
                ok, extra = await self._deliver_one(client, item)
                posted += ok
                await self._ack(outbox_id, {"ok": ok, **extra})
            return posted

    async def _ack(self, outbox_id: int, result: dict[str, Any]) -> None:
        """Report one item's result. A successful post whose report does not reach
        the backend is kept and reported again, never posted again (a second
        request thread would leave the first one cut off from the hand-off)."""

        reply = await self._call(f"outbox/{outbox_id}", result)
        if "status" in reply:
            self._acks.pop(outbox_id, None)
        elif result.get("ok"):
            self._acks[outbox_id] = result

    async def _retry_acks(self) -> None:
        for outbox_id, result in list(self._acks.items()):
            await self._ack(outbox_id, result)

    async def _deliver_one(self, client: Any, item: dict[str, Any]) -> tuple[bool, dict]:
        payload = item.get("payload") or {}
        try:
            if item.get("kind") == "request":
                rec = int(payload["recommendation_id"])
                responder = payload.get("responder_discord_id")
                thread = await self.gate.open_thread(
                    client,
                    title=request_title(rec),
                    content=format_request(payload, self.gate),
                    view=request_view(rec),
                    mention_ids=[responder] if responder else [],
                )
                return True, {"channel_id": str(self.gate.channel_id), "thread_id": str(thread.id)}
            if item.get("kind") == "post" and item.get("thread_id"):
                await self.gate.post(client, int(item["thread_id"]), clip(str(payload["text"])))
                return True, {}
        except OutputRefused:
            logger.warning("refused to post outbox item %s outside the output channel", item)
        except (discord.DiscordException, KeyError, TypeError, ValueError):
            logger.warning("could not post outbox item %s", item.get("id"), exc_info=True)
        return False, {}

    async def run_outbox(self, client: Any, interval: float = OUTBOX_INTERVAL_S) -> None:
        """Deliver forever: every ``interval`` seconds, or at once when kicked."""

        while True:
            try:
                await self.deliver(client)
            except Exception:  # noqa: BLE001 - the loop must survive one bad pass
                logger.exception("outbox delivery failed")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.kick.wait(), timeout=interval)
            self.kick.clear()


# --------------------------------------------------------------------------- #
# persistent buttons
# --------------------------------------------------------------------------- #
_ui: RequestUI | None = None


async def _no_ui(interaction: discord.Interaction) -> None:
    await interaction.response.send_message(_UNAVAILABLE, ephemeral=True)


class SendButton(discord.ui.DynamicItem[discord.ui.Button], template=SEND_TEMPLATE):
    def __init__(self, session_id: str) -> None:
        super().__init__(_button("依頼する", send_id(session_id), discord.ButtonStyle.primary))
        self.session_id = session_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match: re.Match[str]):  # type: ignore[override]
        return cls(match["session_id"])

    async def callback(self, interaction: discord.Interaction) -> None:
        await (_ui.on_send(interaction, self.session_id) if _ui else _no_ui(interaction))


class RequestButton(discord.ui.DynamicItem[discord.ui.Button], template=ACTION_TEMPLATE):
    def __init__(self, action: str, recommendation_id: int) -> None:
        super().__init__(
            _button(action, action_id(action, recommendation_id), discord.ButtonStyle.secondary)
        )
        self.action = action
        self.recommendation_id = recommendation_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match: re.Match[str]):  # type: ignore[override]
        return cls(match["action"], int(match["rec"]))

    async def callback(self, interaction: discord.Interaction) -> None:
        if _ui is None:
            await _no_ui(interaction)
            return
        await _ui.on_action(interaction, self.recommendation_id, self.action)


class ReferSelect(discord.ui.DynamicItem[discord.ui.UserSelect], template=REFER_TEMPLATE):
    def __init__(self, recommendation_id: int) -> None:
        super().__init__(discord.ui.UserSelect(custom_id=refer_id(recommendation_id)))
        self.recommendation_id = recommendation_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match: re.Match[str]):  # type: ignore[override]
        return cls(int(match["rec"]))

    async def callback(self, interaction: discord.Interaction) -> None:
        if _ui is None:
            await _no_ui(interaction)
            return
        await _ui.on_refer(interaction, self.recommendation_id)


class ReviewButton(discord.ui.DynamicItem[discord.ui.Button], template=REVIEW_TEMPLATE):
    def __init__(self, action: str, article_id: int) -> None:
        super().__init__(
            _button(action, review_id(action, article_id), discord.ButtonStyle.secondary)
        )
        self.action = action
        self.article_id = article_id

    @classmethod
    async def from_custom_id(cls, interaction, item, match: re.Match[str]):  # type: ignore[override]
        return cls(match["action"], int(match["article"]))

    async def callback(self, interaction: discord.Interaction) -> None:
        if _ui is None:
            await _no_ui(interaction)
            return
        await _ui.on_review_action(interaction, self.article_id, self.action)


class ReviewPageButton(discord.ui.DynamicItem[discord.ui.Button], template=REVIEW_PAGE_TEMPLATE):
    def __init__(self, offset: int, shown: list[int]) -> None:
        super().__init__(
            _button("次へ", review_page_id(offset, shown), discord.ButtonStyle.primary)
        )
        self.offset = offset
        self.shown = shown

    @classmethod
    async def from_custom_id(cls, interaction, item, match: re.Match[str]):  # type: ignore[override]
        return cls(int(match["offset"]), [int(i) for i in match["shown"].split(".")])

    async def callback(self, interaction: discord.Interaction) -> None:
        if _ui is None:
            await _no_ui(interaction)
            return
        await _ui.on_review(interaction, self.offset, self.shown)


def install(client: discord.Client, group: app_commands.Group, ui: RequestUI) -> None:
    """Register the persistent buttons and ``/tekijin review`` on the bot."""

    global _ui
    _ui = ui
    client.add_dynamic_items(SendButton, RequestButton, ReferSelect, ReviewButton, ReviewPageButton)

    @group.command(name="review", description="確認を待っている知識の下書きを見る(管理者のみ)")
    async def review(interaction: discord.Interaction) -> None:
        await ui.on_review(interaction)

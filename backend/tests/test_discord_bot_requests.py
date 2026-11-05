"""The bot's side of requests and review (#114, ``discord_bot/requests.py``).

Uses fake interactions and channels, no Discord. Pins: replies to the presser are
ephemeral and ping no one; buttons carry ids the persistent handlers match; the
outbox is posted only through the output gate; only messages in request threads
are forwarded.
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from typing import Any

import pytest

from tekijin.discord_bot import requests as bot_requests
from tekijin.discord_bot.output import OutputGate
from tekijin.discord_bot.request_format import format_request, format_review
from tekijin.discord_bot.requests import (
    ACTION_TEMPLATE,
    REFER_TEMPLATE,
    REVIEW_TEMPLATE,
    SEND_TEMPLATE,
    RequestUI,
    request_view,
    review_view,
    send_view,
)

GUILD = 10
OUTPUT = 20
SESSION = "discord_" + "a" * 32


class _Response:
    def __init__(self) -> None:
        self.deferred: list[dict] = []
        self.sent: list[tuple[str, dict]] = []

    async def defer(self, **kwargs) -> None:
        self.deferred.append(kwargs)

    async def send_message(self, text: str, **kwargs) -> None:
        self.sent.append((text, kwargs))


class _Followup:
    def __init__(self) -> None:
        self.sent: list[tuple[str, dict]] = []

    async def send(self, text: str, **kwargs) -> None:
        self.sent.append((text, kwargs))


class _Message:
    def __init__(self, channel) -> None:
        self.channel = channel
        self.edits: list[dict] = []

    async def edit(self, **kwargs) -> None:
        self.edits.append(kwargs)


def _interaction(user_id: int = 2001, *, message=None, data=None):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name="押した人"),
        guild_id=GUILD,
        response=_Response(),
        followup=_Followup(),
        message=message,
        data=data,
    )


class _Channel:
    def __init__(self, cid: int, *, parent_id: int | None = None) -> None:
        self.id = cid
        self.parent_id = parent_id
        self.guild = SimpleNamespace(id=GUILD)
        self.sent: list[tuple[str, dict]] = []
        self.threads: list[_Channel] = []

    async def send(self, content: str, **kwargs):
        self.sent.append((content, kwargs))

    async def create_thread(self, **kwargs):
        thread = _Channel(700 + len(self.threads), parent_id=self.id)
        self.threads.append(thread)
        return thread


class _Client:
    def __init__(self, *channels) -> None:
        self.channels = {c.id: c for c in channels}

    def get_channel(self, cid):
        return self.channels.get(cid)

    async def fetch_channel(self, cid):
        import discord

        raise discord.NotFound(SimpleNamespace(status=404, reason="nf"), "missing")


class _Calls(list):
    """The backend calls made, plus ``replies``: what each path answers."""

    def __init__(self) -> None:
        super().__init__()
        self.replies: dict[str, dict] = {}


@pytest.fixture
def calls(monkeypatch) -> _Calls:
    made = _Calls()
    replies = made.replies

    async def fake_post(http, *, base_url, secret, path, body):
        made.append((path, body))
        return replies.get(path.split("/")[0] if path.startswith("outbox") else path, {})

    monkeypatch.setattr(bot_requests, "internal_post", fake_post)
    return made


def _ui(channel: int | None = OUTPUT, *, mention: bool = False) -> RequestUI:
    return RequestUI(
        http=None,  # type: ignore[arg-type]
        base_url="http://b",
        secret="s",
        gate=OutputGate(channel, guild_id=GUILD, mention=mention),
    )


def _ids(view) -> list[str]:
    return [item.custom_id for item in view.children]


def test_button_ids_match_their_handlers() -> None:
    assert re.fullmatch(SEND_TEMPLATE, _ids(send_view(SESSION))[0])
    ids = _ids(request_view(42))
    assert ids == ["tk:req:accept:42", "tk:req:decline:42", "tk:req:refer:42"]
    assert all(re.fullmatch(ACTION_TEMPLATE, i) for i in ids)
    assert re.fullmatch(REFER_TEMPLATE, "tk:ref:42")
    review = _ids(review_view([{"id": 7}, {"id": 8}]))
    assert review == ["tk:rev:verify:7", "tk:rev:back:7", "tk:rev:verify:8", "tk:rev:back:8"]
    assert all(re.fullmatch(REVIEW_TEMPLATE, i) for i in review)
    # Ids from the client that are not ours never match.
    assert not re.fullmatch(SEND_TEMPLATE, "tk:send:discord_../../x")
    assert not re.fullmatch(ACTION_TEMPLATE, "tk:req:delete:42")


def test_send_replies_only_to_the_presser(calls) -> None:
    calls.replies["request"] = {"status": "requested", "responder_name": "たろう"}
    ui = _ui()
    interaction = _interaction(1001)
    asyncio.run(ui.on_send(interaction, SESSION))

    assert calls == [
        (
            "request",
            {
                "discord_user_id": "1001",
                "guild_id": str(GUILD),
                "display_name": "押した人",
                "session_id": SESSION,
            },
        )
    ]
    assert interaction.response.deferred == [{"ephemeral": True, "thinking": True}]
    ((text, kwargs),) = interaction.followup.sent
    assert kwargs["ephemeral"] is True and kwargs["allowed_mentions"].users is False
    assert "たろう" in text and f"<#{OUTPUT}>" in text
    assert ui.kick.is_set()


def test_send_without_an_output_channel_calls_nothing(calls) -> None:
    interaction = _interaction()
    asyncio.run(_ui(None).on_send(interaction, SESSION))
    assert calls == []
    assert interaction.followup.sent[0][1]["ephemeral"] is True


def test_accept_takes_the_buttons_off_in_the_output_channel_only(calls) -> None:
    calls.replies["request/action"] = {"status": "accepted"}
    in_thread = _Message(_Channel(700, parent_id=OUTPUT))
    elsewhere = _Message(_Channel(31, parent_id=30))
    ui = _ui()
    for message in (in_thread, elsewhere):
        asyncio.run(ui.on_action(_interaction(message=message), 42, "accept"))
    assert in_thread.edits == [{"view": None}]
    assert elsewhere.edits == []
    assert calls[0][1]["recommendation_id"] == 42 and calls[0][1]["action"] == "accept"


def test_forbidden_keeps_the_buttons(calls) -> None:
    calls.replies["request/action"] = {"status": "forbidden"}
    message = _Message(_Channel(700, parent_id=OUTPUT))
    interaction = _interaction(message=message)
    asyncio.run(_ui().on_action(interaction, 42, "decline"))
    assert message.edits == []
    assert "本人だけ" in interaction.followup.sent[0][0]


def test_refer_asks_for_the_person_then_sends_their_id(calls) -> None:
    calls.replies["request/action"] = {"status": "referred", "next_name": "はなこ"}
    ui = _ui()
    first = _interaction()
    asyncio.run(ui.on_action(first, 42, "refer"))
    assert calls == []
    ((_, kwargs),) = first.response.sent
    assert kwargs["ephemeral"] is True and _ids(kwargs["view"]) == ["tk:ref:42"]

    picked = _interaction(data={"values": ["3003"]})
    asyncio.run(ui.on_refer(picked, 42))
    assert calls[0][1]["referred_discord_user_id"] == "3003"
    assert "はなこ" in picked.followup.sent[0][0]

    bogus = _interaction(data={"values": ["<@3003>"]})
    asyncio.run(ui.on_refer(bogus, 42))
    assert "referred_discord_user_id" not in calls[1][1]


def test_review_shows_buttons_only_with_drafts(calls) -> None:
    calls.replies["review"] = {
        "status": "ok",
        "total": 1,
        "articles": [{"id": 7, "kind": "case", "issue": "VPN", "resolution": "再起動"}],
    }
    interaction = _interaction(9001)
    asyncio.run(_ui().on_review(interaction))
    ((text, kwargs),) = interaction.followup.sent
    assert kwargs["ephemeral"] is True and "#7" in text and "VPN" in text
    assert _ids(kwargs["view"]) == ["tk:rev:verify:7", "tk:rev:back:7"]

    calls.replies["review"] = {"status": "forbidden"}
    denied = _interaction(1)
    asyncio.run(_ui().on_review(denied))
    assert "view" not in denied.followup.sent[0][1]
    assert "管理者" in denied.followup.sent[0][0]


def test_review_action_maps_back_to_send_back(calls) -> None:
    calls.replies["review/action"] = {"status": "sent_back"}
    interaction = _interaction(9001)
    asyncio.run(_ui().on_review_action(interaction, 7, "back"))
    assert calls[0][1]["action"] == "send_back" and calls[0][1]["article_id"] == 7
    assert "差し戻しました" in interaction.followup.sent[0][0]


def _outbox(monkeypatch, items: list[dict[str, Any]]) -> None:
    async def fake_get(http, *, base_url, secret, limit=20):
        return items

    monkeypatch.setattr(bot_requests, "get_outbox", fake_get)


REQUEST_ITEM = {
    "id": 1,
    "kind": "request",
    "thread_id": None,
    "payload": {
        "recommendation_id": 42,
        "question": "VPNが切れる",
        "draft": "見てもらえますか",
        "asker_name": "質問者",
        "responder_name": "回答者",
        "responder_discord_id": "2001",
    },
}


def test_outbox_opens_the_request_thread_in_the_output_channel(monkeypatch, calls) -> None:
    output = _Channel(OUTPUT)
    _outbox(monkeypatch, [REQUEST_ITEM])
    assert asyncio.run(_ui().deliver(_Client(output))) == 1

    (thread,) = output.threads
    ((content, kwargs),) = thread.sent
    assert "回答者さん" in content and "<@2001>" not in content
    assert kwargs["allowed_mentions"].users is False
    assert _ids(kwargs["view"]) == ["tk:req:accept:42", "tk:req:decline:42", "tk:req:refer:42"]
    assert calls == [("outbox/1", {"ok": True, "channel_id": str(OUTPUT), "thread_id": "700"})]


def test_outbox_mentions_the_member_only_when_on(monkeypatch, calls) -> None:
    output = _Channel(OUTPUT)
    _outbox(monkeypatch, [REQUEST_ITEM])
    asyncio.run(_ui(mention=True).deliver(_Client(output)))
    ((content, kwargs),) = output.threads[0].sent
    assert "<@2001>" in content
    assert [u.id for u in kwargs["allowed_mentions"].users] == [2001]
    assert kwargs["allowed_mentions"].everyone is False


def test_outbox_never_posts_outside_the_output_channel(monkeypatch, calls) -> None:
    other_thread = _Channel(31, parent_id=30)
    _outbox(
        monkeypatch,
        [{"id": 2, "kind": "post", "thread_id": "31", "payload": {"text": "x"}}],
    )
    assert asyncio.run(_ui().deliver(_Client(other_thread))) == 0
    assert other_thread.sent == []
    assert calls == [("outbox/2", {"ok": False})]


def test_outbox_posts_into_a_request_thread(monkeypatch, calls) -> None:
    thread = _Channel(700, parent_id=OUTPUT)
    _outbox(
        monkeypatch,
        [{"id": 3, "kind": "post", "thread_id": "700", "payload": {"text": "@everyone 承諾"}}],
    )
    asyncio.run(_ui().deliver(_Client(thread)))
    ((content, kwargs),) = thread.sent
    assert content == "@everyone 承諾"
    assert kwargs["allowed_mentions"].everyone is False


def test_outbox_does_nothing_without_an_output_channel(monkeypatch, calls) -> None:
    _outbox(monkeypatch, [REQUEST_ITEM])
    assert asyncio.run(_ui(None).deliver(_Client(_Channel(OUTPUT)))) == 0
    assert calls == []


def _thread_message(channel, *, bot: bool = False, content: str = "答えです"):
    return SimpleNamespace(
        id=55,
        guild=SimpleNamespace(id=GUILD),
        author=SimpleNamespace(id=2001, bot=bot, display_name="回答者"),
        channel=channel,
        content=content,
    )


def test_only_request_thread_messages_are_forwarded(calls) -> None:
    ui = _ui()
    asyncio.run(ui.on_thread_message(_thread_message(_Channel(700, parent_id=OUTPUT))))
    asyncio.run(ui.on_thread_message(_thread_message(_Channel(OUTPUT))))
    asyncio.run(ui.on_thread_message(_thread_message(_Channel(31, parent_id=30))))
    asyncio.run(ui.on_thread_message(_thread_message(_Channel(700, parent_id=OUTPUT), bot=True)))
    asyncio.run(
        ui.on_thread_message(_thread_message(_Channel(700, parent_id=OUTPUT), content="  "))
    )
    assert [(p, b["thread_id"], b["body"]) for p, b in calls] == [
        ("thread-message", "700", "答えです")
    ]


def test_request_text_names_without_mentioning() -> None:
    gate = OutputGate(OUTPUT, guild_id=GUILD, mention=False)
    text = format_request({**REQUEST_ITEM["payload"], "question": "あ" * 5000}, gate)
    assert len(text) <= 2000 and "回答者さん" in text


def test_review_text_handles_every_status() -> None:
    assert "ありません" in format_review({"status": "ok", "articles": []})
    assert "使えません" in format_review({"status": "unlinked"})
    assert "エラー" in format_review({"kind": "error"})


def test_persistent_items_rebuild_from_their_ids() -> None:
    """After a restart a press is matched by template and rebuilt from the id."""

    cases = [
        (bot_requests.SendButton, SEND_TEMPLATE, send_view(SESSION).children[0].custom_id),
        (bot_requests.RequestButton, ACTION_TEMPLATE, "tk:req:decline:42"),
        (bot_requests.ReferSelect, REFER_TEMPLATE, "tk:ref:42"),
        (bot_requests.ReviewButton, REVIEW_TEMPLATE, "tk:rev:back:7"),
        (bot_requests.ReviewPageButton, bot_requests.REVIEW_PAGE_TEMPLATE, "tk:revn:8:5.6.7.8"),
    ]
    for cls, template, custom_id in cases:
        item = asyncio.run(cls.from_custom_id(None, None, re.fullmatch(template, custom_id)))
        assert item.item.custom_id == custom_id


def test_review_text_shows_every_draft_it_has_buttons_for() -> None:
    articles = [
        {"id": 100 + i, "kind": "procedure", "issue": "問" * 200, "resolution": "答" * 300}
        for i in range(4)
    ]
    text = format_review({"status": "ok", "total": 40, "offset": 8, "articles": articles})
    assert len(text) <= 2000
    assert all(f"#{a['id']}" in text for a in articles)
    assert "9〜12件目" in text and "「次へ」" in text


def test_review_offers_the_next_page(calls) -> None:
    calls.replies["review"] = {
        "status": "ok",
        "total": 9,
        "offset": 4,
        "articles": [{"id": i, "kind": "fact", "issue": "x"} for i in range(4)],
    }
    interaction = _interaction(9001)
    asyncio.run(_ui().on_review(interaction, 4))
    assert calls[0][1]["offset"] == 4 and calls[0][1]["shown"] == []
    ids = _ids(interaction.followup.sent[0][1]["view"])
    assert ids[-1] == "tk:revn:8:0.1.2.3" and len(ids) == 9
    assert re.fullmatch(bot_requests.REVIEW_PAGE_TEMPLATE, ids[-1])


def test_an_edit_in_a_request_thread_is_forwarded_as_an_edit(calls) -> None:
    thread = _Channel(700, parent_id=OUTPUT)
    message = _thread_message(thread, content="直した答え")

    async def fetch_message(mid):
        return message

    thread.fetch_message = fetch_message
    asyncio.run(_ui().on_thread_edit(_Client(thread), 700, 55))
    assert calls[0][1]["edited"] is True and calls[0][1]["body"] == "直した答え"
    other = _Channel(31, parent_id=30)
    other.fetch_message = fetch_message
    asyncio.run(_ui().on_thread_edit(_Client(other), 31, 55))
    assert len(calls) == 1


def test_a_thread_message_the_backend_missed_is_sent_again(monkeypatch, calls) -> None:
    answers = iter([{"kind": "error"}, {"status": "answered"}])

    async def flaky(http, *, base_url, secret, path, body):
        calls.append((path, body))
        return next(answers)

    monkeypatch.setattr(bot_requests, "internal_post", flaky)
    ui = _ui()
    asyncio.run(ui.on_thread_message(_thread_message(_Channel(700, parent_id=OUTPUT))))
    asyncio.run(ui.retry_messages())
    assert [b["message_id"] for _, b in calls] == ["55", "55"]
    asyncio.run(ui.retry_messages())  # nothing left
    assert len(calls) == 2


class _History(_Channel):
    def __init__(self, cid: int, *, parent_id: int | None, messages: list) -> None:
        super().__init__(cid, parent_id=parent_id)
        self.messages = messages

    def history(self, **kwargs):
        async def gen():
            for m in self.messages:
                yield m

        return gen()


def test_replay_reads_only_request_threads(calls) -> None:
    request_thread = _History(700, parent_id=OUTPUT, messages=[])
    request_thread.messages = [_thread_message(request_thread)]
    other = _History(31, parent_id=30, messages=[])
    other.messages = [_thread_message(other)]
    asyncio.run(_ui().replay([request_thread, other]))
    assert [b["thread_id"] for _, b in calls] == ["700"]


def test_a_lost_report_is_sent_again_without_posting_again(monkeypatch, calls) -> None:
    output = _Channel(OUTPUT)
    _outbox(monkeypatch, [REQUEST_ITEM])
    replies = iter([{"kind": "error"}, {"kind": "error"}, {"status": "ok"}])

    async def flaky(http, *, base_url, secret, path, body):
        calls.append((path, body))
        return next(replies)

    monkeypatch.setattr(bot_requests, "internal_post", flaky)
    ui = _ui()
    asyncio.run(ui.deliver(_Client(output)))
    # The backend still lists the item; the bot reports again and posts nothing new.
    asyncio.run(ui.deliver(_Client(output)))
    assert len(output.threads) == 1
    assert [p for p, _ in calls] == ["outbox/1", "outbox/1"]
    assert calls[1][1] == {"ok": True, "channel_id": str(OUTPUT), "thread_id": "700"}
    # Once the report is taken, it is not kept any longer.
    _outbox(monkeypatch, [])
    asyncio.run(ui.deliver(_Client(output)))
    assert len(calls) == 3 and ui._acks == {}


def test_deletions_are_forwarded_and_retried(monkeypatch, calls) -> None:
    replies = iter([{"kind": "error"}, {"status": "ok"}])

    async def flaky(http, *, base_url, secret, path, body):
        calls.append((path, body))
        return next(replies)

    monkeypatch.setattr(bot_requests, "internal_post", flaky)
    ui = _ui()
    asyncio.run(ui.on_thread_delete([55, 56]))
    asyncio.run(ui.retry_deletes())
    assert calls == [
        ("thread-message/delete", {"guild_id": str(GUILD), "message_ids": ["55", "56"]}),
        ("thread-message/delete", {"guild_id": str(GUILD), "message_ids": ["55", "56"]}),
    ]
    asyncio.run(ui.retry_deletes())
    assert len(calls) == 2


def test_a_thread_known_late_is_read_again(monkeypatch, calls) -> None:
    """Messages written before the backend learned the thread are sent again."""

    output = _History(OUTPUT, parent_id=None, messages=[])
    _outbox(monkeypatch, [REQUEST_ITEM])
    replies = iter([{"kind": "error"}, {"status": "ok"}, {"status": "answered"}])

    async def flaky(http, *, base_url, secret, path, body):
        calls.append((path, body))
        return next(replies)

    monkeypatch.setattr(bot_requests, "internal_post", flaky)
    ui = _ui()

    async def create_thread(**kwargs):
        thread = _History(700, parent_id=OUTPUT, messages=[])
        thread.messages = [_thread_message(thread)]
        output.threads.append(thread)
        return thread

    output.create_thread = create_thread
    client = _Client(output)
    asyncio.run(ui.deliver(client))
    client.channels[700] = output.threads[0]
    _outbox(monkeypatch, [])
    asyncio.run(ui.deliver(client))
    assert [p for p, _ in calls] == ["outbox/1", "outbox/1", "thread-message"]

"""Unit tests for Discord ``/ask`` (D1, #60): event summary and reply formatting.

``summarize_events`` turns the same SSE event list the web client receives into one
outcome, and ``format_reply`` turns that outcome into a Discord message. Both are pure,
so every route the graph can end on is pinned here without a DB, a model or Discord.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from tekijin.channels.ask_summary import summarize_events
from tekijin.discord_bot.backend import (
    SECRET_HEADER,
    add_backend,
    ask_backend,
    discord_user,
    member_backend,
)
from tekijin.discord_bot.format import (
    DISCORD_MESSAGE_LIMIT,
    format_add_reply,
    format_member_reply,
    format_reply,
)

WEB = "https://tekijin.example/"
REC = {
    "person_id": "E001",
    "name": "社員1",
    "dept": "営業部",
    "score": 0.9,
    "confidence": "高",
    "reasons": [{"type": "answers", "detail": "類似の質問に過去3件回答"}],
}


def test_self_answer_becomes_answered_with_citations() -> None:
    out = summarize_events(
        [
            ("understood", {"topics": ["ネットワーク・VPN"]}),
            (
                "message",
                {
                    "status": "self_answered",
                    "message": "保守時間内に更新します。",
                    "citations": [{"source_id": "doc_001", "kind": "document"}],
                },
            ),
        ]
    )
    assert out.kind == "answered"
    assert out.text == "保守時間内に更新します。"
    assert [c.source_id for c in out.citations] == ["doc_001"]


def test_document_route_cites_its_document() -> None:
    data = {"status": "document", "message": "手順書を参照してください。", "doc_id": "doc_9"}
    out = summarize_events([("message", data)])
    assert out.kind == "answered"
    assert [(c.source_id, c.kind) for c in out.citations] == [("doc_9", "document")]


def test_person_route_becomes_handoff_with_reference() -> None:
    out = summarize_events(
        [
            ("route", {"route": "person", "reason": "", "confidence": 0.5}),
            ("reference", {"answer": "過去にはこう答えています。", "citations": []}),
            ("recommend", {"recommendations": [REC]}),
            ("draft", {"draft": "社員1さん、ご相談です。"}),
        ]
    )
    assert out.kind == "handoff"
    assert [r.name for r in out.recommendations] == ["社員1"]
    assert out.text == "過去にはこう答えています。"


def test_followup_and_dead_ends() -> None:
    followup = summarize_events([("followup", {"question": "どの拠点ですか?", "missing": []})])
    assert (followup.kind, followup.text) == ("followup", "どの拠点ですか?")

    none = summarize_events([("message", {"status": "no_candidate", "message": "見つかりません"})])
    assert (none.kind, none.text) == ("no_answer", "見つかりません")


def test_error_or_empty_stream_is_an_error() -> None:
    assert summarize_events([("error", {"error": "x"})]).kind == "error"
    assert summarize_events([]).kind == "error"


def test_format_answered_lists_sources() -> None:
    out = summarize_events(
        [
            (
                "message",
                {
                    "status": "self_answered",
                    "message": "更新します。",
                    "citations": [{"source_id": "doc_001", "kind": "document"}],
                },
            )
        ]
    )
    text = format_reply(out.model_dump(), web_url=WEB)
    assert "更新します。" in text
    assert "doc_001" in text and "文書" in text


def test_format_handoff_names_people_and_points_to_web() -> None:
    out = summarize_events([("recommend", {"recommendations": [REC]}), ("draft", {"draft": "d"})])
    text = format_reply(out.model_dump(), web_url=WEB)
    assert "社員1" in text and "類似の質問に過去3件回答" in text
    assert "営業部" not in text  # #114: no company-era department
    assert WEB in text


def test_format_handoff_offers_the_button_when_it_can_be_sent() -> None:
    """#114: with an output channel, the reply points to 「依頼する」, not the web."""

    out = summarize_events([("recommend", {"recommendations": [REC]})])
    text = format_reply(out.model_dump(), web_url=WEB, send_channel_id=5555)
    assert "「依頼する」" in text and "<#5555>" in text and "社員1" in text
    assert WEB not in text


def test_format_handoff_lists_a_members_evidence_one_per_line() -> None:
    """#107: a member is recommended from evidence, each reason naming its source."""

    member = {
        **REC,
        "dept": None,
        "name": "岩田 颯太",
        "reasons": [
            {"type": "self_intro", "detail": "自己紹介: 3年。望遠鏡と赤道儀の整備が趣味です。"},
            {"type": "document", "detail": "『機材の使い方と管理マニュアル』を書いた"},
        ],
    }
    out = summarize_events([("recommend", {"recommendations": [member]})])
    text = format_reply(out.model_dump(), web_url=WEB)

    assert "1. 岩田 颯太\n   - 自己紹介: 3年。望遠鏡と赤道儀の整備が趣味です。\n" in text
    assert "   - 『機材の使い方と管理マニュアル』を書いた" in text


def test_sources_without_the_answer_point_to_the_person_and_the_web() -> None:
    """#107: a member's question whose sources held no answer names a person."""

    text = "資料の中に、この質問の答えは見つかりませんでした。岩田 颯太さんに相談できます。"
    out = summarize_events(
        [
            (
                "message",
                {
                    "status": "document",
                    "message": text,
                    "fallback_responder": {"name": "岩田 颯太"},
                },
            )
        ]
    )
    reply = format_reply(out.model_dump(), web_url=WEB)

    assert reply.startswith(text)
    assert reply.endswith(f"依頼を送るには、Webで同じ質問をしてください: {WEB}")


def test_format_followup_and_unlinked_point_to_web() -> None:
    followup = summarize_events([("followup", {"question": "どの拠点ですか?"})])
    assert WEB in format_reply(followup.model_dump(), web_url=WEB)
    unlinked = format_reply({"kind": "unlinked", "text": ""}, web_url=WEB)
    assert "使えません" in unlinked


def test_format_never_exceeds_discord_limit() -> None:
    out = summarize_events([("message", {"status": "self_answered", "message": "あ" * 5000})])
    text = format_reply(out.model_dump(), web_url=WEB)
    assert len(text) <= DISCORD_MESSAGE_LIMIT


def test_handoff_keeps_the_reference_citations() -> None:
    out = summarize_events(
        [
            (
                "reference",
                {
                    "answer": "手順書を見てください。",
                    "citations": [{"source_id": "doc_7", "kind": "document"}],
                },
            ),
            ("recommend", {"recommendations": [REC]}),
        ]
    )
    assert out.kind == "handoff"
    assert [c.source_id for c in out.citations] == ["doc_7"]
    text = format_reply(out.model_dump(), web_url=WEB)
    assert "doc_7" in text and "社員1" in text


def test_document_route_names_the_document_and_points_hand_off_to_web() -> None:
    out = summarize_events(
        [
            (
                "message",
                {
                    "status": "document",
                    "message": "社内文書に該当がありそうです(文書ID: doc_027)。",
                    "doc_id": "doc_027",
                    "fallback_responder": REC,
                },
            )
        ]
    )
    assert (out.doc_id, out.fallback_name) == ("doc_027", "社員1")
    reply = out.model_copy(
        update={"document_title": "経理・決算FAQ", "document_excerpt": "月次決算は前倒しで。"}
    )
    text = format_reply(reply.model_dump(), web_url=WEB)
    assert "経理・決算FAQ" in text and "月次決算は前倒しで。" in text and "doc_027" in text
    assert "社員1" in text and WEB in text
    assert "取り次げます" not in text

    # Without a preview (lookup failed) the reply still names the document.
    bare = format_reply(out.model_dump(), web_url=WEB)
    assert "doc_027" in bare and WEB in bare


def test_self_answer_has_no_document_fields() -> None:
    out = summarize_events(
        [("message", {"status": "self_answered", "message": "ok", "doc_id": "doc_1"})]
    )
    assert out.doc_id is None and out.fallback_name is None


def test_format_keeps_sources_and_web_link_when_clipping() -> None:
    long_answer = summarize_events(
        [
            (
                "message",
                {
                    "status": "document",
                    "message": "あ" * 5000,
                    "doc_id": "doc_009",
                },
            )
        ]
    )
    text = format_reply(long_answer.model_dump(), web_url=WEB)
    assert len(text) <= DISCORD_MESSAGE_LIMIT
    assert "doc_009" in text

    long_ref = summarize_events(
        [("reference", {"answer": "い" * 5000}), ("recommend", {"recommendations": [REC]})]
    )
    text = format_reply(long_ref.model_dump(), web_url=WEB)
    assert len(text) <= DISCORD_MESSAGE_LIMIT
    assert "社員1" in text and WEB in text

    wordy = {**REC, "reasons": [{"type": "answers", "detail": "え" * 3000}]}
    long_people = summarize_events([("recommend", {"recommendations": [wordy, wordy]})])
    text = format_reply(long_people.model_dump(), web_url=WEB)
    assert len(text) <= DISCORD_MESSAGE_LIMIT
    assert text.endswith(WEB)

    long_followup = summarize_events([("followup", {"question": "う" * 5000})])
    text = format_reply(long_followup.model_dump(), web_url=WEB)
    assert len(text) <= DISCORD_MESSAGE_LIMIT
    assert WEB in text


USER = discord_user(1001, 42, "たろう")


def test_discord_user_handles_dms() -> None:
    assert discord_user(7, None, "x")["guild_id"] is None


def test_member_backend_posts_the_action() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "opted_out"})

    async def run() -> dict:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await member_backend(
                http, base_url="http://api", secret="k", user=USER, action="optout"
            )

    assert asyncio.run(run()) == {"status": "opted_out"}
    assert seen["url"] == "http://api/internal/discord/member"
    assert seen["body"]["action"] == "optout"


def test_add_backend_posts_the_note() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "added", "chunks": 1})

    async def run() -> dict:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await add_backend(
                http,
                base_url="http://api",
                secret="k",
                user=USER,
                text="本文",
                title=None,
                url="https://ex.com/a",
            )

    assert asyncio.run(run()) == {"status": "added", "chunks": 1}
    assert seen["url"] == "http://api/internal/discord/add"
    assert seen["body"]["text"] == "本文" and seen["body"]["url"] == "https://ex.com/a"
    assert "title" not in seen["body"]


def test_format_add_reply() -> None:
    assert "追加しました" in format_add_reply({"status": "added", "chunks": 2})
    assert "更新しました" in format_add_reply({"status": "updated", "chunks": 1})
    assert "変わっていない" in format_add_reply({"status": "unchanged", "chunks": 1})
    assert "/tekijin join" in format_add_reply({"status": "not_joined"})
    assert "本人だけ" in format_add_reply({"status": "not_owner"})
    assert "使えません" in format_add_reply({"status": "unlinked"})
    assert "エラー" in format_add_reply({"kind": "error"})


def test_format_member_reply() -> None:
    joined = format_member_reply({"status": "joined", "login_url": "https://w/login#t"})
    assert "https://w/login#t" in joined and "10分" in joined
    assert "optin" in format_member_reply({"status": "opted_out"})
    assert "戻しました" in format_member_reply({"status": "opted_in"})
    assert "使えません" in format_member_reply({"status": "unlinked"})
    assert "エラー" in format_member_reply({"kind": "error"})
    assert "エラー" in format_member_reply({"status": "joined"})


def test_format_member_reply_about_intros() -> None:
    """#106: join with an introduction, ``/tekijin intro``, and what optout does."""

    url = "https://w/login#t"
    saved = format_member_reply({"status": "joined", "login_url": url, "intro_saved": True})
    assert "自己紹介を登録しました" in saved and url in saved
    refused = format_member_reply({"status": "joined", "login_url": url, "intro_saved": False})
    assert "登録しませんでした" in refused and url in refused
    assert "自己紹介" not in format_member_reply({"status": "joined", "login_url": url})
    assert "置き換わります" in format_member_reply({"status": "intro_saved"})
    assert "何も変えていません" in format_member_reply({"status": "intro_unchanged"})
    assert "optin" in format_member_reply({"status": "intro_refused"})
    assert "join" in format_member_reply({"status": "not_joined"})
    assert "自己紹介を消しました" in format_member_reply({"status": "opted_out"})
    assert "/tekijin intro" in format_member_reply({"status": "opted_in"})


def test_member_backend_sends_the_intro_only_when_given() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "intro_saved"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            common = {"base_url": "http://api", "secret": "k", "user": USER}
            await member_backend(http, action="intro", intro="会計です", **common)
            await member_backend(http, action="join", **common)

    asyncio.run(run())
    assert bodies[0]["action"] == "intro" and bodies[0]["intro"] == "会計です"
    assert "intro" not in bodies[1]


def _ask(handler) -> dict:
    async def run() -> dict:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as http:
            return await ask_backend(
                http, base_url="http://api/", secret="k", user=USER, question="q"
            )

    return asyncio.run(run())


def test_ask_backend_sends_secret_and_user() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["secret"] = request.headers.get(SECRET_HEADER)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"kind": "answered", "text": "ok"})

    assert _ask(handler) == {"kind": "answered", "text": "ok"}
    assert seen == {
        "url": "http://api/internal/discord/ask",
        "secret": "k",
        "body": {
            "discord_user_id": "1001",
            "guild_id": "42",
            "display_name": "たろう",
            "question": "q",
        },
    }


def test_ask_backend_turns_failures_into_replies() -> None:
    assert _ask(lambda r: httpx.Response(503))["kind"] == "no_answer"
    assert _ask(lambda r: httpx.Response(500, text="boom")) == {"kind": "error"}

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    assert _ask(unreachable) == {"kind": "error"}
    assert _ask(lambda r: httpx.Response(200, text="<html>proxy</html>")) == {"kind": "error"}
    assert _ask(lambda r: httpx.Response(200, json=["not", "an", "object"])) == {"kind": "error"}


def test_ingested_source_is_shown_by_title_and_link() -> None:
    reply = format_reply(
        {
            "kind": "answered",
            "text": "守衛室で借ります。",
            "citations": [
                {"source_id": "src_1", "kind": "source", "title": "部室", "url": "https://e.jp/1"},
                {"source_id": "src_2", "kind": "source", "title": "合宿"},
            ],
        },
        web_url="https://e.jp",
    )
    assert "- 部室: <https://e.jp/1>" in reply
    assert "- 合宿\n" in reply + "\n"
    assert "src_1" not in reply

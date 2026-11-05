"""Texts for requests sent from Discord and for ``/tekijin review`` (#114)."""

from __future__ import annotations

from typing import Any

from tekijin.discord_bot.format import DISCORD_MESSAGE_LIMIT
from tekijin.discord_bot.output import OutputGate

_QUESTION_MAX = 500
_DRAFT_MAX = 1000
# Five drafts of these sizes, with the header and the footer, stay within one
# message, so every draft that gets buttons is also shown.
_REVIEW_ISSUE_MAX = 120
_REVIEW_RESOLUTION_MAX = 200

_ERROR = "処理中にエラーが起きました。しばらくしてから、もう一度お試しください。"
_UNLINKED = "このサーバーでは、TEKIJINを使えません。"
_DISABLED = "いまはDiscordから依頼を送れません。"

_SEND_REPLIES = {
    "already": "この依頼はもう送ってあります。",
    "forbidden": "依頼を送れるのは、質問した本人だけです。",
    "gone": "この依頼はもう送れません。もう一度 `/ask` で質問してください。",
    "disabled": _DISABLED,
    "unlinked": _UNLINKED,
}

_ACTION_REPLIES = {
    "accepted": "依頼を受けました。回答はこのスレッドに書いてください。",
    "forbidden": "このボタンを押せるのは、依頼された本人だけです。",
    "stale": "この依頼は、もう別の人に移っています。",
    "gone": "この依頼は、もう受け付けていません。",
    "refer_unknown": (
        "その人はまだTEKIJINを使ったことがないため、依頼を回せません。"
        "その人に一度 `/tekijin join` を打ってもらってください。"
    ),
    "refer_invalid": "その人には依頼を回せません。質問した人や、すでに辞退した人は選べません。",
    "disabled": _DISABLED,
    "unlinked": _UNLINKED,
}

_REVIEW_ACTION_REPLIES = {
    "verified": "確認済みにしました。この知識は、これから質問への回答に使われます。",
    "sent_back": "差し戻しました。下書きは確認待ちに戻り、回答には使われません。",
    "invalid": (
        "回答が書かれていない下書きは確認済みにできません。Webの知識の画面で書き足してください。"
    ),
    "gone": "この下書きは見つかりませんでした。",
    "forbidden": "知識の確認は、管理者だけが行えます。",
    "unlinked": _UNLINKED,
}

_KIND_LABELS = {"case": "事例", "procedure": "手順", "fact": "事実", "decision": "決定"}


def clip(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def request_title(recommendation_id: int) -> str:
    return f"TEKIJINの依頼 #{recommendation_id}"


def format_request(payload: dict[str, Any], gate: OutputGate) -> str:
    """The first message of a request thread."""

    responder = gate.label(payload.get("responder_name") or "", payload.get("responder_discord_id"))
    asker = gate.label(payload.get("asker_name") or "", None)
    question = clip(" ".join((payload.get("question") or "").split()), _QUESTION_MAX)
    draft = clip((payload.get("draft") or "").strip(), _DRAFT_MAX)
    parts = [
        f"{responder}さんへの相談の依頼です(質問した人: {asker}さん)。",
        f"質問\n> {question}" if question else "",
        f"依頼の文\n{draft}" if draft else "",
        "受けるか辞退するかを、下のボタンで選んでください。押せるのは依頼された本人だけです。"
        "受けたあとにこのスレッドに書いた内容は、回答として記録され、知識の下書きになります。",
    ]
    return clip("\n\n".join(p for p in parts if p))


def format_send_reply(result: dict[str, Any], *, channel_id: int | None) -> str:
    status = result.get("status")
    if status == "requested":
        name = result.get("responder_name") or "依頼先"
        where = f"<#{channel_id}>" if channel_id else "依頼用のチャンネル"
        return (
            f"{name}さんへの依頼を、{where} のスレッドに投稿します。"
            "返事や回答はそのスレッドに届きます。"
        )
    return _SEND_REPLIES.get(str(status), _ERROR)


def format_action_reply(result: dict[str, Any]) -> str:
    status = result.get("status")
    if status in ("declined", "referred"):
        head = "辞退しました。" if status == "declined" else "より適任の人に回しました。"
        name = result.get("next_name")
        tail = (
            f"次の候補の{name}さんに依頼しました。"
            if name
            else "次に依頼できる人はいませんでした。"
        )
        return head + tail
    return _ACTION_REPLIES.get(str(status), _ERROR)


def format_review_action_reply(result: dict[str, Any]) -> str:
    return _REVIEW_ACTION_REPLIES.get(str(result.get("status")), _ERROR)


def format_review(result: dict[str, Any]) -> str:
    """The ephemeral list ``/tekijin review`` shows."""

    status = result.get("status")
    if status == "forbidden":
        return _REVIEW_ACTION_REPLIES["forbidden"]
    if status == "unlinked":
        return _UNLINKED
    if status != "ok":
        return _ERROR
    articles = result.get("articles") or []
    if not articles:
        return "確認を待っている下書きはありません。"
    total = int(result.get("total") or len(articles))
    lines = [f"確認を待っている下書きが{total}件あります。よく求められているものから並べています。"]
    for a in articles:
        kind = _KIND_LABELS.get(a.get("kind", ""), a.get("kind", ""))
        issue = clip(" ".join((a.get("issue") or "").split()), _REVIEW_ISSUE_MAX)
        resolution = " ".join((a.get("resolution") or "(回答がまだ書かれていません)").split())
        lines.append(f"#{a['id']} [{kind}] {issue}\n> {clip(resolution, _REVIEW_RESOLUTION_MAX)}")
    if total > len(articles):
        lines.append("残りは、ここにある分を処理したあとに `/tekijin review` で表示されます。")
    return clip("\n\n".join(lines))

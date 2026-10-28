"""Turn the backend's answers into Discord messages (#60, #66, #70)."""

from __future__ import annotations

from typing import Any

# Discord rejects a message whose content is longer than this.
DISCORD_MESSAGE_LIMIT = 2000

# Same labels the web chat uses for SourceCitation.kind (frontend SourceCitations).
_SOURCE_LABELS = {
    "qa": "過去の回答",
    "document": "文書",
    "daily": "日報",
    "knowledge": "知識",
}

_FIXED_REPLIES = {
    "no_answer": "答えられる情報が見つかりませんでした。",
    "error": "処理中にエラーが起きました。しばらくしてから、もう一度お試しください。",
    "unlinked": "このサーバーでは、TEKIJINを使えません。",
}


def format_member_reply(result: dict[str, Any]) -> str:
    """The reply to ``/tekijin join|optout|optin``."""

    status = result.get("status")
    if status == "joined" and result.get("login_url"):
        return (
            "TEKIJINに参加しました。次のURLからWebにログインできます。"
            "10分以内に1回だけ使えます。ほかの人には送らないでください。\n"
            f"{result['login_url']}\n\n"
            "期限が切れたら、もう一度 `/tekijin join` を打ってください。"
        )
    if status == "opted_out":
        return (
            "あなたを、TEKIJINが詳しい人として推薦する対象から外しました。"
            "戻すときは `/tekijin optin` を打ってください。"
        )
    if status == "opted_in":
        return "あなたを、TEKIJINが詳しい人として推薦する対象に戻しました。"
    if status == "unlinked":
        return _FIXED_REPLIES["unlinked"]
    return _FIXED_REPLIES["error"]


def format_add_reply(result: dict[str, Any]) -> str:
    """The reply to ``/tekijin add`` (#70)."""

    status = result.get("status")
    if status in ("added", "updated"):
        verb = "追加しました" if status == "added" else "更新しました"
        return (
            f"TEKIJINの知識に{verb}({result.get('chunks', 0)}個の断片に分けました)。"
            "質問への答えに使われるのは、検索に組み込む次の段階からです。"
        )
    if status == "unchanged":
        return "同じ内容がすでに登録されています。中身は変わっていないので、そのままにしました。"
    if status == "not_joined":
        return "知識を追加するには、先に `/tekijin join` でTEKIJINに参加してください。"
    if status == "unlinked":
        return _FIXED_REPLIES["unlinked"]
    return _FIXED_REPLIES["error"]


def format_reply(outcome: dict[str, Any], *, web_url: str) -> str:
    """Compose the reply, clipping only the free text so the trailing parts (the
    sources of an answer, the Web link of a hand-off) always survive."""

    kind = outcome.get("kind")
    text = (outcome.get("text") or "").strip()
    if kind == "answered" and outcome.get("doc_id"):
        return _document(outcome, web_url)
    if kind == "answered":
        return _join(text, _sources(outcome.get("citations") or []))
    if kind == "handoff":
        return _handoff(
            text, outcome.get("citations") or [], outcome.get("recommendations") or [], web_url
        )
    if kind == "followup":
        return _join(
            f"もう少し詳しく教えてください。\n> {text}",
            f"質問を詳しくして `/ask` し直すか、Webで質問してください: {web_url}",
        )
    if kind == "no_answer" and text:
        return _clip(text)
    return _FIXED_REPLIES.get(str(kind), _FIXED_REPLIES["error"])


def _document(outcome: dict[str, Any], web_url: str) -> str:
    """The document route. The node's own text names only the document id and offers
    a hand-off the chat cannot perform, so the reply is composed here instead."""

    title = outcome.get("document_title") or outcome.get("doc_id")
    head = f"社内文書「{title}」が参考になりそうです。"
    excerpt = outcome.get("document_excerpt")
    if excerpt:
        head += f"\n> {excerpt}"
    tail = _sources(outcome.get("citations") or [])
    name = outcome.get("fallback_name")
    if name:
        tail += (
            f"\n\n解決しない場合は、{name}さんに相談できます。"
            f"依頼を送るには、Webで同じ質問をしてください: {web_url}"
        )
    return _join(head, tail)


def _sources(citations: list[dict[str, Any]]) -> str:
    if not citations:
        return ""
    lines = [
        f"- {_SOURCE_LABELS.get(c.get('kind', ''), '出典')} `{c.get('source_id', '')}`"
        for c in citations
    ]
    return "出典\n" + "\n".join(lines)


def _handoff(
    reference: str,
    citations: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    web_url: str,
) -> str:
    people = ["この質問は、次の人に相談するのがよさそうです。"]
    for i, rec in enumerate(recommendations, start=1):
        dept = f"({rec['dept']})" if rec.get("dept") else ""
        reasons = "、".join(r.get("detail", "") for r in rec.get("reasons") or [])
        people.append(f"{i}. {rec.get('name', '')}{dept}" + (f": {reasons}" if reasons else ""))
    sources = _sources(citations) if reference else ""
    middle = "\n\n".join(part for part in (sources, "\n".join(people)) if part)
    head = f"参考になりそうな情報\n> {reference}" if reference else ""
    # The Web link is the only way to send the request, so it is never clipped;
    # the reference is clipped first, then the source/people list.
    link = f"相談の依頼を送るには、Webで同じ質問をしてください: {web_url}"
    return _join(_join(head, middle, DISCORD_MESSAGE_LIMIT - len(link) - 2), link)


def _join(head: str, tail: str, limit: int = DISCORD_MESSAGE_LIMIT) -> str:
    """``head`` and ``tail`` separated by a blank line, within ``limit``.

    ``head`` is clipped first; ``tail`` is clipped only when it alone does not fit.
    """

    if not tail:
        return _clip(head, limit)
    if not head:
        return _clip(tail, limit)
    room = limit - len(tail) - 2
    if room <= 0:
        return _clip(tail, limit)
    return _clip(head, room) + "\n\n" + tail


def _clip(text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"

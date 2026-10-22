"""Turn an ``/internal/discord/ask`` outcome into one Discord message (Discord D1, #60)."""

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
    "unlinked": (
        "このDiscordアカウントは、TEKIJINの社員とまだ連携されていません。"
        "管理者に連携を依頼してください。"
    ),
}


def format_reply(outcome: dict[str, Any], *, web_url: str) -> str:
    kind = outcome.get("kind")
    text = (outcome.get("text") or "").strip()
    if kind == "answered":
        body = _with_sources(text, outcome.get("citations") or [])
    elif kind == "handoff":
        body = _handoff(text, outcome.get("recommendations") or [], web_url)
    elif kind == "followup":
        body = (
            f"もう少し詳しく教えてください。\n> {text}\n\n"
            f"質問を詳しくして `/ask` し直すか、Webで続けてください: {web_url}"
        )
    elif kind == "no_answer" and text:
        body = text
    else:
        body = _FIXED_REPLIES.get(str(kind), _FIXED_REPLIES["error"])
    return _clip(body)


def _with_sources(text: str, citations: list[dict[str, Any]]) -> str:
    if not citations:
        return text
    lines = [
        f"- {_SOURCE_LABELS.get(c.get('kind', ''), '出典')} `{c.get('source_id', '')}`"
        for c in citations
    ]
    return text + "\n\n出典\n" + "\n".join(lines)


def _handoff(reference: str, recommendations: list[dict[str, Any]], web_url: str) -> str:
    parts = []
    if reference:
        parts.append(f"参考になりそうな過去の回答\n> {reference}")
    people = ["この質問は、次の人に相談するのがよさそうです。"]
    for i, rec in enumerate(recommendations, start=1):
        dept = f"({rec['dept']})" if rec.get("dept") else ""
        reasons = "、".join(r.get("detail", "") for r in rec.get("reasons") or [])
        people.append(f"{i}. {rec.get('name', '')}{dept}" + (f": {reasons}" if reasons else ""))
    parts.append("\n".join(people))
    parts.append(f"相談の依頼はWebから送れます: {web_url}")
    return "\n\n".join(parts)


def _clip(text: str) -> str:
    if len(text) <= DISCORD_MESSAGE_LIMIT:
        return text
    return text[: DISCORD_MESSAGE_LIMIT - 1] + "…"

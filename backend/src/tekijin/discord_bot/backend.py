"""The bot's calls into the backend's ``/internal/discord/*`` endpoints (#60, #66, #70)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SECRET_HEADER = "X-Tekijin-Bot-Secret"

_BUSY_TEXT = "現在混雑しています。少し待ってから、もう一度お試しください。"


def discord_user(user_id: int, guild_id: int | None, display_name: str) -> dict[str, Any]:
    """Who typed the command, in the shape every internal endpoint takes."""

    return {
        "discord_user_id": str(user_id),
        "guild_id": str(guild_id) if guild_id is not None else None,
        "display_name": display_name[:255],
    }


async def ask_backend(
    http: httpx.AsyncClient,
    *,
    base_url: str,
    secret: str,
    user: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    """Return the backend's outcome dict, or a ``no_answer``/``error`` stand-in.

    Never raises: the bot has already deferred the interaction and must always send
    a follow-up, so every failure becomes a reply the user can read.
    """

    return await _post(http, base_url, secret, "ask", {**user, "question": question})


async def member_backend(
    http: httpx.AsyncClient,
    *,
    base_url: str,
    secret: str,
    user: dict[str, Any],
    action: str,
) -> dict[str, Any]:
    """``/tekijin join|optout|optin``. Same never-raise contract as :func:`ask_backend`."""

    return await _post(http, base_url, secret, "member", {**user, "action": action})


async def add_backend(
    http: httpx.AsyncClient,
    *,
    base_url: str,
    secret: str,
    user: dict[str, Any],
    text: str,
    title: str | None,
    url: str | None,
) -> dict[str, Any]:
    """``/tekijin add`` (#70). Same never-raise contract as :func:`ask_backend`."""

    body: dict[str, Any] = {**user, "text": text}
    if title:
        body["title"] = title
    if url:
        body["url"] = url
    return await _post(http, base_url, secret, "add", body)


async def post_messages(
    http: httpx.AsyncClient,
    *,
    base_url: str,
    secret: str,
    guild_id: int,
    messages: list[dict[str, Any]],
) -> bool:
    """Forward new or edited messages (#76). False when they were not stored."""

    result = await _post(
        http, base_url, secret, "messages", {"guild_id": str(guild_id), "messages": messages}
    )
    return result.get("status") == "ok"


async def post_deletes(
    http: httpx.AsyncClient, *, base_url: str, secret: str, guild_id: int, message_ids: list[int]
) -> bool:
    """Forward deletions (#76). False when they were not applied."""

    result = await _post(
        http,
        base_url,
        secret,
        "messages/delete",
        {"guild_id": str(guild_id), "message_ids": message_ids},
    )
    return result.get("status") == "ok"


async def get_cursors(
    http: httpx.AsyncClient, *, base_url: str, secret: str, guild_id: int
) -> dict[str, int] | None:
    """Where each channel's stored history ends (#76), or ``None`` when unknown.

    ``None`` (backend away, or the server not ours) means "do not read history
    now": reading everything from the beginning is the expensive mistake.
    """

    try:
        resp = await http.get(
            f"{base_url.rstrip('/')}/internal/discord/cursors",
            params={"guild_id": str(guild_id)},
            headers={SECRET_HEADER: secret},
        )
    except httpx.HTTPError:
        logger.warning("backend unreachable", exc_info=True)
        return None
    if resp.status_code != 200:
        logger.warning("backend answered %s: %s", resp.status_code, resp.text[:200])
        return None
    body = resp.json()
    if not isinstance(body, dict) or body.get("status") != "ok":
        return None
    return {str(k): int(v) for k, v in body.get("cursors", {}).items()}


async def _post(
    http: httpx.AsyncClient, base_url: str, secret: str, path: str, body: dict[str, Any]
) -> dict[str, Any]:
    try:
        resp = await http.post(
            f"{base_url.rstrip('/')}/internal/discord/{path}",
            json=body,
            headers={SECRET_HEADER: secret},
        )
    except httpx.HTTPError:
        logger.warning("backend unreachable", exc_info=True)
        return {"kind": "error"}
    if resp.status_code == 503:
        return {"kind": "no_answer", "text": _BUSY_TEXT}
    if resp.status_code != 200:
        logger.warning("backend answered %s: %s", resp.status_code, resp.text[:200])
        return {"kind": "error"}
    try:
        outcome = resp.json()
    except ValueError:
        logger.warning("backend answered 200 with a non-JSON body: %s", resp.text[:200])
        return {"kind": "error"}
    if not isinstance(outcome, dict):
        logger.warning("backend answered 200 with a non-object body: %s", resp.text[:200])
        return {"kind": "error"}
    return outcome

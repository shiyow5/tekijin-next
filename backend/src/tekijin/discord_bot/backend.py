"""The bot's only call into the backend: ``POST /internal/discord/ask`` (Discord D1, #60)."""

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

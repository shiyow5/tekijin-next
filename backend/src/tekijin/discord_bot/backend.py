"""The bot's only call into the backend: ``POST /internal/discord/ask`` (Discord D1, #60)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

SECRET_HEADER = "X-Tekijin-Bot-Secret"

_BUSY_TEXT = "現在混雑しています。少し待ってから、もう一度お試しください。"


async def ask_backend(
    http: httpx.AsyncClient,
    *,
    base_url: str,
    secret: str,
    discord_user_id: int,
    question: str,
) -> dict[str, Any]:
    """Return the backend's outcome dict, or a ``no_answer``/``error`` stand-in.

    Never raises: the bot has already deferred the interaction and must always send
    a follow-up, so every failure becomes a reply the user can read.
    """

    try:
        resp = await http.post(
            f"{base_url.rstrip('/')}/internal/discord/ask",
            json={"discord_user_id": str(discord_user_id), "question": question},
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
    return resp.json()

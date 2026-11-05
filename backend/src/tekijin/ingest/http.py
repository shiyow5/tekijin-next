"""HTTP with bounded retries, for the providers the poller reads from (#99, R4).

Rate limits (429) and server errors (5xx) are retried with exponential backoff,
honouring ``Retry-After`` when the server sends one. Anything else is returned to
the caller as it is, so that a 404 or a 403 is handled where its meaning is known.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 5
BASE_DELAY_S = 1.0
MAX_DELAY_S = 60.0

Sleep = Callable[[float], None]


def send_with_retries(
    send: Callable[[], httpx.Response],
    *,
    should_retry: Callable[[httpx.Response], bool] | None = None,
    sleep: Sleep = time.sleep,
    max_attempts: int = MAX_ATTEMPTS,
) -> httpx.Response:
    """Call ``send`` until it gives a response that is not worth retrying.

    ``should_retry`` adds provider-specific cases (Drive reports some rate limits
    as 403). The last response is returned once the attempts run out, and the last
    transport error is raised.
    """

    for attempt in range(1, max_attempts + 1):
        try:
            response = send()
        except httpx.TransportError as exc:
            if attempt == max_attempts:
                raise
            delay = _backoff(attempt)
            logger.warning("request failed (%s); retrying in %.1fs", type(exc).__name__, delay)
            sleep(delay)
            continue
        retry = response.status_code in RETRY_STATUSES or (
            should_retry is not None and should_retry(response)
        )
        if not retry or attempt == max_attempts:
            return response
        delay = _retry_after(response) or _backoff(attempt)
        logger.warning("HTTP %d; retrying in %.1fs", response.status_code, delay)
        sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover - the loop always returns


def _backoff(attempt: int) -> float:
    return min(MAX_DELAY_S, BASE_DELAY_S * 2 ** (attempt - 1))


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "")
    try:
        seconds = float(value)
    except ValueError:
        return None
    return min(MAX_DELAY_S, max(0.0, seconds))

"""Notion pages shared with an internal integration (#101, R4).

A workspace owner creates an internal integration and members connect it to the
pages TEKIJIN may read (a page's 「接続」 menu; child pages inherit the access).
The poller lists those pages with ``POST /v1/search`` and turns each page's blocks
into plain text by walking ``GET /v1/blocks/{id}/children``.

Notion allows about three requests a second per integration, so every request
waits until :data:`MIN_INTERVAL_S` has passed since the previous one, and a 429
is retried after its ``Retry-After`` by the shared helper in ``ingest/http.py``.

The integration token never reaches a log line.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import httpx

from tekijin.ingest.http import Sleep, send_with_retries
from tekijin.ingest.provider import Listing, RemoteDocument, SkipDocument, parse_time

logger = logging.getLogger(__name__)

NOTION = "notion"
API_URL = "https://api.notion.com/v1"
# The API version every request asks for. 2026-03-11 is the newest at the time of
# writing; it reports trashed pages as ``in_trash`` (``archived`` is still read for
# safety). The search filter value "page" is the same in every version.
NOTION_VERSION = "2026-03-11"
# Notion's limit is an average of three requests a second.
MIN_INTERVAL_S = 0.34
PAGE_SIZE = 100
# Nesting deeper than this is not read: a toggle in a list in a column is depth 3.
MAX_DEPTH = 8
# A page whose text grows past this is skipped, like a Google file over 10 MB.
MAX_TEXT_CHARS = 1_000_000

_HEADINGS = {"heading_1": "# ", "heading_2": "## ", "heading_3": "### "}
_PREFIXES = {
    **_HEADINGS,
    "paragraph": "",
    "bulleted_list_item": "- ",
    "numbered_list_item": "1. ",
    "toggle": "",
    "quote": "> ",
    "callout": "",
    "code": "",
}
# Their children are listed by search themselves (a child page) or are rows that
# are pages of their own (a database), so they are never walked from the parent.
_NOT_WALKED = frozenset({"child_page", "child_database"})

Clock = Callable[[], float]


class NotionProvider:
    """Lists the pages the integration can see and reads their blocks as text."""

    name = NOTION

    def __init__(
        self,
        client: httpx.Client,
        token: str,
        *,
        sleep: Sleep | None = None,
        clock: Clock | None = None,
        min_interval_s: float = MIN_INTERVAL_S,
        max_depth: int = MAX_DEPTH,
        max_chars: int = MAX_TEXT_CHARS,
    ) -> None:
        self._client = client
        self._token = token
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._min_interval_s = min_interval_s
        self._max_depth = max_depth
        self._max_chars = max_chars
        self._last_request: float | None = None
        # Pages the last search listed: a child page among them is ingested on its
        # own and not again inside its parent.
        self._listed: frozenset[str] = frozenset()

    def __repr__(self) -> str:
        return "NotionProvider()"

    def list_documents(self) -> Listing:
        documents: list[RemoteDocument] = []
        cursor: str | None = None
        while True:
            body: dict[str, Any] = {
                "filter": {"property": "object", "value": "page"},
                "page_size": PAGE_SIZE,
            }
            if cursor:
                body["start_cursor"] = cursor
            response = self._send("POST", f"{API_URL}/search", json=body)
            response.raise_for_status()
            page = response.json()
            documents.extend(
                _document(result) for result in page.get("results", ()) if _is_live_page(result)
            )
            cursor = page.get("next_cursor") if page.get("has_more") else None
            if not cursor:
                self._listed = frozenset(_normalise(d.external_id) for d in documents)
                return Listing(documents)

    def fetch_text(self, document: RemoteDocument) -> str:
        lines: list[str] = []
        size = 0
        for line in self._walk(document.external_id, depth=0):
            size += len(line) + 1
            if size > self._max_chars:
                raise SkipDocument(f"longer than {self._max_chars} characters")
            lines.append(line)
        return "\n".join(lines)

    def _walk(self, block_id: str, *, depth: int) -> Iterator[str]:
        for block in self._children(block_id):
            kind = block.get("type")
            if kind == "child_page":
                if _normalise(str(block.get("id", ""))) in self._listed:
                    continue
                title = str((block.get("child_page") or {}).get("title") or "")
                if title:
                    yield f"# {title}"
            else:
                text = block_text(block)
                if text is not None:
                    yield text
            if not block.get("has_children") or kind == "child_database":
                continue
            if depth + 1 > self._max_depth:
                logger.debug("notion: not reading below depth %d", self._max_depth)
                continue
            yield from self._walk(str(block["id"]), depth=depth + 1)

    def _children(self, block_id: str) -> Iterator[Mapping[str, Any]]:
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": PAGE_SIZE}
            if cursor:
                params["start_cursor"] = cursor
            response = self._send("GET", f"{API_URL}/blocks/{block_id}/children", params=params)
            response.raise_for_status()
            page = response.json()
            yield from page.get("results", ())
            cursor = page.get("next_cursor") if page.get("has_more") else None
            if not cursor:
                return

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": NOTION_VERSION,
        }

        def send() -> httpx.Response:
            self._wait_turn()
            return self._client.request(method, url, headers=headers, **kwargs)

        return send_with_retries(send, sleep=self._sleep)

    def _wait_turn(self) -> None:
        if self._last_request is not None:
            wait = self._last_request + self._min_interval_s - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()


def block_text(block: Mapping[str, Any]) -> str | None:
    """One block as a line of plain text, or None for a block without text or of a
    type that is not read (images, embeds, dividers and the like)."""

    kind = str(block.get("type", ""))
    payload = block.get(kind) or {}
    if kind == "table_row":
        return " | ".join(_plain(cell) for cell in payload.get("cells", ()))
    if kind == "to_do":
        text = _plain(payload.get("rich_text", ()))
        return f"[{'x' if payload.get('checked') else ' '}] {text}" if text else None
    if kind not in _PREFIXES:
        return None
    text = _plain(payload.get("rich_text", ()))
    return f"{_PREFIXES[kind]}{text}" if text else None


def _plain(rich_text: Any) -> str:
    return "".join(str(part.get("plain_text") or "") for part in rich_text or ())


def _is_live_page(result: Mapping[str, Any]) -> bool:
    return (
        result.get("object") == "page" and not result.get("in_trash") and not result.get("archived")
    )


def _document(page: Mapping[str, Any]) -> RemoteDocument:
    return RemoteDocument(
        external_id=str(page["id"]),
        title=_title(page.get("properties") or {}),
        url=page.get("url"),
        modified_at=parse_time(str(page["last_edited_time"])),
    )


def _title(properties: Mapping[str, Any]) -> str:
    for prop in properties.values():
        if isinstance(prop, Mapping) and prop.get("type") == "title":
            return _plain(prop.get("title")).strip()
    return ""


def _normalise(page_id: str) -> str:
    """Notion writes the same id with or without hyphens."""

    return page_id.replace("-", "").lower()


def build_notion_provider(token: str, client: httpx.Client) -> NotionProvider:
    """The provider for an internal integration token. Raises ``ValueError`` with a
    message that never contains the token."""

    token = token.strip()
    if not token:
        raise ValueError("TEKIJIN_NOTION_TOKEN is empty")
    logger.info("notion: ingesting the pages connected to the integration")
    return NotionProvider(client, token)

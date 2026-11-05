"""The Notion provider against a fake Notion API (#101, R4). No network is used:
every request goes to an ``httpx.MockTransport``."""

from __future__ import annotations

import datetime as dt
import json

import httpx
import pytest

from tekijin.ingest import http as retry_http
from tekijin.ingest.notion import (
    API_URL,
    MIN_INTERVAL_S,
    NOTION_VERSION,
    NotionProvider,
    block_text,
    build_notion_provider,
)
from tekijin.ingest.provider import RemoteDocument, SkipDocument

PAGE = RemoteDocument(
    "11111111-1111-1111-1111-111111111111",
    "部室の使い方",
    "https://www.notion.so/11111111111111111111111111111111",
    dt.datetime(2026, 11, 1, 3, 0),
)


def _rich(text: str) -> list[dict]:
    return [{"type": "text", "plain_text": text, "text": {"content": text}}]


def _page(page_id: str, title: str = "題名", **extra) -> dict:
    page = {
        "object": "page",
        "id": page_id,
        "url": f"https://www.notion.so/{page_id.replace('-', '')}",
        "last_edited_time": "2026-11-01T03:00:00.000Z",
        "in_trash": False,
        "properties": {
            "タグ": {"id": "a", "type": "multi_select", "multi_select": []},
            "名前": {"id": "title", "type": "title", "title": _rich(title)},
        },
    }
    page.update(extra)
    return page


def _block(block_id: str, kind: str, text: str = "", *, children: bool = False, **payload):
    body = {"rich_text": _rich(text), **payload} if text or kind != "child_page" else payload
    return {"object": "block", "id": block_id, "type": kind, "has_children": children, kind: body}


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _provider(handler, clock: FakeClock | None = None, **kwargs) -> NotionProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    clock = clock or FakeClock()
    return NotionProvider(client, "secret-token", sleep=clock.sleep, clock=clock, **kwargs)


def _children_handler(tree: dict[str, list[list[dict]]], seen: list[httpx.Request] | None = None):
    """Serve ``tree[block_id]`` as pages of block children."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        block_id = request.url.path.split("/")[-2]
        pages = tree[block_id]
        index = int(request.url.params.get("start_cursor", "0"))
        more = index + 1 < len(pages)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "results": pages[index],
                "has_more": more,
                "next_cursor": str(index + 1) if more else None,
            },
        )

    return handler


# ---- listing ------------------------------------------------------------------------ #


def test_search_asks_for_pages_and_follows_cursors() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        if body.get("start_cursor") == "c2":
            return httpx.Response(
                200, json={"results": [_page("p2", "会計")], "has_more": False, "next_cursor": None}
            )
        return httpx.Response(
            200, json={"results": [_page("p1", "新歓")], "has_more": True, "next_cursor": "c2"}
        )

    listing = _provider(handler).list_documents()

    assert [d.external_id for d in listing.documents] == ["p1", "p2"]
    assert listing.complete
    first = seen[0]
    assert first.method == "POST"
    assert str(first.url) == f"{API_URL}/search"
    assert first.headers["authorization"] == "Bearer secret-token"
    assert first.headers["notion-version"] == NOTION_VERSION
    body = json.loads(first.content)
    assert body["filter"] == {"property": "object", "value": "page"}
    assert "start_cursor" not in body
    assert json.loads(seen[1].content)["start_cursor"] == "c2"
    page = listing.documents[0]
    assert page.title == "新歓"
    assert page.url == "https://www.notion.so/p1"
    assert page.modified_at == dt.datetime(2026, 11, 1, 3, 0)


def test_the_title_comes_from_whichever_property_is_the_title() -> None:
    row = _page("r1")
    row["properties"] = {
        "Name": {"type": "title", "title": _rich("議事録") + _rich(" 11月")},
        "日付": {"type": "date", "date": None},
    }
    untitled = _page("u1")
    untitled["properties"] = {"Name": {"type": "title", "title": []}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [row, untitled], "has_more": False})

    titles = [d.title for d in _provider(handler).list_documents().documents]
    assert titles == ["議事録 11月", ""]


def test_pages_in_the_trash_and_non_pages_are_skipped() -> None:
    results = [
        _page("keep"),
        _page("trashed", in_trash=True),
        _page("archived", archived=True),
        {"object": "data_source", "id": "ds1"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": results, "has_more": False})

    assert [d.external_id for d in _provider(handler).list_documents().documents] == ["keep"]


def test_an_error_in_the_middle_of_the_search_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content).get("start_cursor"):
            return httpx.Response(400, json={"object": "error", "code": "validation_error"})
        return httpx.Response(
            200, json={"results": [_page("p1")], "has_more": True, "next_cursor": "c2"}
        )

    with pytest.raises(httpx.HTTPStatusError):
        _provider(handler).list_documents()


# ---- text --------------------------------------------------------------------------- #


def test_block_text_covers_the_common_block_types() -> None:
    assert block_text(_block("b", "paragraph", "本文")) == "本文"
    assert block_text(_block("b", "heading_1", "見出し")) == "# 見出し"
    assert block_text(_block("b", "heading_3", "小見出し")) == "### 小見出し"
    assert block_text(_block("b", "bulleted_list_item", "項目")) == "- 項目"
    assert block_text(_block("b", "numbered_list_item", "手順")) == "1. 手順"
    assert block_text(_block("b", "to_do", "買う", checked=True)) == "[x] 買う"
    assert block_text(_block("b", "to_do", "払う", checked=False)) == "[ ] 払う"
    assert block_text(_block("b", "toggle", "詳細")) == "詳細"
    assert block_text(_block("b", "quote", "名言")) == "> 名言"
    assert block_text(_block("b", "callout", "注意")) == "注意"
    assert block_text(_block("b", "code", "print(1)", language="python")) == "print(1)"
    row = {"type": "table_row", "table_row": {"cells": [_rich("鍵"), [], _rich("守衛室")]}}
    assert block_text(row) == "鍵 |  | 守衛室"
    assert block_text(_block("b", "paragraph", "")) is None
    assert block_text({"type": "image", "image": {"type": "external"}}) is None
    assert block_text({"type": "divider", "divider": {}}) is None


def test_the_text_walks_nested_blocks_and_their_pages() -> None:
    page_id = PAGE.external_id
    tree = {
        page_id: [
            [
                _block("h", "heading_2", "鍵"),
                _block("t", "toggle", "借り方", children=True),
            ],
            [_block("tb", "table", children=True, table_width=2)],
        ],
        "t": [[_block("li", "bulleted_list_item", "守衛室で名前を書く", children=True)]],
        "li": [[_block("p", "paragraph", "学生証が要る")]],
        "tb": [
            [
                {
                    "type": "table_row",
                    "has_children": False,
                    "table_row": {"cells": [_rich("曜日"), _rich("時間")]},
                }
            ]
        ],
    }
    seen: list[httpx.Request] = []
    text = _provider(_children_handler(tree, seen)).fetch_text(PAGE)

    assert text.splitlines() == [
        "## 鍵",
        "借り方",
        "- 守衛室で名前を書く",
        "学生証が要る",
        "曜日 | 時間",
    ]
    assert seen[0].url.path == f"/v1/blocks/{page_id}/children"
    assert seen[0].url.params["page_size"] == "100"
    second_page = [r for r in seen if r.url.params.get("start_cursor") == "1"]
    assert [r.url.path for r in second_page] == [f"/v1/blocks/{page_id}/children"]


def test_child_pages_are_included_only_when_search_does_not_list_them() -> None:
    parent, listed, hidden = "parent", "listed-child", "hidden-child"
    tree = {
        parent: [
            [
                _block("a", "paragraph", "親の本文"),
                {
                    "id": listed,
                    "type": "child_page",
                    "has_children": True,
                    "child_page": {"title": "別に取り込む子"},
                },
                {
                    "id": hidden,
                    "type": "child_page",
                    "has_children": True,
                    "child_page": {"title": "一覧に無い子"},
                },
                {
                    "id": "db",
                    "type": "child_database",
                    "has_children": True,
                    "child_database": {"title": "表"},
                },
            ]
        ],
        hidden: [[_block("c", "paragraph", "子の本文")]],
    }
    seen: list[httpx.Request] = []
    children = _children_handler(tree, seen)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search":
            pages = [_page(parent), _page(listed.upper())]
            return httpx.Response(200, json={"results": pages, "has_more": False})
        return children(request)

    provider = _provider(handler)
    documents = provider.list_documents().documents
    text = provider.fetch_text(documents[0])

    assert text.splitlines() == ["親の本文", "# 一覧に無い子", "子の本文"]
    fetched = {r.url.path for r in seen}
    assert f"/v1/blocks/{listed}/children" not in fetched
    assert "/v1/blocks/db/children" not in fetched


def test_recursion_stops_at_the_depth_limit() -> None:
    tree = {
        PAGE.external_id: [[_block("d1", "toggle", "1", children=True)]],
        "d1": [[_block("d2", "toggle", "2", children=True)]],
        "d2": [[_block("d3", "toggle", "3", children=True)]],
        "d3": [[_block("d4", "paragraph", "4")]],
    }
    text = _provider(_children_handler(tree), max_depth=2).fetch_text(PAGE)
    assert text.splitlines() == ["1", "2", "3"]


def test_a_page_over_the_size_limit_is_skipped() -> None:
    tree = {PAGE.external_id: [[_block(f"b{i}", "paragraph", "あ" * 40) for i in range(5)]]}
    with pytest.raises(SkipDocument):
        _provider(_children_handler(tree), max_chars=100).fetch_text(PAGE)


# ---- rate limits -------------------------------------------------------------------- #


def test_requests_are_spaced_to_stay_under_three_per_second() -> None:
    tree = {PAGE.external_id: [[_block("a", "paragraph", "1")], [_block("b", "paragraph", "2")]]}
    clock = FakeClock()
    provider = _provider(_children_handler(tree), clock)

    provider.fetch_text(PAGE)

    assert MIN_INTERVAL_S >= 0.34
    assert clock.sleeps == [pytest.approx(MIN_INTERVAL_S)]


def test_slow_responses_are_not_spaced_further() -> None:
    clock = FakeClock()
    tree = {PAGE.external_id: [[_block("a", "paragraph", "1")], [_block("b", "paragraph", "2")]]}
    serve = _children_handler(tree)

    def handler(request: httpx.Request) -> httpx.Response:
        clock.now += 1.0  # the request itself took a second
        return serve(request)

    _provider(handler, clock).fetch_text(PAGE)
    assert clock.sleeps == []


def test_a_rate_limited_request_waits_for_retry_after() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "2"}, json={"code": "rate_limited"}),
            httpx.Response(
                200, json={"results": [_block("a", "paragraph", "ok")], "has_more": False}
            ),
        ]
    )
    clock = FakeClock()
    assert _provider(lambda r: next(responses), clock).fetch_text(PAGE) == "ok"
    assert clock.sleeps == [2.0]


def test_retries_are_bounded() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    with pytest.raises(httpx.HTTPStatusError):
        _provider(handler).fetch_text(PAGE)
    assert len(calls) == retry_http.MAX_ATTEMPTS


def test_a_page_that_is_no_longer_shared_raises_without_retrying() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"object": "error", "code": "object_not_found"})

    with pytest.raises(httpx.HTTPStatusError):
        _provider(handler).fetch_text(PAGE)
    assert len(calls) == 1


# ---- building ----------------------------------------------------------------------- #


def test_build_refuses_an_empty_token() -> None:
    with pytest.raises(ValueError, match="TEKIJIN_NOTION_TOKEN"):
        build_notion_provider("  ", httpx.Client())


def test_build_never_logs_the_token(caplog) -> None:
    with caplog.at_level("DEBUG"):
        provider = build_notion_provider("ntn_super_secret", httpx.Client())
    assert provider.name == "notion"
    assert "ntn_super_secret" not in caplog.text
    assert "ntn_super_secret" not in repr(provider)

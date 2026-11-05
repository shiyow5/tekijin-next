"""The Google Drive provider against a fake Drive (#99, R4). No network is used:
every request goes to an ``httpx.MockTransport``."""

from __future__ import annotations

import datetime as dt
import json
import os

import httpx
import pytest

from tekijin.ingest import http as retry_http
from tekijin.ingest.google import (
    DOCUMENT,
    EXPORT_LIMIT_BYTES,
    FILES_URL,
    PRESENTATION,
    SCOPE,
    GoogleDriveProvider,
    ServiceAccountToken,
    build_google_provider,
    parse_time,
)
from tekijin.ingest.provider import RemoteDocument, SkipDocument

DOC = RemoteDocument(
    "doc1",
    "部室の使い方",
    "https://docs.google.com/document/d/doc1/edit",
    dt.datetime(2026, 11, 1, 3, 0),
)


def _file(
    file_id: str, name: str, mime: str = DOCUMENT, modified: str = "2026-11-01T03:00:00.000Z"
):
    kind = "document" if mime == DOCUMENT else "presentation"
    return {
        "id": file_id,
        "name": name,
        "mimeType": mime,
        "modifiedTime": modified,
        "webViewLink": f"https://docs.google.com/{kind}/d/{file_id}/edit",
    }


def _provider(handler, sleeps=None) -> GoogleDriveProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    recorded = sleeps if sleeps is not None else []
    return GoogleDriveProvider(client, lambda: "tok", sleep=recorded.append)


def test_listing_asks_for_docs_and_slides_in_all_drives_and_follows_pages() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params.get("pageToken") == "p2":
            return httpx.Response(200, json={"files": [_file("s1", "新歓", PRESENTATION)]})
        return httpx.Response(200, json={"files": [_file("d1", "会計")], "nextPageToken": "p2"})

    listing = _provider(handler).list_documents()

    assert [d.external_id for d in listing.documents] == ["d1", "s1"]
    assert listing.complete
    first = seen[0]
    assert str(first.url).startswith(FILES_URL)
    assert first.headers["authorization"] == "Bearer tok"
    params = first.url.params
    assert DOCUMENT in params["q"] and PRESENTATION in params["q"]
    assert "trashed = false" in params["q"]
    assert params["includeItemsFromAllDrives"] == "true"
    assert params["supportsAllDrives"] == "true"
    assert params["corpora"] == "allDrives"
    assert "modifiedTime" in params["fields"] and "webViewLink" in params["fields"]
    assert "pageToken" not in params
    assert seen[1].url.params["pageToken"] == "p2"
    slides = listing.documents[1]
    assert slides.title == "新歓"
    assert slides.url == "https://docs.google.com/presentation/d/s1/edit"
    assert slides.modified_at == dt.datetime(2026, 11, 1, 3, 0)


def test_an_incomplete_search_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"files": [], "incompleteSearch": True})

    assert not _provider(handler).list_documents().complete


def test_a_failed_listing_raises() -> None:
    provider = _provider(lambda r: httpx.Response(401, json={}))
    with pytest.raises(httpx.HTTPStatusError):
        provider.list_documents()


def test_parse_time_converts_to_naive_utc() -> None:
    assert parse_time("2026-11-01T12:00:00.250+09:00") == dt.datetime(2026, 11, 1, 3, 0, 0, 250000)
    assert parse_time("2026-11-01T03:00:00") == dt.datetime(2026, 11, 1, 3, 0)


def test_export_reads_plain_text_without_the_byte_order_mark() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content="﻿鍵は守衛室".encode())

    assert _provider(handler).fetch_text(DOC) == "鍵は守衛室"
    assert seen[0].url.path == "/drive/v3/files/doc1/export"
    assert seen[0].url.params["mimeType"] == "text/plain"


def test_rate_limits_and_server_errors_are_retried() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "3"}),
            httpx.Response(503),
            httpx.Response(403, json={"error": {"errors": [{"reason": "userRateLimitExceeded"}]}}),
            httpx.Response(200, content=b"ok"),
        ]
    )
    sleeps: list[float] = []
    assert _provider(lambda r: next(responses), sleeps).fetch_text(DOC) == "ok"
    assert sleeps == [3.0, 2.0, 4.0]


def test_retries_are_bounded() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500)

    with pytest.raises(httpx.HTTPStatusError):
        _provider(handler).fetch_text(DOC)
    assert len(calls) == retry_http.MAX_ATTEMPTS


def test_other_errors_are_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, json={"error": {"errors": [{"reason": "notFound"}]}})

    with pytest.raises(httpx.HTTPStatusError):
        _provider(handler).fetch_text(DOC)
    assert len(calls) == 1


def test_transport_errors_are_retried_then_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    sleeps: list[float] = []
    with pytest.raises(httpx.ConnectError):
        _provider(handler, sleeps).fetch_text(DOC)
    assert len(sleeps) == retry_http.MAX_ATTEMPTS - 1


def test_a_document_over_the_export_limit_is_skipped() -> None:
    too_large = httpx.Response(
        403, json={"error": {"errors": [{"reason": "exportSizeLimitExceeded"}]}}
    )
    with pytest.raises(SkipDocument):
        _provider(lambda r: too_large).fetch_text(DOC)
    huge = httpx.Response(200, content=b"a" * (EXPORT_LIMIT_BYTES + 1))
    with pytest.raises(SkipDocument):
        _provider(lambda r: huge).fetch_text(DOC)


def test_a_403_without_json_is_not_retried() -> None:
    with pytest.raises(httpx.HTTPStatusError):
        _provider(lambda r: httpx.Response(403, content=b"<html>")).fetch_text(DOC)


# ---- service account token ---------------------------------------------------------- #


def _key_file(tmp_path) -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    path = tmp_path / "sa.json"
    path.write_text(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "test",
                "private_key_id": "kid",
                "private_key": pem,
                "client_email": "tekijin@test.iam.gserviceaccount.com",
                "client_id": "1",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )
    )
    os.chmod(path, 0o600)
    return str(path)


def test_the_token_is_fetched_once_and_reused(tmp_path) -> None:
    token_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        token_calls.append(request)
        return httpx.Response(200, json={"access_token": "ya29.test", "expires_in": 3600})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    token = ServiceAccountToken.from_file(_key_file(tmp_path), client)

    assert token.account_email == "tekijin@test.iam.gserviceaccount.com"
    assert token() == "ya29.test"
    assert token() == "ya29.test"
    assert len(token_calls) == 1
    assert token_calls[0].url == "https://oauth2.googleapis.com/token"
    assert b"grant_type" in token_calls[0].content


def test_a_failed_token_request_raises(tmp_path) -> None:
    import google.auth.exceptions

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    token = ServiceAccountToken.from_file(_key_file(tmp_path), client)
    with pytest.raises(google.auth.exceptions.TransportError):
        token()


def test_build_reads_the_key_with_the_read_only_scope(tmp_path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    provider = build_google_provider(_key_file(tmp_path), client)
    assert provider.name == "google"
    assert provider._token._credentials.scopes == [SCOPE]  # type: ignore[attr-defined]


def test_build_refuses_a_missing_or_broken_key_without_echoing_it(tmp_path) -> None:
    client = httpx.Client()
    with pytest.raises(ValueError, match="does not point to a file"):
        build_google_provider(str(tmp_path / "missing.json"), client)
    broken = tmp_path / "broken.json"
    broken.write_text('{"private_key": "SECRET-MATERIAL"}')
    os.chmod(broken, 0o600)
    with pytest.raises(ValueError) as info:
        build_google_provider(str(broken), client)
    assert "SECRET-MATERIAL" not in str(info.value)


def test_a_key_readable_by_others_is_warned_about(tmp_path, caplog) -> None:
    path = _key_file(tmp_path)
    os.chmod(path, 0o644)
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with caplog.at_level("WARNING"):
        build_google_provider(path, client)
    assert "chmod 600" in caplog.text

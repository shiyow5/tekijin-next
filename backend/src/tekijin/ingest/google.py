"""Google Docs and Slides shared with a service account (#99, R4).

The service account is a Google account for programs, with its own e-mail address.
Members share a document or a folder with that address (as a viewer), and the
poller lists what the account can see with Drive v3 ``files.list`` and reads each
document's text with ``files.export``. The account only needs the read-only Drive
scope; nothing in the organization's Google admin console is required.

The key file and the access token never reach a log line.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import stat
import time
from collections.abc import Callable, Mapping
from typing import Any

import google.auth.exceptions
import google.auth.transport
import httpx
from google.oauth2 import service_account

from tekijin.ingest.http import Sleep, send_with_retries
from tekijin.ingest.provider import Listing, RemoteDocument, SkipDocument

logger = logging.getLogger(__name__)

GOOGLE = "google"
SCOPE = "https://www.googleapis.com/auth/drive.readonly"
FILES_URL = "https://www.googleapis.com/drive/v3/files"
DOCUMENT = "application/vnd.google-apps.document"
PRESENTATION = "application/vnd.google-apps.presentation"
# Drive refuses to export more than this; a larger body is skipped all the same.
EXPORT_LIMIT_BYTES = 10 * 1024 * 1024
PAGE_SIZE = 100
QUERY = f"(mimeType = '{DOCUMENT}' or mimeType = '{PRESENTATION}') and trashed = false"
FIELDS = "nextPageToken, incompleteSearch, files(id, name, mimeType, modifiedTime, webViewLink)"
# Drive reports these per-user quotas as 403 instead of 429.
_RATE_LIMIT_REASONS = frozenset({"rateLimitExceeded", "userRateLimitExceeded"})
_TOO_LARGE_REASON = "exportSizeLimitExceeded"

TokenSource = Callable[[], str]


class GoogleDriveProvider:
    """Lists and exports the Docs and Slides the service account can see."""

    name = GOOGLE

    def __init__(
        self,
        client: httpx.Client,
        token: TokenSource,
        *,
        sleep: Sleep | None = None,
        page_size: int = PAGE_SIZE,
    ) -> None:
        self._client = client
        self._token = token
        self._sleep = sleep or time.sleep
        self._page_size = page_size

    def list_documents(self) -> Listing:
        documents: list[RemoteDocument] = []
        complete = True
        page_token: str | None = None
        while True:
            params: dict[str, Any] = {
                "q": QUERY,
                "fields": FIELDS,
                "pageSize": self._page_size,
                "corpora": "allDrives",
                "includeItemsFromAllDrives": "true",
                "supportsAllDrives": "true",
            }
            if page_token:
                params["pageToken"] = page_token
            response = self._get(FILES_URL, params)
            response.raise_for_status()
            page = response.json()
            documents.extend(_document(f) for f in page.get("files", ()))
            complete = complete and not page.get("incompleteSearch", False)
            page_token = page.get("nextPageToken")
            if not page_token:
                return Listing(documents, complete=complete)

    def fetch_text(self, document: RemoteDocument) -> str:
        response = self._get(
            f"{FILES_URL}/{document.external_id}/export", {"mimeType": "text/plain"}
        )
        if response.status_code == 403 and _TOO_LARGE_REASON in _reasons(response):
            raise SkipDocument("larger than Drive's 10 MB export limit")
        response.raise_for_status()
        if len(response.content) > EXPORT_LIMIT_BYTES:
            raise SkipDocument("larger than the 10 MB export limit")
        # Drive starts a text export with a byte order mark.
        return response.content.decode("utf-8", errors="replace").lstrip("﻿")

    def _get(self, url: str, params: Mapping[str, Any]) -> httpx.Response:
        def send() -> httpx.Response:
            headers = {"Authorization": f"Bearer {self._token()}"}
            return self._client.get(url, params=params, headers=headers)

        return send_with_retries(send, should_retry=_is_rate_limited, sleep=self._sleep)


def _document(file: Mapping[str, Any]) -> RemoteDocument:
    return RemoteDocument(
        external_id=str(file["id"]),
        title=str(file.get("name") or ""),
        url=file.get("webViewLink"),
        modified_at=parse_time(str(file["modifiedTime"])),
    )


def parse_time(value: str) -> dt.datetime:
    """An RFC 3339 time from Drive as naive UTC, the form ``source_items`` stores."""

    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(dt.UTC).replace(tzinfo=None)


def _reasons(response: httpx.Response) -> set[str]:
    try:
        errors = response.json().get("error", {}).get("errors", ())
        return {str(e.get("reason")) for e in errors}
    except (ValueError, AttributeError):
        return set()


def _is_rate_limited(response: httpx.Response) -> bool:
    return response.status_code == 403 and bool(_reasons(response) & _RATE_LIMIT_REASONS)


class ServiceAccountToken:
    """An access token for the service account, refreshed when it has expired."""

    def __init__(self, credentials: Any, client: httpx.Client) -> None:
        self._credentials = credentials
        self._request = HttpxRequest(client)

    @classmethod
    def from_file(cls, path: str, client: httpx.Client) -> ServiceAccountToken:
        credentials = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            path, scopes=[SCOPE]
        )
        return cls(credentials, client)

    @property
    def account_email(self) -> str:
        return str(self._credentials.service_account_email)

    def __call__(self) -> str:
        if not self._credentials.valid:
            self._credentials.refresh(self._request)
        return str(self._credentials.token)


class HttpxRequest(google.auth.transport.Request):  # type: ignore[misc]
    """google-auth's transport interface over httpx, so no second HTTP stack is
    needed to fetch the token."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def __call__(
        self,
        url: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> _HttpxResponse:
        try:
            response = self._client.request(
                method, url, content=body, headers=dict(headers or {}), timeout=timeout or 60.0
            )
        except httpx.HTTPError as exc:
            raise google.auth.exceptions.TransportError(type(exc).__name__) from exc
        return _HttpxResponse(response)


class _HttpxResponse(google.auth.transport.Response):  # type: ignore[misc]
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    @property
    def status(self) -> int:
        return self._response.status_code

    @property
    def headers(self) -> Mapping[str, str]:
        return self._response.headers

    @property
    def data(self) -> bytes:
        return self._response.content


def warn_if_readable_by_others(path: str) -> None:
    """Log a warning when the key file can be read by other users of the server."""

    mode = os.stat(path).st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        logger.warning("the service account key file is readable by others; chmod 600 it")


def build_google_provider(path: str, client: httpx.Client) -> GoogleDriveProvider:
    """The provider for the key file at ``path``. Raises ``ValueError`` with a message
    that names the problem but not the file's contents."""

    if not os.path.isfile(path):
        raise ValueError("TEKIJIN_GOOGLE_SERVICE_ACCOUNT_FILE does not point to a file")
    warn_if_readable_by_others(path)
    try:
        token = ServiceAccountToken.from_file(path, client)
    except (ValueError, KeyError, OSError) as exc:
        raise ValueError(
            f"could not read the service account key file ({type(exc).__name__})"
        ) from None
    logger.info("google: ingesting what is shared with %s", token.account_email)
    return GoogleDriveProvider(client, token)

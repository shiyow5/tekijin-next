"""Scheduled ingestion of shared documents (#99, R4).

Every ``TEKIJIN_POLL_INTERVAL_S`` seconds each configured provider lists its
documents. New or changed ones are fetched and stored as ``source_items``; ones
whose modification time and title are unchanged are not fetched at all; stored
ones the provider no longer lists (unshared, trashed) are deleted. One failing
document never stops the pass, and a failed listing deletes nothing.

A provider is anything that satisfies :class:`tekijin.ingest.provider.Provider`.
Adding one means writing that class and one branch in
:func:`build_providers`.

CLI::

    python -m tekijin.ingest.poll           # until SIGINT / SIGTERM
    python -m tekijin.ingest.poll --once    # one pass, then exit
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import FrameType
from typing import Any

import httpx
from sqlalchemy.orm import Session, sessionmaker

from tekijin.config import Settings, get_settings
from tekijin.data.sources import (
    StoredVersion,
    delete_source_item,
    set_source_updated_at,
    stored_versions,
    upsert_source_item,
)
from tekijin.ingest.provider import Provider, RemoteDocument, SkipDocument
from tekijin.retrieval.embedding import Embedder

logger = logging.getLogger(__name__)

HTTP_TIMEOUT_S = 60.0

SessionFactory = sessionmaker[Session]


@dataclass(frozen=True, slots=True)
class SyncReport:
    provider: str
    listed: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    failed: int = 0
    deleted: int = 0
    listing_failed: bool = False

    @property
    def ok(self) -> bool:
        return not self.listing_failed and self.failed == 0


def sync_provider(
    session_factory: SessionFactory, embedder: Embedder, *, org_id: int, provider: Provider
) -> SyncReport:
    """One pass over one provider. Each document is stored in its own transaction."""

    report = SyncReport(provider.name)
    try:
        listing = provider.list_documents()
    except Exception as exc:
        logger.error("%s: listing failed (%s); nothing is deleted", provider.name, _describe(exc))
        return replace(report, listing_failed=True)

    documents = {d.external_id: d for d in listing.documents}
    with session_factory() as session:
        stored = stored_versions(session, org_id=org_id, provider=provider.name)
    report = replace(report, listed=len(documents))
    for document in documents.values():
        outcome = _sync_document(
            session_factory, embedder, org_id, provider, document, stored.get(document.external_id)
        )
        report = replace(report, **{outcome: getattr(report, outcome) + 1})

    gone = [v.item_id for key, v in stored.items() if key not in documents]
    if gone and not listing.complete:
        logger.warning("%s: the listing may be incomplete; nothing is deleted", provider.name)
    elif gone:
        with session_factory() as session:
            for item_id in gone:
                delete_source_item(session, item_id=item_id, org_id=org_id)
            session.commit()
        report = replace(report, deleted=len(gone))
    logger.info("%s", report)
    return report


def _sync_document(
    session_factory: SessionFactory,
    embedder: Embedder,
    org_id: int,
    provider: Provider,
    document: RemoteDocument,
    known: StoredVersion | None,
) -> str:
    """Store one document and say what happened: a :class:`SyncReport` field."""

    if (
        known is not None
        and known.updated_at == document.modified_at
        and known.title == document.title
        and known.url == document.url
    ):
        return "unchanged"
    try:
        text = provider.fetch_text(document)
    except SkipDocument as exc:
        logger.warning("%s: skipped %s: %s", provider.name, document.external_id, exc)
        return "skipped"
    except Exception as exc:
        logger.error(
            "%s: could not fetch %s (%s)", provider.name, document.external_id, _describe(exc)
        )
        return "failed"
    try:
        with session_factory() as session:
            outcome = _store(session, embedder, org_id, provider.name, document, text, known)
            session.commit()
        return outcome
    except Exception:
        logger.exception("%s: could not store %s", provider.name, document.external_id)
        return "failed"


def _store(
    session: Session,
    embedder: Embedder,
    org_id: int,
    provider: str,
    document: RemoteDocument,
    text: str,
    known: StoredVersion | None,
) -> str:
    if not text.strip():
        # An emptied document must not keep answering with its old text.
        if known is not None:
            delete_source_item(session, item_id=known.item_id, org_id=org_id)
        logger.info("%s: %s has no text; not stored", provider, document.external_id)
        return "skipped"
    result = upsert_source_item(
        session,
        embedder,
        org_id=org_id,
        provider=provider,
        external_id=document.external_id,
        title=document.title or None,
        body=text,
        url=document.url,
        author_id=None,
        now=document.modified_at,
    )
    if result.status == "unchanged":
        # Same text, newer time: remember the time so it is not fetched again.
        set_source_updated_at(session, item_id=result.item_id, updated_at=document.modified_at)
    return result.status


def _describe(exc: BaseException) -> str:
    """An error for a log line: the HTTP status when there is one, else the type.
    Never the request headers, which carry the token."""

    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


def run_pass(
    session_factory: SessionFactory,
    embedder: Embedder,
    *,
    org_id: int,
    providers: Sequence[Provider],
) -> list[SyncReport]:
    return [sync_provider(session_factory, embedder, org_id=org_id, provider=p) for p in providers]


def run_forever(run_once: Callable[[], object], *, interval_s: float, stop: threading.Event) -> int:
    """Run passes ``interval_s`` apart until ``stop`` is set. A pass that raises is
    logged and the next one runs on schedule. Returns how many passes ran."""

    passes = 0
    while not stop.is_set():
        try:
            run_once()
        except Exception:
            logger.exception("ingestion pass failed")
        passes += 1
        if stop.wait(interval_s):
            break
    return passes


def build_providers(settings: Settings, client: httpx.Client) -> list[Provider]:
    """The providers whose settings are present. Raises ``ValueError`` when one is
    configured but cannot start."""

    providers: list[Provider] = []
    if settings.google_service_account_file:
        from tekijin.ingest.google import build_google_provider

        providers.append(build_google_provider(settings.google_service_account_file, client))
    notion_token = settings.notion_token.get_secret_value()
    if notion_token.strip():
        from tekijin.ingest.notion import build_notion_provider

        providers.append(build_notion_provider(notion_token, client))
    return providers


def resolve_org_id(session_factory: SessionFactory, guild_id: int) -> int:
    from tekijin.data.members import get_or_create_org_for_guild

    with session_factory() as session:
        org = get_or_create_org_for_guild(session, str(guild_id), name=f"Discord {guild_id}")
        session.commit()
        return org.id


def _connect(settings: Settings) -> tuple[SessionFactory, Embedder]:  # pragma: no cover
    from tekijin.data.db import get_engine, get_sessionmaker
    from tekijin.retrieval.embedding import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        use_e5_prefix=settings.embedding_use_e5_prefix,
        query_prefix=settings.embedding_query_prefix,
        passage_prefix=settings.embedding_passage_prefix,
        trust_remote_code=settings.embedding_trust_remote_code,
        revision=settings.embedding_model_revision,
        app_env=settings.app_env,
    )
    return get_sessionmaker(get_engine(settings.database_url)), embedder


@contextmanager
def _stop_on_signals(stop: threading.Event) -> Iterator[None]:
    def handle(signum: int, frame: FrameType | None) -> None:
        logger.info("stopping after the current pass")
        stop.set()

    previous: dict[signal.Signals, Any] = {
        sig: signal.signal(sig, handle) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(
    argv: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    make_providers: Callable[[Settings, httpx.Client], list[Provider]] = build_providers,
    connect: Callable[[Settings], tuple[SessionFactory, Embedder]] = _connect,
    stop: threading.Event | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="python -m tekijin.ingest.poll")
    parser.add_argument("--once", action="store_true", help="run one pass and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx logs every request URL at INFO; keep the log to what the poller says.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = settings or get_settings()
    if settings.discord_guild_id is None:
        sys.stderr.write("TEKIJIN_DISCORD_GUILD_ID is not set: it names the organization.\n")
        return 2
    with httpx.Client(timeout=HTTP_TIMEOUT_S) as client:
        try:
            providers = make_providers(settings, client)
        except ValueError as exc:
            sys.stderr.write(f"{exc}\n")
            return 2
        if not providers:
            sys.stderr.write(
                "Nothing to ingest: set TEKIJIN_GOOGLE_SERVICE_ACCOUNT_FILE to the path of "
                "a Google service account key, or TEKIJIN_NOTION_TOKEN to the token of a "
                "Notion internal integration.\n"
            )
            return 2
        session_factory, embedder = connect(settings)
        org_id = resolve_org_id(session_factory, settings.discord_guild_id)

        def once() -> list[SyncReport]:
            return run_pass(session_factory, embedder, org_id=org_id, providers=providers)

        if args.once:
            return 0 if all(r.ok for r in once()) else 1
        stop = stop or threading.Event()
        with _stop_on_signals(stop):
            run_forever(once, interval_s=settings.poll_interval_s, stop=stop)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())

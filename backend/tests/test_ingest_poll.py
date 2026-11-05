"""The provider-neutral poller (#99, R4), with a fake provider and a real database.

What a pass must guarantee: new and changed documents are stored, unchanged ones
are not even fetched, documents no longer listed are deleted, a failed listing
deletes nothing, and one failing document does not stop the others.
"""

from __future__ import annotations

import datetime as dt
import threading

import httpx
import pytest
from sqlalchemy import select, text

from tekijin.config import Settings
from tekijin.data.db import get_sessionmaker
from tekijin.data.sources import upsert_source_item
from tekijin.ingest.poll import (
    build_providers,
    main,
    run_forever,
    sync_provider,
)
from tekijin.ingest.provider import Listing, RemoteDocument, SkipDocument
from tekijin.models.tables import Organization, SourceItem

T1 = dt.datetime(2026, 11, 1, 3, 0)
T2 = dt.datetime(2026, 11, 2, 3, 0)
GUILD = 4242


def _doc(external_id: str, *, title: str = "題名", modified: dt.datetime = T1) -> RemoteDocument:
    return RemoteDocument(
        external_id, title, f"https://docs.google.com/document/d/{external_id}/edit", modified
    )


class FakeProvider:
    name = "google"

    def __init__(self, documents=(), texts=None, *, complete=True, fail_listing=False):
        self.documents = list(documents)
        self.texts = dict(texts or {})
        self.complete = complete
        self.fail_listing = fail_listing
        self.fetched: list[str] = []

    def list_documents(self) -> Listing:
        if self.fail_listing:
            raise httpx.ConnectError("down")
        return Listing(self.documents, complete=self.complete)

    def fetch_text(self, document: RemoteDocument) -> str:
        self.fetched.append(document.external_id)
        value = self.texts.get(document.external_id, f"{document.external_id} 本文")
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def org(engine):
    factory = get_sessionmaker(engine)
    with factory() as session:
        mine = Organization(name="テストサークル", discord_guild_id=str(GUILD))
        other = Organization(name="ほかの組織", discord_guild_id="999")
        session.add_all([mine, other])
        session.commit()
        ids = (mine.id, other.id)
    yield ids
    with factory() as session:
        session.execute(text("DELETE FROM chunks"))
        session.execute(text("DELETE FROM source_items"))
        session.execute(text("DELETE FROM organizations"))
        session.commit()


def _items(engine, org_id: int, provider: str = "google") -> dict[str, SourceItem]:
    with get_sessionmaker(engine)() as session:
        rows = session.scalars(
            select(SourceItem).where(SourceItem.org_id == org_id, SourceItem.provider == provider)
        ).all()
        return {r.external_id: r for r in rows}


def _sync(engine, embedder, org_id, provider):
    return sync_provider(get_sessionmaker(engine), embedder, org_id=org_id, provider=provider)


def test_new_documents_are_stored(engine, fake_embedder, org) -> None:
    org_id, _ = org
    provider = FakeProvider([_doc("a", title="会計"), _doc("b")])

    report = _sync(engine, fake_embedder, org_id, provider)

    assert (report.listed, report.added, report.ok) == (2, 2, True)
    items = _items(engine, org_id)
    a = items["a"]
    assert (a.title, a.body, a.url) == (
        "会計",
        "a 本文",
        "https://docs.google.com/document/d/a/edit",
    )
    assert a.updated_at == T1
    assert a.author_id is None


def test_unchanged_documents_are_not_fetched_again(engine, fake_embedder, org) -> None:
    org_id, _ = org
    _sync(engine, fake_embedder, org_id, FakeProvider([_doc("a")]))
    again = FakeProvider([_doc("a")])

    report = _sync(engine, fake_embedder, org_id, again)

    assert again.fetched == []
    assert report.unchanged == 1


def test_a_changed_or_renamed_document_is_fetched_and_updated(engine, fake_embedder, org) -> None:
    org_id, _ = org
    _sync(engine, fake_embedder, org_id, FakeProvider([_doc("a"), _doc("b")]))
    changed = FakeProvider(
        [_doc("a", modified=T2), _doc("b", title="新しい題名")], {"a": "書き換えた本文"}
    )

    report = _sync(engine, fake_embedder, org_id, changed)

    assert sorted(changed.fetched) == ["a", "b"]
    assert report.updated == 2
    items = _items(engine, org_id)
    assert (items["a"].body, items["a"].updated_at) == ("書き換えた本文", T2)
    assert items["b"].title == "新しい題名"


def test_a_newer_time_with_the_same_text_is_remembered(engine, fake_embedder, org) -> None:
    org_id, _ = org
    _sync(engine, fake_embedder, org_id, FakeProvider([_doc("a")]))
    touched = FakeProvider([_doc("a", modified=T2)])
    assert _sync(engine, fake_embedder, org_id, touched).unchanged == 1
    assert _items(engine, org_id)["a"].updated_at == T2

    later = FakeProvider([_doc("a", modified=T2)])
    _sync(engine, fake_embedder, org_id, later)
    assert later.fetched == []


def test_documents_no_longer_listed_are_deleted(engine, fake_embedder, org) -> None:
    org_id, other_id = org
    factory = get_sessionmaker(engine)
    with factory() as session:
        for oid, provider in ((org_id, "manual"), (other_id, "google")):
            upsert_source_item(
                session,
                fake_embedder,
                org_id=oid,
                provider=provider,
                external_id="a",
                title=None,
                body="残る 本文",
                url=None,
                author_id=None,
                now=T1,
            )
        session.commit()
    _sync(engine, fake_embedder, org_id, FakeProvider([_doc("a"), _doc("b")]))

    report = _sync(engine, fake_embedder, org_id, FakeProvider([_doc("b")]))

    assert report.deleted == 1
    assert set(_items(engine, org_id)) == {"b"}
    assert set(_items(engine, org_id, "manual")) == {"a"}
    assert set(_items(engine, other_id)) == {"a"}


def test_a_failed_listing_deletes_nothing(engine, fake_embedder, org) -> None:
    org_id, _ = org
    _sync(engine, fake_embedder, org_id, FakeProvider([_doc("a")]))

    report = _sync(engine, fake_embedder, org_id, FakeProvider(fail_listing=True))

    assert report.listing_failed and not report.ok
    assert set(_items(engine, org_id)) == {"a"}


def test_an_incomplete_listing_stores_but_deletes_nothing(engine, fake_embedder, org) -> None:
    org_id, _ = org
    _sync(engine, fake_embedder, org_id, FakeProvider([_doc("a")]))

    report = _sync(engine, fake_embedder, org_id, FakeProvider([_doc("b")], complete=False))

    assert (report.added, report.deleted) == (1, 0)
    assert set(_items(engine, org_id)) == {"a", "b"}


def test_one_failing_document_does_not_stop_the_pass(engine, fake_embedder, org) -> None:
    org_id, _ = org
    _sync(engine, fake_embedder, org_id, FakeProvider([_doc("a")]))
    provider = FakeProvider(
        [_doc("a", modified=T2), _doc("b"), _doc("c")],
        {"a": httpx.ConnectError("down"), "b": SkipDocument("too large")},
    )

    report = _sync(engine, fake_embedder, org_id, provider)

    assert (report.failed, report.skipped, report.added, report.deleted) == (1, 1, 1, 0)
    assert not report.ok
    items = _items(engine, org_id)
    assert set(items) == {"a", "c"}
    assert items["a"].updated_at == T1  # kept as stored, tried again next pass


def test_a_storage_error_is_counted_and_the_pass_goes_on(engine, org) -> None:
    org_id, _ = org

    class BrokenEmbedder:
        def encode(self, texts, *, kind="passage"):
            raise RuntimeError("no model")

    report = _sync(engine, BrokenEmbedder(), org_id, FakeProvider([_doc("a"), _doc("b")]))

    assert report.failed == 2
    assert _items(engine, org_id) == {}


def test_an_emptied_document_is_removed(engine, fake_embedder, org) -> None:
    org_id, _ = org
    _sync(engine, fake_embedder, org_id, FakeProvider([_doc("a")]))

    report = _sync(
        engine, fake_embedder, org_id, FakeProvider([_doc("a", modified=T2)], {"a": " \n"})
    )

    assert report.skipped == 1
    assert _items(engine, org_id) == {}


# ---- loop and CLI ---------------------------------------------------------------- #


def test_run_forever_repeats_until_stopped_and_survives_a_failing_pass() -> None:
    waits: list[float] = []
    calls: list[int] = []

    class Clock(threading.Event):
        def wait(self, timeout=None):
            waits.append(timeout)
            return len(waits) >= 3

    stop = Clock()

    def once() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")

    assert run_forever(once, interval_s=900, stop=stop) == 3
    assert waits == [900, 900, 900]


def test_run_forever_does_nothing_once_stopped() -> None:
    stop = threading.Event()
    stop.set()
    assert run_forever(lambda: None, interval_s=1, stop=stop) == 0


def test_build_providers_is_empty_without_settings() -> None:
    assert build_providers(Settings(google_service_account_file=""), httpx.Client()) == []


def test_build_providers_rejects_a_missing_key(tmp_path) -> None:
    settings = Settings(google_service_account_file=str(tmp_path / "none.json"))
    with pytest.raises(ValueError):
        build_providers(settings, httpx.Client())


def test_main_exits_with_a_message_when_nothing_is_configured(capsys) -> None:
    settings = Settings(discord_guild_id=GUILD, google_service_account_file="")
    assert main(["--once"], settings=settings) == 2
    assert "TEKIJIN_GOOGLE_SERVICE_ACCOUNT_FILE" in capsys.readouterr().err


def test_main_needs_the_organization(capsys) -> None:
    assert main(["--once"], settings=Settings(discord_guild_id=None)) == 2
    assert "TEKIJIN_DISCORD_GUILD_ID" in capsys.readouterr().err


def test_main_reports_a_provider_that_cannot_start(capsys) -> None:
    def refuse(settings, client):
        raise ValueError("could not read the service account key file (KeyError)")

    assert main(["--once"], settings=Settings(discord_guild_id=GUILD), make_providers=refuse) == 2
    assert "could not read" in capsys.readouterr().err


def test_main_once_runs_one_pass_into_the_guild_organization(engine, fake_embedder, org) -> None:
    org_id, _ = org
    provider = FakeProvider([_doc("a")])
    status = main(
        ["--once"],
        settings=Settings(discord_guild_id=GUILD),
        make_providers=lambda s, c: [provider],
        connect=lambda s: (get_sessionmaker(engine), fake_embedder),
    )
    assert status == 0
    assert set(_items(engine, org_id)) == {"a"}

    failing = FakeProvider(fail_listing=True)
    status = main(
        ["--once"],
        settings=Settings(discord_guild_id=GUILD),
        make_providers=lambda s, c: [failing],
        connect=lambda s: (get_sessionmaker(engine), fake_embedder),
    )
    assert status == 1


def test_main_polls_until_stopped(engine, fake_embedder, org) -> None:
    org_id, _ = org
    stop = threading.Event()

    class StopsAfterListing(FakeProvider):
        def list_documents(self) -> Listing:
            stop.set()  # as SIGTERM would: the pass finishes, then the loop ends
            return super().list_documents()

    provider = StopsAfterListing([_doc("a")])
    status = main(
        [],
        settings=Settings(discord_guild_id=GUILD, poll_interval_s=5),
        make_providers=lambda s, c: [provider],
        connect=lambda s: (get_sessionmaker(engine), fake_embedder),
        stop=stop,
    )
    assert status == 0
    assert provider.fetched == ["a"]
    assert set(_items(engine, org_id)) == {"a"}

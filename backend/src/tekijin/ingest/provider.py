"""What a scheduled ingestion source looks like to the poller (#99, R4).

A provider (Google Drive now, Notion next) only has to list the documents it can
see and fetch the text of one of them. Deciding what to fetch, storing it and
deleting what is gone is the poller's job (``ingest/poll.py``), shared by all.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RemoteDocument:
    """One document as the provider lists it.

    ``modified_at`` is naive UTC. A document whose ``modified_at`` and title match
    the stored copy is not fetched again.
    """

    external_id: str
    title: str
    url: str | None
    modified_at: dt.datetime


@dataclass(frozen=True, slots=True)
class Listing:
    """Everything a provider listed.

    ``complete`` is False when the provider says some results may be missing; the
    poller then stores what was listed but deletes nothing.
    """

    documents: Sequence[RemoteDocument]
    complete: bool = True


class SkipDocument(Exception):
    """This document cannot be ingested (too large, say). It is logged and left as
    it is stored, without counting as a failure."""


class Provider(Protocol):
    """A source the poller ingests from. ``name`` is the ``source_items.provider``."""

    name: str

    def list_documents(self) -> Listing:
        """Every document the provider can see. Raises when listing fails, so that
        nothing is deleted on the strength of a partial list."""
        ...

    def fetch_text(self, document: RemoteDocument) -> str:
        """The plain text of one document. Raises :class:`SkipDocument` to skip it."""
        ...

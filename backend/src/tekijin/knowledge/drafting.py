"""Run knowledge drafting off the request path (#113, R6).

Reading a document or a conversation with the LLM takes seconds per passage, far
longer than the bot's flush or a web request should wait. The API therefore hands
the work to :class:`KnowledgeDrafting`, which runs one job at a time on a worker
thread (one at a time also keeps the model from being flooded by a history
replay). Each job opens its own transaction, embeds the drafts it made, and logs
rather than raises.

* :meth:`document_changed` (a document was added or its text changed) runs only
  with ``knowledge_from_documents_enabled``.
* :meth:`conversation_messages` (new Discord messages) and :meth:`reaction` (a check
  mark) run only for an organization with ``knowledge_from_conversations``.

With the ``stub`` LLM backend there is no model to read with, so nothing runs.
Tests pass their own extractors and ``inline=True``.
"""

from __future__ import annotations

import datetime as dt
import logging
import queue
import threading
from collections.abc import Callable, Sequence

from sqlalchemy.orm import Session, sessionmaker

from tekijin.config import Settings
from tekijin.data.db import session_scope
from tekijin.knowledge.conversations import (
    capture_conversation_case,
    closing_candidates,
    conversations_enabled,
)
from tekijin.knowledge.documents import DocumentDraftExtractor, DocumentExtractor, draft_document
from tekijin.knowledge.extract import DraftExtractor, Extractor
from tekijin.knowledge.index import embed_articles
from tekijin.retrieval.embedding import Embedder

logger = logging.getLogger(__name__)

Job = Callable[[], None]


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


class KnowledgeDrafting:
    """Queues drafting jobs and runs them one at a time."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        embedder: Embedder,
        extractor: Extractor | None,
        document_extractor: DocumentExtractor | None,
        documents_enabled: bool,
        inline: bool = False,
        now: Callable[[], dt.datetime] = _utc_now,
    ) -> None:
        self._session_factory = session_factory
        self._embedder = embedder
        self._extractor = extractor
        self._document_extractor = document_extractor
        self._documents_enabled = documents_enabled
        self._inline = inline
        self._now = now
        self._queue: queue.Queue[Job] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        session_factory: sessionmaker[Session],
        embedder: Embedder,
        inline: bool = False,
    ) -> KnowledgeDrafting:
        """The app's runner: real extractors on a real LLM backend, none on the stub."""

        real = settings.llm_backend != "stub"
        return cls(
            session_factory=session_factory,
            embedder=embedder,
            extractor=DraftExtractor(settings=settings) if real else None,
            document_extractor=DocumentDraftExtractor(settings=settings) if real else None,
            documents_enabled=settings.knowledge_from_documents_enabled,
            inline=inline,
        )

    # -- what callers report ------------------------------------------------ #
    def document_changed(self, *, org_id: int, item_id: int) -> bool:
        """A document was added or changed. True when a job was queued."""

        extractor = self._document_extractor
        if not self._documents_enabled or extractor is None:
            return False

        def job() -> None:
            with session_scope(self._session_factory) as session:
                report = draft_document(
                    session, org_id=org_id, item_id=item_id, extractor=extractor, now=self._now()
                )
                self._embed(session, report.article_ids)
            logger.info("document %s: %d knowledge draft(s)", item_id, report.stored)

        return self._submit(job)

    def conversation_messages(self, *, org_id: int, message_ids: Sequence[int]) -> bool:
        """New Discord messages were stored; any may close a question."""

        extractor = self._extractor
        if extractor is None or not message_ids:
            return False
        ids = list(message_ids)

        def job() -> None:
            with session_scope(self._session_factory) as session:
                if not conversations_enabled(session, org_id):
                    return
                candidates = closing_candidates(session, org_id=org_id, message_ids=ids)
            for message_id in candidates:
                self._capture(org_id, message_id, extractor, None)

        return self._submit(job)

    def reaction(self, *, org_id: int, message_id: int, reactor_discord_id: str) -> bool:
        """A check mark was put on a message."""

        extractor = self._extractor
        if extractor is None:
            return False
        return self._submit(
            lambda: self._capture(org_id, message_id, extractor, reactor_discord_id)
        )

    # -- running ------------------------------------------------------------- #
    def _capture(
        self, org_id: int, message_id: int, extractor: Extractor, reactor: str | None
    ) -> None:
        with session_scope(self._session_factory) as session:
            report = capture_conversation_case(
                session,
                org_id=org_id,
                message_id=message_id,
                extractor=extractor,
                now=self._now(),
                reactor_discord_id=reactor,
            )
            if report is not None:
                self._embed(session, report.article_ids)
        if report is not None and report.stored:
            logger.info("message %s: stored a case draft", message_id)

    def _embed(self, session: Session, article_ids: Sequence[int]) -> None:
        if article_ids:
            embed_articles(session, self._embedder, article_ids=list(article_ids))

    def _submit(self, job: Job) -> bool:
        if self._inline:
            self._run(job)
            return True
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._loop, daemon=True, name="knowledge-drafting"
                )
                self._worker.start()
        self._queue.put(job)
        return True

    def _loop(self) -> None:  # pragma: no cover - exercised through join() in tests
        while True:
            job = self._queue.get()
            try:
                self._run(job)
            finally:
                self._queue.task_done()

    @staticmethod
    def _run(job: Job) -> None:
        try:
            job()
        except Exception:  # noqa: BLE001 - background job boundary
            logger.warning("knowledge drafting job failed", exc_info=True)

    def join(self) -> None:
        """Wait until every queued job has run (tests and scripts)."""

        self._queue.join()

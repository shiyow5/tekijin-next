"""Procedure and fact drafts from the organization's documents (#113, R6).

A manual or a hand-over note holds many small pieces of knowledge: where the key
is kept, how to assemble the telescope, who pays for repairs. Each passage of a
document (the same split as its search chunks) is given to the model, which lists
the procedures and facts it states. Each one becomes a draft that cites the
document and quotes the passage, so it waits for an admin before it answers
anything.

When a document changes, :func:`draft_document` runs again. The drafts nobody has
reviewed yet are replaced by the new ones. Verified knowledge was already sent back
to review with a note by :func:`tekijin.data.sources.upsert_source_item`, and is
kept so the admin can compare it with the new drafts.

The extractor takes an injected ``model`` so tests drive it with a fake.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.config import Settings, get_settings
from tekijin.data.knowledge import (
    SourceRef,
    articles_citing,
    create_draft,
    drop_unreviewed_drafts,
)
from tekijin.data.self_intros import DISCORD_PROVIDER
from tekijin.ingest.chunking import split_text
from tekijin.llm.schemas import (
    MAX_DRAFTS_PER_PASSAGE,
    DocumentKnowledgeItem,
    DocumentKnowledgeSchema,
)
from tekijin.models.tables import SourceItem

logger = logging.getLogger(__name__)

#: How much of the passage is kept as the quoted part of the source.
QUOTE_MAX = 1000

_SYSTEM_PROMPT = (
    "あなたは、組織の資料(マニュアル、引継ぎ書、企画書など)の一部を読み、"
    "あとでメンバーの質問に答えられるよう、手順と事実を書き出す担当です。"
    "与えられた本文だけを根拠に、書かれている手順(procedure)と事実・決まりごと(fact)を"
    f"多くて{MAX_DRAFTS_PER_PASSAGE}件まで挙げてください。"
    "1件ごとに、issue にはメンバーが尋ねそうな問いを、resolution にはその答えを書きます。"
    "environment はそれが成り立つ場面、cause は理由で、本文に無ければ null にします。"
    "守ること: (1) 本文に書かれていないことを書かない。"
    "(2) 人の紹介、あいさつ、目次のように問いと答えにならないものは挙げない。"
    "(3) 挙げるものが無ければ items を空にする。"
)


class DocumentExtractor(Protocol):
    def extract_passage(self, *, title: str | None, passage: str) -> list[DocumentKnowledgeItem]:
        """The procedures and facts in one passage. Raises when the model fails."""
        ...


class DocumentDraftExtractor:
    """Asks the model for the procedures and facts in one passage."""

    def __init__(self, *, model: Any | None = None, settings: Settings | None = None) -> None:
        self._model = model
        self._settings = settings or get_settings()

    def _structured(self) -> Any:  # pragma: no cover - builds a network client
        from tekijin.llm.vllm import build_structured_model

        return build_structured_model(DocumentKnowledgeSchema, self._settings)

    @staticmethod
    def prompt(*, title: str | None, passage: str) -> list[tuple[str, str]]:
        head = f"資料の題名: {title}\n\n" if title else ""
        return [("system", _SYSTEM_PROMPT), ("human", f"{head}本文:\n{passage}")]

    def extract_passage(self, *, title: str | None, passage: str) -> list[DocumentKnowledgeItem]:
        if self._model is None:  # pragma: no cover - network
            self._model = self._structured()
        out: DocumentKnowledgeSchema | None = self._model.invoke(
            self.prompt(title=title, passage=passage)
        )
        if out is None:
            raise ValueError("document extraction: the model returned no structured output")
        return list(out.items)


@dataclass(frozen=True, slots=True)
class DocumentDraftReport:
    item_id: int
    passages: int = 0
    stored: int = 0
    # Unreviewed drafts of the earlier text that the new ones replaced.
    replaced: int = 0
    # True when the model failed on a passage; nothing was changed then.
    errored: bool = False
    # True when the document changed while it was read; nothing was written (the
    # change queued its own drafting).
    outdated: bool = False
    article_ids: tuple[int, ...] = field(default_factory=tuple)


def _usable(entry: DocumentKnowledgeItem) -> bool:
    return bool(entry.issue.strip()) and bool(entry.resolution.strip())


def _issue_key(text: str) -> str:
    return "".join(text.split())


def draft_document(
    session: Session,
    *,
    org_id: int,
    item_id: int,
    extractor: DocumentExtractor,
    now: dt.datetime,
) -> DocumentDraftReport:
    """Make (or remake) the drafts of one document; returns what was done.

    A Discord conversation is not a document and gives nothing (cases come from
    solved conversations, ``knowledge/conversations.py``). Every passage is read
    before anything is written, so a model failure leaves the document's drafts
    as they were. The writes run in one savepoint. The caller owns the
    transaction and embeds the new drafts.
    """

    item = session.get(SourceItem, item_id)
    if item is None or item.org_id != org_id or item.provider == DISCORD_PROVIDER:
        return DocumentDraftReport(item_id=item_id)
    digest = item.content_hash
    passages = split_text(item.body or "")
    found: list[tuple[str, DocumentKnowledgeItem]] = []
    for passage in passages:
        try:
            entries = extractor.extract_passage(title=item.title, passage=passage)
        except Exception:
            logger.warning("document extraction failed for item %s", item_id, exc_info=True)
            return DocumentDraftReport(item_id=item_id, passages=len(passages), errored=True)
        found += [(passage, e) for e in entries[:MAX_DRAFTS_PER_PASSAGE] if _usable(e)]

    with session.begin_nested():
        # The document may have changed while the model read it. Locked, so an
        # update waits for this write and then sends these drafts back itself.
        current = session.scalars(
            select(SourceItem)
            .where(SourceItem.id == item_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).first()
        if current is None or current.content_hash != digest:
            return DocumentDraftReport(item_id=item_id, passages=len(passages), outdated=True)
        # Drafts nobody verified, made only from this document: the new ones replace
        # them. One that was verified once stays for the admin, with its note.
        replaced = drop_unreviewed_drafts(session, org_id=org_id, source_item_id=item_id)
        existing = articles_citing(session, org_id=org_id, source_item_id=item_id)
        # An issue already answered from this document is not drafted again, unless
        # the existing article is back in review: the new text may answer it anew.
        kept = {_issue_key(a.issue) for a, _ in existing if a.review_note is None}
        created: list[int] = []
        for passage, entry in found:
            key = _issue_key(entry.issue)
            if key in kept:
                continue
            kept.add(key)
            article = create_draft(
                session,
                org_id=org_id,
                kind=entry.kind,
                issue=entry.issue,
                environment=entry.environment,
                resolution=entry.resolution,
                cause=entry.cause,
                author_id=item.author_id,
                sources=[SourceRef(source_item_id=item_id, quote=passage[:QUOTE_MAX])],
                now=now,
            )
            created.append(article.id)
    return DocumentDraftReport(
        item_id=item_id,
        passages=len(passages),
        stored=len(created),
        replaced=replaced,
        article_ids=tuple(created),
    )


def document_item_ids(
    session: Session, *, org_id: int, item_ids: Sequence[int] | None = None
) -> list[int]:
    """The organization's documents (every item that is not a Discord conversation)."""

    stmt = select(SourceItem.id).where(
        SourceItem.org_id == org_id, SourceItem.provider != DISCORD_PROVIDER
    )
    if item_ids is not None:
        stmt = stmt.where(SourceItem.id.in_(list(item_ids)))
    return list(session.scalars(stmt.order_by(SourceItem.id)))

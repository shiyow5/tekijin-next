"""Read a record and write a knowledge draft with an LLM (#112, R6).

A record is something that may hold knowledge: a resolved hand-off thread, or a
solved Discord conversation (#113, ``knowledge/conversations.py``). Documents give
several drafts per passage and use ``knowledge/documents.py``. The model is asked
for one draft in the KCS shape (issue, environment, resolution, cause) and the
draft is stored with :func:`tekijin.data.knowledge.create_draft`, so it waits for
an admin before it can answer anything. The prompt carries no topic vocabulary
and no company wording.

The extractor takes an injected ``model`` (a ``with_structured_output``-bound
runnable) so tests drive it with a fake and never touch a network.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from sqlalchemy.orm import Session

from tekijin.config import Settings, get_settings
from tekijin.data.knowledge import SourceRef, create_draft
from tekijin.llm.schemas import KnowledgeDraftSchema

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "あなたは、組織のメンバーのやり取りや資料から、あとで別の人の役に立つ知識を書き出す担当です。"
    "与えられた1つの記録だけを根拠に、次の項目を日本語で短く書いてください。"
    "kind は case(困りごとと解決)、procedure(手順)、fact(事実・決まりごと)、"
    "decision(決めたことと理由)のどれか。"
    "issue は困りごとや問いを、質問する人が使いそうな言葉で書く。"
    "resolution は答えや解決のしかた。environment はそれが成り立つ場面、cause は原因で、"
    "どちらも記録に無ければ null。"
    "守ること: (1) 記録に書かれていないことを書かない。"
    "答えが書かれていなければ resolution は null。"
    "(2) あいさつ、連絡だけのもの、雑談、問いも答えも無いものは extractable を false にする。"
)


@dataclass(frozen=True, slots=True)
class ExtractionSource:
    """One record to read: its organization, its text, and what to cite.

    ``kind`` fixes the draft's kind whatever the model says (a solved conversation
    is always a ``case``, #113); ``None`` keeps the model's.
    """

    org_id: int
    text: str
    sources: tuple[SourceRef, ...]
    author_id: int | None = None
    kind: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    seen: int = 0
    stored: int = 0
    skipped: int = 0
    errored: int = 0
    article_ids: tuple[int, ...] = field(default_factory=tuple)
    # Records whose source changed while the model read them; nothing was stored.
    outdated: int = 0


class Extractor(Protocol):
    def extract(self, source: ExtractionSource) -> KnowledgeDraftSchema | None: ...


class DraftExtractor:
    """Asks the model for one draft. ``None`` means the record holds nothing to keep."""

    def __init__(self, *, model: Any | None = None, settings: Settings | None = None) -> None:
        self._model = model
        self._settings = settings or get_settings()

    def _structured(self) -> Any:  # pragma: no cover - builds a network client
        from tekijin.llm.vllm import build_structured_model

        return build_structured_model(KnowledgeDraftSchema, self._settings)

    @staticmethod
    def prompt(source: ExtractionSource) -> list[tuple[str, str]]:
        return [("system", _SYSTEM_PROMPT), ("human", source.text)]

    def extract(self, source: ExtractionSource) -> KnowledgeDraftSchema | None:
        """The draft, or ``None`` when the model says there is nothing to keep.

        An empty structured output (no tool call) is raised as ``ValueError``: it is
        what a refusal or an injected prompt tends to produce, and is not the same as
        "nothing to keep". :func:`extract_and_store` isolates it per record.
        """

        model = self._model if self._model is not None else self._structured()
        out: KnowledgeDraftSchema | None = model.invoke(self.prompt(source))
        if out is None:
            raise ValueError("knowledge extraction: the model returned no structured output")
        if not out.extractable:
            return None
        return out


def extract_and_store(
    session: Session,
    sources: Sequence[ExtractionSource],
    extractor: Extractor,
    *,
    now: dt.datetime,
    still_current: Callable[[ExtractionSource], bool] | None = None,
) -> ExtractionReport:
    """Extract each record and store a draft for it.

    A record that fails (the model or the write raised) is logged and counted as
    ``errored``, and its partial write is rolled back to a savepoint, so one bad
    record does not lose the others. ``still_current`` is asked after the model
    answered and before the draft is written: ``False`` means the record changed
    meanwhile, and nothing is stored for it (``outdated``). The caller owns the
    transaction.
    """

    seen = stored = skipped = errored = outdated = 0
    created: list[int] = []
    for source in sources:
        seen += 1
        try:
            draft = extractor.extract(source)
            if draft is None:
                skipped += 1
                continue
            if still_current is not None and not still_current(source):
                outdated += 1
                continue
            with session.begin_nested():
                article = create_draft(
                    session,
                    org_id=source.org_id,
                    kind=source.kind or draft.kind,
                    issue=draft.issue,
                    environment=draft.environment,
                    resolution=draft.resolution,
                    cause=draft.cause,
                    author_id=source.author_id,
                    sources=source.sources,
                    now=now,
                )
            created.append(article.id)
            stored += 1
        except Exception:
            errored += 1
            logger.warning("knowledge extraction failed for a record", exc_info=True)
    return ExtractionReport(
        seen=seen,
        stored=stored,
        skipped=skipped,
        errored=errored,
        article_ids=tuple(created),
        outdated=outdated,
    )

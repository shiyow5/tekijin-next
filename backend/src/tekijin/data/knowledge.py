"""The organization's knowledge in the KCS shape: drafts, review, search (#112, R6).

Knowledge is written as drafts by whatever finds it (a resolved Slack thread now;
documents, Discord conversations and answered requests in #113 / #114), checked by
an admin of the organization, and used to answer questions once verified.

* :func:`create_draft` is the one way in. A draft without a resolution is ``wip``.
* :func:`update_article`, :func:`verify_article`, :func:`send_back_article` and
  :func:`retire_article` are the review actions.
* :func:`search_verified` finds verified knowledge near a question, inside one
  organization and limited to what the reader may see; :func:`record_use` counts an
  answer it gave. :func:`matching_drafts` / :func:`record_demand` count the
  questions a draft would have answered, which orders the review queue.

Visibility: an article is shown to a reader only when the reader may see every one
of its sources (a private channel's message hides what was made from it, and so
does a message hidden from the web). A source whose item was deleted is lost and
hides the article from everyone but the admin. An article with no sources was
written by hand and is visible to the whole organization. Callers own the
transaction.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, and_, exists, func, not_, or_, select, update
from sqlalchemy.orm import Session

from tekijin.data.visibility import EVERYTHING, SourceScope, stream_visible, visible
from tekijin.models.tables import (
    ChannelMessage,
    KnowledgeArticle,
    KnowledgeSource,
    Organization,
    Question,
    SourceItem,
)

KINDS: tuple[str, ...] = ("case", "procedure", "fact", "decision")
STATES: tuple[str, ...] = ("wip", "draft", "verified", "retired")
#: What the admin's review queue lists.
REVIEW_STATES: tuple[str, ...] = ("wip", "draft")
VERIFIED = "verified"

#: The fields a review may edit.
EDITABLE_FIELDS: frozenset[str] = frozenset({"kind", "issue", "environment", "resolution", "cause"})

_DISCORD_MESSAGE_URL = "https://discord.com/channels/{guild}/{channel}/{message}"


class ArticleNotFound(LookupError):
    """No article with that id in that organization."""


class ArticleStateError(ValueError):
    """The action does not apply to the article's current state."""


@dataclass(frozen=True, slots=True)
class SourceRef:
    """One thing an article was made from, given to :func:`create_draft`.

    Exactly one of the three ids is set. ``quote`` is the part relied on.
    """

    source_item_id: int | None = None
    channel_message_id: int | None = None
    question_id: str | None = None
    quote: str | None = None

    def __post_init__(self) -> None:
        named = [
            v
            for v in (self.source_item_id, self.channel_message_id, self.question_id)
            if v is not None
        ]
        if len(named) != 1:
            raise ValueError("a knowledge source names exactly one item, message or question")


@dataclass(frozen=True, slots=True)
class ArticleDTO:
    id: int
    org_id: int
    kind: str
    issue: str
    environment: str | None
    resolution: str | None
    cause: str | None
    state: str
    reuse_count: int
    demand_count: int
    last_used_at: dt.datetime | None
    superseded_by: int | None
    author_id: int | None
    verifier_id: int | None
    created_at: dt.datetime
    updated_at: dt.datetime
    verified_at: dt.datetime | None

    @classmethod
    def from_row(cls, row: KnowledgeArticle) -> ArticleDTO:
        return cls(
            id=row.id,
            org_id=row.org_id,
            kind=row.kind,
            issue=row.issue,
            environment=row.environment,
            resolution=row.resolution,
            cause=row.cause,
            state=row.state,
            reuse_count=row.reuse_count or 0,
            demand_count=row.demand_count or 0,
            last_used_at=row.last_used_at,
            superseded_by=row.superseded_by,
            author_id=row.author_id,
            verifier_id=row.verifier_id,
            created_at=row.created_at,
            updated_at=row.updated_at,
            verified_at=row.verified_at,
        )


@dataclass(frozen=True, slots=True)
class ArticleSource:
    """A source of an article, with what a reader needs to find it."""

    id: int
    source_item_id: int | None
    channel_message_id: int | None
    question_id: str | None
    quote: str | None
    title: str | None = None
    url: str | None = None
    channel_name: str | None = None


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown knowledge kind: {kind!r}")


def _check_sources(session: Session, org_id: int, sources: Sequence[SourceRef]) -> None:
    """Refuse a source that does not exist or belongs to another organization."""

    item_ids = {s.source_item_id for s in sources if s.source_item_id is not None}
    message_ids = {s.channel_message_id for s in sources if s.channel_message_id is not None}
    question_ids = {s.question_id for s in sources if s.question_id is not None}
    if item_ids:
        found = set(
            session.scalars(
                select(SourceItem.id).where(
                    SourceItem.id.in_(item_ids), SourceItem.org_id == org_id
                )
            )
        )
        if found != item_ids:
            raise ValueError("a source item is missing or belongs to another organization")
    if message_ids:
        found = set(
            session.scalars(
                select(ChannelMessage.id).where(
                    ChannelMessage.id.in_(message_ids), ChannelMessage.org_id == org_id
                )
            )
        )
        if found != message_ids:
            raise ValueError("a message is missing or belongs to another organization")
    if question_ids:
        found_q = set(session.scalars(select(Question.id).where(Question.id.in_(question_ids))))
        if found_q != question_ids:
            raise ValueError("a question source is missing")


def create_draft(
    session: Session,
    *,
    org_id: int,
    kind: str,
    issue: str,
    resolution: str | None = None,
    environment: str | None = None,
    cause: str | None = None,
    author_id: int | None = None,
    sources: Sequence[SourceRef] = (),
    now: dt.datetime,
) -> ArticleDTO:
    """Add one draft and its sources; returns it.

    The state is ``draft`` when ``resolution`` has text and ``wip`` otherwise. The
    embedding is left empty; :func:`tekijin.knowledge.index.embed_articles` fills it.
    Raises ``ValueError`` for an unknown kind, a blank issue, or a source of another
    organization.
    """

    _check_kind(kind)
    clean_issue = _clean(issue)
    if clean_issue is None:
        raise ValueError("a knowledge article needs an issue")
    clean_resolution = _clean(resolution)
    _check_sources(session, org_id, sources)
    row = KnowledgeArticle(
        org_id=org_id,
        kind=kind,
        issue=clean_issue,
        environment=_clean(environment),
        resolution=clean_resolution,
        cause=_clean(cause),
        state="draft" if clean_resolution else "wip",
        reuse_count=0,
        demand_count=0,
        author_id=author_id,
        created_at=now,
        updated_at=now,
    )
    session.add(row)
    session.flush()
    session.add_all(
        KnowledgeSource(
            article_id=row.id,
            source_item_id=s.source_item_id,
            channel_message_id=s.channel_message_id,
            question_id=s.question_id,
            quote=_clean(s.quote),
        )
        for s in sources
    )
    session.flush()
    return ArticleDTO.from_row(row)


def _load(session: Session, article_id: int, org_id: int) -> KnowledgeArticle:
    row = session.get(KnowledgeArticle, article_id)
    if row is None or row.org_id != org_id:
        raise ArticleNotFound(article_id)
    return row


def update_article(
    session: Session,
    article_id: int,
    *,
    org_id: int,
    changes: Mapping[str, str | None],
    now: dt.datetime,
) -> ArticleDTO:
    """Edit the KCS fields of one article.

    Only :data:`EDITABLE_FIELDS` may change. Emptying the resolution turns the
    article into ``wip`` (and un-verifies it); writing one into a ``wip`` article
    makes it a ``draft``. A verified article whose resolution is kept stays verified.
    The embedding is cleared so it is rebuilt from the new text.
    """

    unknown = set(changes) - EDITABLE_FIELDS
    if unknown:
        raise ValueError(f"cannot edit: {sorted(unknown)}")
    row = _load(session, article_id, org_id)
    if "kind" in changes:
        kind = changes["kind"] or ""
        _check_kind(kind)
        row.kind = kind
    if "issue" in changes:
        issue = _clean(changes["issue"])
        if issue is None:
            raise ValueError("a knowledge article needs an issue")
        row.issue = issue
    for field in ("environment", "cause", "resolution"):
        if field in changes:
            setattr(row, field, _clean(changes[field]))
    if row.resolution is None:
        if row.state != "retired":
            row.state = "wip"
        row.verifier_id = None
        row.verified_at = None
    elif row.state == "wip":
        row.state = "draft"
    row.updated_at = now
    row.embedding = None
    session.flush()
    return ArticleDTO.from_row(row)


def verify_article(
    session: Session,
    article_id: int,
    *,
    org_id: int,
    verifier_id: int | None,
    now: dt.datetime,
) -> ArticleDTO:
    """Mark an article verified, whatever its state, as long as it has an answer.

    A ``wip`` article that was sent back keeps its answer, so it can be verified
    again without an edit. ``verifier_id`` is ``None`` for the settings admin, who
    is not a member."""

    row = _load(session, article_id, org_id)
    if row.resolution is None:
        raise ArticleStateError("only an article with a resolution can be verified")
    row.state = VERIFIED
    row.verifier_id = verifier_id
    row.verified_at = now
    row.superseded_by = None
    row.updated_at = now
    session.flush()
    return ArticleDTO.from_row(row)


def send_back_article(
    session: Session, article_id: int, *, org_id: int, now: dt.datetime
) -> ArticleDTO:
    """Return an article for more work (``wip``). Its text is kept."""

    row = _load(session, article_id, org_id)
    if row.state == "retired":
        raise ArticleStateError("a retired article cannot be sent back")
    row.state = "wip"
    row.verifier_id = None
    row.verified_at = None
    row.updated_at = now
    session.flush()
    return ArticleDTO.from_row(row)


def retire_article(
    session: Session,
    article_id: int,
    *,
    org_id: int,
    now: dt.datetime,
    superseded_by: int | None = None,
) -> ArticleDTO:
    """Stop using an article, optionally naming the one that replaces it.

    Any article can be retired, including a ``wip`` one with no answer."""

    row = _load(session, article_id, org_id)
    if superseded_by is not None:
        if superseded_by == article_id:
            raise ValueError("an article cannot replace itself")
        _load(session, superseded_by, org_id)
    row.state = "retired"
    row.superseded_by = superseded_by
    row.updated_at = now
    session.flush()
    return ArticleDTO.from_row(row)


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def article_visible(org_id: int, scope: SourceScope) -> ColumnElement[bool]:
    """A condition on ``knowledge_articles``: every source may be seen by ``scope``."""

    if scope.everything:
        return and_(KnowledgeArticle.org_id == org_id)
    readable_items = select(SourceItem.id).where(
        SourceItem.org_id == org_id, visible(org_id, scope)
    )
    readable_messages = select(ChannelMessage.id).where(
        ChannelMessage.org_id == org_id,
        ChannelMessage.hidden_at.is_(None),
        stream_visible(org_id, scope, ChannelMessage.channel_id, ChannelMessage.thread_id),
    )
    # Each branch tests its own column for NULL first: ``NULL IN (...)`` is NULL,
    # and NOT NULL would let a hidden source slip through the check below.
    readable = or_(
        KnowledgeSource.question_id.isnot(None),
        and_(
            KnowledgeSource.source_item_id.isnot(None),
            KnowledgeSource.source_item_id.in_(readable_items),
        ),
        and_(
            KnowledgeSource.channel_message_id.isnot(None),
            KnowledgeSource.channel_message_id.in_(readable_messages),
        ),
    )
    hidden_source = exists().where(
        KnowledgeSource.article_id == KnowledgeArticle.id, not_(readable)
    )
    return and_(KnowledgeArticle.org_id == org_id, not_(hidden_source))


def _check_states(states: Iterable[str]) -> list[str]:
    chosen = list(states)
    for state in chosen:
        if state not in STATES:
            raise ValueError(f"unknown knowledge state: {state!r}")
    return chosen


def get_article(
    session: Session,
    article_id: int,
    *,
    org_id: int,
    scope: SourceScope,
    states: Iterable[str] | None = None,
) -> ArticleDTO | None:
    """One article of ``org_id`` that ``scope`` may see, or ``None``."""

    stmt = select(KnowledgeArticle).where(
        KnowledgeArticle.id == article_id, article_visible(org_id, scope)
    )
    if states is not None:
        stmt = stmt.where(KnowledgeArticle.state.in_(_check_states(states)))
    row = session.scalars(stmt).first()
    return ArticleDTO.from_row(row) if row is not None else None


def list_articles(
    session: Session,
    *,
    org_id: int,
    states: Iterable[str],
    scope: SourceScope,
    q: str | None = None,
    offset: int = 0,
    limit: int = 50,
) -> tuple[list[ArticleDTO], int]:
    """A page of the organization's articles in ``states``, and how many match.

    Ordered most wanted first: demand (questions a draft would have answered),
    then use, then the newest change.
    """

    chosen = _check_states(states)
    conditions: list[Any] = [article_visible(org_id, scope), KnowledgeArticle.state.in_(chosen)]
    if q:
        like = f"%{q}%"
        conditions.append(
            or_(
                KnowledgeArticle.issue.ilike(like),
                KnowledgeArticle.environment.ilike(like),
                KnowledgeArticle.resolution.ilike(like),
                KnowledgeArticle.cause.ilike(like),
            )
        )
    total = session.scalar(select(func.count()).select_from(KnowledgeArticle).where(*conditions))
    rows = session.scalars(
        select(KnowledgeArticle)
        .where(*conditions)
        .order_by(
            KnowledgeArticle.demand_count.desc(),
            KnowledgeArticle.reuse_count.desc(),
            KnowledgeArticle.updated_at.desc(),
            KnowledgeArticle.id.desc(),
        )
        .offset(offset)
        .limit(limit)
    )
    return [ArticleDTO.from_row(r) for r in rows], int(total or 0)


def article_sources(
    session: Session, article_ids: Sequence[int], *, org_id: int, scope: SourceScope
) -> dict[int, list[ArticleSource]]:
    """The sources of each article, keyed by article id, in the order they were added.

    Only articles of ``org_id`` that ``scope`` may see are included.
    """

    if not article_ids:
        return {}
    allowed = set(
        session.scalars(
            select(KnowledgeArticle.id).where(
                KnowledgeArticle.id.in_(list(article_ids)), article_visible(org_id, scope)
            )
        )
    )
    if not allowed:
        return {}
    guild = session.scalar(select(Organization.discord_guild_id).where(Organization.id == org_id))
    rows = session.execute(
        select(KnowledgeSource, SourceItem, ChannelMessage)
        .outerjoin(SourceItem, SourceItem.id == KnowledgeSource.source_item_id)
        .outerjoin(ChannelMessage, ChannelMessage.id == KnowledgeSource.channel_message_id)
        .where(KnowledgeSource.article_id.in_(allowed))
        .order_by(KnowledgeSource.article_id, KnowledgeSource.id)
    ).all()
    found: dict[int, list[ArticleSource]] = {}
    for source, item, message in rows:
        found.setdefault(source.article_id, []).append(_describe(source, item, message, guild))
    return found


def _describe(
    source: KnowledgeSource,
    item: SourceItem | None,
    message: ChannelMessage | None,
    guild: str | None,
) -> ArticleSource:
    title = url = channel_name = None
    if item is not None:
        title, url = item.title, item.url
    elif message is not None:
        channel_name = message.thread_name or message.channel_name
        if guild:
            url = _DISCORD_MESSAGE_URL.format(
                guild=guild,
                channel=message.thread_id or message.channel_id,
                message=message.message_id,
            )
    return ArticleSource(
        id=source.id,
        source_item_id=source.source_item_id,
        channel_message_id=source.channel_message_id,
        question_id=source.question_id,
        quote=source.quote,
        title=title,
        url=url,
        channel_name=channel_name,
    )


def articles_from_question(session: Session, question_id: str) -> list[ArticleDTO]:
    """Every article (any state) made from a question, oldest first."""

    rows = session.scalars(
        select(KnowledgeArticle)
        .join(KnowledgeSource, KnowledgeSource.article_id == KnowledgeArticle.id)
        .where(KnowledgeSource.question_id == question_id)
        .order_by(KnowledgeArticle.id)
    )
    return [ArticleDTO.from_row(r) for r in rows]


# --------------------------------------------------------------------------- #
# search and use
# --------------------------------------------------------------------------- #
def _nearest(
    session: Session,
    *,
    org_id: int,
    query_vec: Sequence[float],
    state: str,
    scope: SourceScope,
    top_k: int,
    min_similarity: float,
) -> list[tuple[ArticleDTO, float]]:
    column: Any = KnowledgeArticle.embedding
    distance = column.cosine_distance(list(query_vec)).label("distance")
    rows = session.execute(
        select(KnowledgeArticle, distance)
        .where(
            article_visible(org_id, scope),
            KnowledgeArticle.state == state,
            KnowledgeArticle.embedding.isnot(None),
        )
        .order_by(distance, KnowledgeArticle.id)
        .limit(top_k)
    ).all()
    hits = [(ArticleDTO.from_row(row), 1.0 - float(dist)) for row, dist in rows]
    return [(a, sim) for a, sim in hits if sim >= min_similarity]


def search_verified(
    session: Session,
    *,
    org_id: int,
    query_vec: Sequence[float],
    scope: SourceScope,
    top_k: int = 5,
    min_similarity: float = 0.0,
) -> list[tuple[ArticleDTO, float]]:
    """Verified knowledge of ``org_id`` nearest to the question, most similar first.

    Similarity is ``1 - cosine distance``; hits below ``min_similarity`` are dropped.
    Only articles ``scope`` may see are returned.
    """

    if not query_vec:
        return []
    return _nearest(
        session,
        org_id=org_id,
        query_vec=query_vec,
        state=VERIFIED,
        scope=scope,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def matching_drafts(
    session: Session,
    *,
    org_id: int,
    query_vec: Sequence[float],
    min_similarity: float,
    limit: int = 5,
) -> list[int]:
    """Ids of drafts that would have answered the question (for the review order)."""

    if not query_vec:
        return []
    hits = _nearest(
        session,
        org_id=org_id,
        query_vec=query_vec,
        state="draft",
        scope=EVERYTHING,
        top_k=limit,
        min_similarity=min_similarity,
    )
    return [a.id for a, _ in hits]


def record_use(session: Session, article_ids: Sequence[int], *, now: dt.datetime) -> None:
    """Count one answer given by each article and stamp when."""

    if not article_ids:
        return
    session.execute(
        update(KnowledgeArticle)
        .where(KnowledgeArticle.id.in_(list(article_ids)))
        .values(reuse_count=KnowledgeArticle.reuse_count + 1, last_used_at=now)
    )


def record_demand(session: Session, article_ids: Sequence[int]) -> None:
    """Count one question each draft would have answered."""

    if not article_ids:
        return
    session.execute(
        update(KnowledgeArticle)
        .where(KnowledgeArticle.id.in_(list(article_ids)))
        .values(demand_count=KnowledgeArticle.demand_count + 1)
    )

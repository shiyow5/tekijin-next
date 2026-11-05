"""Members' self-introductions, stored and embedded as evidence (#106, R5).

A self-introduction is what a member says about themselves, so it is the first
evidence of who knows what in an organization that has just started (design §7).
They come from three places, each kept as ``self_intros`` rows:

* ``slides``: an ingested document whose title contains 「自己紹介」 is split into
  sections, and a section that names exactly one member becomes theirs. This runs
  on every upsert of such a document (``data/sources.py``), so ``/tekijin add``,
  the web, Google/Notion polling and the circle seed all get it. Deleting the
  document deletes its rows (foreign key cascade).
* ``join``: the text typed with ``/tekijin join`` or ``/tekijin intro``.
* ``discord``: posts in a channel named 「自己紹介」 or ``introductions``. These are
  declarations the member wrote for the organization, so unlike other messages
  they do not depend on ``expertise_from_messages`` (ADR-0012).

``/tekijin optout`` deletes all of a member's rows, and none is written for them
while they stay opted out. :func:`search_self_intros` is what ranking people by
these introductions reads (#107).
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy import ColumnElement, and_, delete, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tekijin.config import get_settings
from tekijin.data.visibility import SourceScope, stream_visible
from tekijin.models.tables import ChannelMessage, Employee, SelfIntro, SourceItem
from tekijin.retrieval.bm25_cache import cached_bm25_index
from tekijin.retrieval.embedding import PASSAGE, Embedder
from tekijin.retrieval.fusion import rrf

SLIDES = "slides"
JOIN = "join"
DISCORD = "discord"

INTRO_TITLE = "自己紹介"
# ``/tekijin join`` / ``/tekijin intro``.
MAX_INTRO_LENGTH = 1000
# One section of a document; longer ones are cut so a whole document pasted under
# one name does not become one person's introduction.
_MAX_SECTION_CHARS = 2000
# A one-letter name would match almost any section.
_MIN_NAME_CHARS = 2

# Each channel's depth before fusion.
_POOL = 50

# Discord's conversation provider, whose titles name the channel; a conversation
# in the introduction channel is not a document of introductions.
_DISCORD_PROVIDER = "discord"

_SLIDE_HEADING = re.compile(r"^\s*(?:スライド|slide)\s*[0-9０-９]+", re.IGNORECASE)
_MD_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+\S")


@dataclass(frozen=True, slots=True)
class IntroHit:
    """One self-introduction found by :func:`search_self_intros`."""

    member_id: int
    intro_id: int
    origin: str
    text: str
    # Fused RRF score: the ranking, with no absolute meaning.
    score: float
    # Cosine similarity of this introduction to the query, for absolute thresholds.
    cosine: float


# --------------------------------------------------------------------------- #
# Splitting a document into members' sections
# --------------------------------------------------------------------------- #
def _squash(text: str) -> str:
    return "".join(text.split())


def _heading_level(line: str) -> int | None:
    """0 for 「スライド N」, 1-6 for a Markdown heading, ``None`` otherwise."""

    if _SLIDE_HEADING.match(line):
        return 0
    found = _MD_HEADING.match(line)
    return len(found.group(1)) if found else None


def _names_in(text: str, names: Mapping[int, str]) -> set[int]:
    """Members named in ``text``. A name inside a longer matching name (「佐藤」 in
    「佐藤花子」) counts as the longer one."""

    squashed = _squash(text)
    found = {
        member_id: name
        for member_id, name in names.items()
        if len(name) >= _MIN_NAME_CHARS and name in squashed
    }
    return {
        member_id
        for member_id, name in found.items()
        if not any(name != other and name in other for other in found.values())
    }


def split_sections(body: str, names: Mapping[int, str]) -> list[str]:
    """``body`` cut into the parts that may each introduce one person.

    Headings decide the cut: 「スライド N」 lines and Markdown headings. Of the
    heading levels present, the one whose headings name the most members is used,
    so a document title above the people, or 「得意なこと」 below each person, does
    not cut a person's section apart. Without such headings, blank lines cut it.
    ``names`` maps member ids to their names with whitespace removed.
    """

    lines = body.splitlines()
    levels = [_heading_level(line) for line in lines]
    level = _best_level(lines, levels, names)
    if level is None:
        blocks = re.split(r"\n\s*\n", body)
        return [block.strip() for block in blocks if block.strip()]
    sections: list[list[str]] = []
    for line, line_level in zip(lines, levels, strict=True):
        if line_level == level or not sections:
            sections.append([])
        sections[-1].append(line)
    return [text for text in ("\n".join(s).strip() for s in sections) if text]


def _best_level(
    lines: Sequence[str], levels: Sequence[int | None], names: Mapping[int, str]
) -> int | None:
    counts: dict[int, int] = {}
    for line, level in zip(lines, levels, strict=True):
        if level is not None and _names_in(line, names):
            counts[level] = counts.get(level, 0) + 1
    if not counts:
        return None
    return min(counts, key=lambda level: (-counts[level], level))


def section_member(section: str, names: Mapping[int, str]) -> int | None:
    """The one member ``section`` introduces, or ``None`` when no member or more
    than one is named. The first line (the slide's or heading's title) is looked
    at first; only when it names nobody is the whole section searched."""

    head = section.split("\n", 1)[0]
    found = _names_in(head, names) or _names_in(section, names)
    return next(iter(found)) if len(found) == 1 else None


def intro_sections(body: str, names: Mapping[int, str]) -> dict[int, str]:
    """Member id -> their introduction in ``body``. Sections naming nobody, or
    more than one member, are dropped; two sections of one member are joined."""

    # Two members with the same name are both found, so neither gets the section.
    squashed = {member_id: _squash(name) for member_id, name in names.items()}
    by_member: dict[int, list[str]] = {}
    for section in split_sections(body, squashed):
        member_id = section_member(section, squashed)
        if member_id is None:
            continue
        by_member.setdefault(member_id, []).append(section)
    return {
        member_id: "\n\n".join(parts)[:_MAX_SECTION_CHARS] for member_id, parts in by_member.items()
    }


def is_intro_document(provider: str, title: str | None) -> bool:
    return provider != _DISCORD_PROVIDER and INTRO_TITLE in (title or "")


def is_intro_channel(name: str | None) -> bool:
    """「自己紹介」, 「自己紹介-2026」, ``introductions``, ``#Introductions`` …"""

    folded = (name or "").casefold()
    return INTRO_TITLE in folded or "introductions" in folded


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def _org_members(session: Session, org_id: int) -> list[Employee]:
    return list(session.scalars(select(Employee).where(Employee.org_id == org_id)).all())


def sync_document_intros(
    session: Session,
    embedder: Embedder,
    *,
    org_id: int,
    item_id: int,
    provider: str,
    title: str | None,
    body: str,
    now: dt.datetime,
) -> int:
    """Make the ``slides`` rows of one source item match its text. Returns how
    many rows it has now.

    Rows whose text did not change keep their embedding, so re-running it on an
    unchanged document embeds nothing. A member who has opted out gets no row.
    The caller commits.
    """

    if not is_intro_document(provider, title):
        session.execute(delete(SelfIntro).where(SelfIntro.source_item_id == item_id))
        return 0
    # Serializes with an upsert of the same item, which holds this lock while it
    # writes, and with another sync of it.
    session.execute(select(SourceItem.id).where(SourceItem.id == item_id).with_for_update())
    members = _org_members(session, org_id)
    sections = intro_sections(body, {m.id: m.name for m in members})
    opted_out = {m.id for m in members if m.opted_out_at is not None}
    wanted = {mid: text for mid, text in sections.items() if mid not in opted_out}
    stored = {
        row.member_id: row
        for row in session.scalars(
            select(SelfIntro).where(SelfIntro.source_item_id == item_id)
        ).all()
    }
    for member_id, row in stored.items():
        if member_id not in wanted:
            session.delete(row)
    changed = [mid for mid, text in wanted.items() if mid not in stored or stored[mid].text != text]
    vectors = embedder.encode([wanted[mid] for mid in changed], kind=PASSAGE) if changed else []
    for member_id, vector in zip(changed, vectors, strict=True):
        row = stored.get(member_id) or SelfIntro(
            org_id=org_id, member_id=member_id, origin=SLIDES, source_item_id=item_id
        )
        row.text, row.embedding, row.updated_at = wanted[member_id], vector, now
        session.add(row)
    session.flush()
    return len(wanted)


def set_join_intro(
    session: Session, embedder: Embedder, *, member: Employee, text: str, now: dt.datetime
) -> bool:
    """Set or replace the member's ``join`` row. False (nothing written) when the
    member has opted out, or belongs to no organization. The caller commits."""

    if member.opted_out_at is not None or member.org_id is None:
        return False
    text = text.strip()[:MAX_INTRO_LENGTH]
    if not text:
        return False
    row = _join_row(session, member.id)
    if row is not None and row.text == text:
        return True
    vector = embedder.encode([text], kind=PASSAGE)[0]
    if row is None:
        try:
            with session.begin_nested():
                row = SelfIntro(org_id=member.org_id, member_id=member.id, origin=JOIN)
                row.text, row.embedding, row.updated_at = text, vector, now
                session.add(row)
            return True
        except IntegrityError:
            row = _join_row(session, member.id)
            assert row is not None  # inserted by a concurrent request
    row.text, row.embedding, row.updated_at = text, vector, now
    session.flush()
    return True


def _join_row(session: Session, member_id: int) -> SelfIntro | None:
    return session.scalars(
        select(SelfIntro).where(SelfIntro.member_id == member_id, SelfIntro.origin == JOIN)
    ).first()


def _is_intro_message(row: ChannelMessage) -> bool:
    # A reply inside a thread is not an introduction; a forum post's first message
    # (whose id is the thread's) is.
    return (
        is_intro_channel(row.channel_name)
        and (row.thread_id is None or row.thread_id == row.message_id)
        and row.hidden_at is None
        and row.author_id is not None
        and bool(row.content.strip())
    )


def sync_message_intros(
    session: Session,
    embedder: Embedder,
    *,
    org_id: int,
    rows: Iterable[ChannelMessage],
    now: dt.datetime,
) -> None:
    """Make the ``discord`` rows of these stored messages match them: a post in
    the introduction channel is its author's introduction, anything else (or a
    hidden post) has none. The caller commits."""

    messages = list(rows)
    if not messages:
        return
    stored = {
        row.message_id: row
        for row in session.scalars(
            select(SelfIntro).where(
                SelfIntro.org_id == org_id,
                SelfIntro.message_id.in_([m.message_id for m in messages]),
            )
        ).all()
    }
    authors = {m.author_id for m in messages if m.author_id is not None}
    opted_out = set(
        session.scalars(
            select(Employee.id).where(Employee.id.in_(authors), Employee.opted_out_at.isnot(None))
        ).all()
    )
    wanted = [m for m in messages if _is_intro_message(m) and m.author_id not in opted_out]
    keep = {m.message_id for m in wanted}
    for message_id, row in stored.items():
        if message_id not in keep:
            session.delete(row)
    changed = [m for m in wanted if m.message_id not in stored or _differs(stored[m.message_id], m)]
    vectors = embedder.encode([m.content.strip() for m in changed], kind=PASSAGE) if changed else []
    for message, vector in zip(changed, vectors, strict=True):
        row = stored.get(message.message_id) or SelfIntro(
            org_id=org_id, origin=DISCORD, message_id=message.message_id
        )
        assert message.author_id is not None  # checked by _is_intro_message
        row.member_id = message.author_id
        row.discord_channel_id, row.discord_thread_id = message.channel_id, message.thread_id
        row.text, row.embedding, row.updated_at = message.content.strip(), vector, now
        session.add(row)
    session.flush()


def _differs(row: SelfIntro, message: ChannelMessage) -> bool:
    return (row.member_id, row.discord_channel_id, row.discord_thread_id, row.text) != (
        message.author_id,
        message.channel_id,
        message.thread_id,
        message.content.strip(),
    )


def delete_message_intros(session: Session, *, org_id: int, message_ids: Iterable[int]) -> None:
    ids = list(message_ids)
    if ids:
        session.execute(
            delete(SelfIntro).where(SelfIntro.org_id == org_id, SelfIntro.message_id.in_(ids))
        )


def delete_member_intros(session: Session, *, member_id: int) -> int:
    """``/tekijin optout``: forget every introduction of the member. Returns how
    many were deleted. Opting back in does not restore them."""

    result = session.execute(delete(SelfIntro).where(SelfIntro.member_id == member_id))
    return int(result.rowcount or 0)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Search (#107 ranks people with this)
# --------------------------------------------------------------------------- #
def _visible_intro(org_id: int, scope: SourceScope) -> ColumnElement[bool]:
    return or_(
        SelfIntro.origin != DISCORD,
        stream_visible(org_id, scope, SelfIntro.discord_channel_id, SelfIntro.discord_thread_id),
    )


def search_self_intros(
    session: Session,
    *,
    org_id: int,
    query: str,
    query_vector: Sequence[float],
    scope: SourceScope,
    top_k: int = 20,
) -> list[IntroHit]:
    """The introductions ``scope`` may see that best match the query, best first.

    Dense (pgvector cosine) and BM25 rankings are fused with RRF, as source search
    does (``retrieval/sources.py``). Members who opted out or left are never
    returned. A Discord post is visible to readers of its channel only, as a
    Discord conversation is (#85); slides and ``join`` rows are visible to all.
    """

    distance = SelfIntro.embedding.cosine_distance(query_vector).label("distance")
    rows = session.execute(
        select(SelfIntro.id, SelfIntro.member_id, SelfIntro.origin, SelfIntro.text, distance)
        .join(Employee, Employee.id == SelfIntro.member_id)
        .where(
            and_(
                SelfIntro.org_id == org_id,
                SelfIntro.embedding.isnot(None),
                Employee.opted_out_at.is_(None),
                Employee.is_active.is_(True),
                _visible_intro(org_id, scope),
            )
        )
        .order_by(distance, SelfIntro.id)
    ).all()
    if not rows:
        return []
    found = {row.id: row for row in rows}
    cosine = {row.id: 1.0 - float(row.distance) for row in rows}
    dense = [row.id for row in rows[:_POOL]]
    # One index per organization, shared by every reader; it rebuilds itself when
    # any introduction's text changes.
    corpus = session.execute(
        select(SelfIntro.id, SelfIntro.text)
        .where(SelfIntro.org_id == org_id)
        .order_by(SelfIntro.id)
    ).all()
    index = cached_bm25_index(f"intros:{org_id}", ((r.id, r.text) for r in corpus))
    sparse = [id_ for id_, _ in index.search(query, len(corpus)) if id_ in found][:_POOL]
    fused = rrf([dense, sparse], weights=[1.0, get_settings().bm25_weight])[:top_k]
    return [
        IntroHit(
            member_id=found[id_].member_id,
            intro_id=int(id_),  # type: ignore[call-overload]
            origin=found[id_].origin,
            text=found[id_].text,
            score=score,
            cosine=cosine[id_],
        )
        for id_, score in fused
    ]

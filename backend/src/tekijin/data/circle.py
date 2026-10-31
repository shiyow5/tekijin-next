"""A fictional astronomy circle, for regression tests and threshold tuning (#79).

Real servers' messages cannot live in a public repository, so ``fixtures/circle/``
holds one made-up organization: members, Discord conversations (mostly small
talk, as in a real server), self-introduction slides, a plan, a manual, two
handover notes, and questions with their expected answers.

Seeding goes through the same functions as real ingestion and only touches this
circle's organization. Ids come from the fixture keys, so running it again, or
editing one conversation, leaves the rest as they are.

CLI::

    python -m tekijin.data.circle
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.data.conversations import (
    IncomingMessage,
    conversation_key,
    delete_messages,
    ingest_messages,
)
from tekijin.data.members import get_or_create_org_for_guild, resolve_discord_member
from tekijin.data.sources import MANUAL, delete_source_item, upsert_source_item
from tekijin.models.tables import ChannelMessage, SourceItem
from tekijin.retrieval.embedding import Embedder

GUILD_ID = "900000000000000001"
ORG_NAME = "星見ヶ丘天文同好会"
FIXTURES_DIR = Path(__file__).resolve().parents[4] / "fixtures" / "circle"

_DOC = "doc:"
_CONV = "conv:"
_MEMBER_BASE = 910_000_000_000_000_000
_ID_BASE = 920_000_000_000_000_000
# Lines of a conversation are this far apart, starting at 21:00 Japan time.
_LINE_GAP = dt.timedelta(minutes=2)
_START_UTC = dt.time(12, 0)


@dataclass(frozen=True, slots=True)
class Circle:
    members: list[dict[str, Any]]
    documents: list[dict[str, Any]]
    conversations: list[dict[str, Any]]
    questions: list[dict[str, Any]]
    # Document key -> body.
    bodies: dict[str, str]


@dataclass(frozen=True, slots=True)
class SeedResult:
    org_id: int
    conversations_changed: int
    documents_changed: int


def load_circle(directory: Path = FIXTURES_DIR) -> Circle:
    documents = _read(directory / "documents.json")
    return Circle(
        members=_read(directory / "members.json"),
        documents=documents,
        conversations=_read(directory / "conversations.json"),
        questions=_read(directory / "questions.json"),
        bodies={d["key"]: (directory / d["file"]).read_text("utf-8") for d in documents},
    )


def _read(path: Path) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads(path.read_text("utf-8"))
    return data


def source_text(circle: Circle, ref: str) -> str:
    """The text behind a question's ``doc:…`` or ``conv:…`` reference."""

    if ref.startswith(_DOC):
        return circle.bodies[ref.removeprefix(_DOC)]
    return "\n".join(text for _, text in _conversation(circle, ref)["lines"])


def external_id(circle: Circle, ref: str) -> str:
    """The ``source_items.external_id`` a question's reference is stored under."""

    if ref.startswith(_DOC):
        return f"circle:{ref.removeprefix(_DOC)}"
    conversation = _conversation(circle, ref)
    return conversation_key(
        _stable_id(conversation["channel"]),
        _thread_id(conversation),
        _line_time(conversation, 0),
    )


def _conversation(circle: Circle, ref: str) -> dict[str, Any]:
    if not ref.startswith(_CONV):
        raise ValueError(f"not a source reference: {ref}")
    key = ref.removeprefix(_CONV)
    for conversation in circle.conversations:
        if conversation["key"] == key:
            return conversation
    raise KeyError(key)


def _stable_id(text: str) -> int:
    return _ID_BASE + zlib.crc32(text.encode("utf-8")) * 1000


def _thread_id(conversation: dict[str, Any]) -> int | None:
    return _stable_id(f"thread:{conversation['key']}") if conversation["thread"] else None


def _line_time(conversation: dict[str, Any], index: int) -> dt.datetime:
    day = dt.date.fromisoformat(conversation["date"])
    return dt.datetime.combine(day, _START_UTC) + _LINE_GAP * index


def _discord_id(member_key: str) -> str:
    return str(_MEMBER_BASE + int(member_key.removeprefix("m")))


def _messages(circle: Circle) -> list[IncomingMessage]:
    names = {m["key"]: m["name"] for m in circle.members}
    return [
        IncomingMessage(
            message_id=_stable_id(f"message:{conversation['key']}") + index,
            channel_id=_stable_id(conversation["channel"]),
            channel_name=conversation["channel"],
            thread_id=_thread_id(conversation),
            thread_name=conversation["thread"],
            author_discord_id=_discord_id(speaker),
            author_name=names[speaker],
            content=content,
            created_at=_line_time(conversation, index),
            edited_at=None,
        )
        for conversation in circle.conversations
        for index, (speaker, content) in enumerate(conversation["lines"])
    ]


def seed_circle(
    session: Session, embedder: Embedder, circle: Circle, *, now: dt.datetime
) -> SeedResult:
    """Store the circle under its own organization. The caller commits."""

    org = get_or_create_org_for_guild(session, GUILD_ID, name=ORG_NAME)
    authors = {
        member["key"]: resolve_discord_member(
            session, org, _discord_id(member["key"]), display_name=member["name"], now=now
        ).id
        for member in circle.members
    }
    messages = _messages(circle)
    _forget_removed(session, embedder, circle, messages, org_id=org.id, now=now)
    ingested = ingest_messages(
        session, embedder, org=org, guild_id=GUILD_ID, messages=messages, now=now
    )
    changed = 0
    for document in circle.documents:
        result = upsert_source_item(
            session,
            embedder,
            org_id=org.id,
            provider=MANUAL,
            external_id=external_id(circle, f"{_DOC}{document['key']}"),
            title=document["title"],
            body=circle.bodies[document["key"]],
            url=None,
            author_id=authors[document["author"]],
            now=now,
        )
        changed += result.status != "unchanged"
    return SeedResult(
        org_id=org.id,
        conversations_changed=ingested.conversations,
        documents_changed=changed,
    )


def _forget_removed(
    session: Session,
    embedder: Embedder,
    circle: Circle,
    messages: list[IncomingMessage],
    *,
    org_id: int,
    now: dt.datetime,
) -> None:
    """Drop what an earlier seed stored but the fixtures no longer have, so a
    reused database matches a fresh seed."""

    stored_messages = session.scalars(
        select(ChannelMessage.message_id).where(ChannelMessage.org_id == org_id)
    ).all()
    current_messages = {message.message_id for message in messages}
    delete_messages(
        session,
        embedder,
        org_id=org_id,
        guild_id=GUILD_ID,
        message_ids=[m for m in stored_messages if m not in current_messages],
        now=now,
    )
    current_documents = {external_id(circle, f"{_DOC}{d['key']}") for d in circle.documents}
    stored_documents = session.execute(
        select(SourceItem.id, SourceItem.external_id).where(
            SourceItem.org_id == org_id, SourceItem.provider == MANUAL
        )
    ).all()
    for item_id, stored_id in stored_documents:
        if stored_id not in current_documents:
            delete_source_item(session, item_id=item_id, org_id=org_id)


def main() -> int:  # pragma: no cover - exercised via CLI
    from tekijin.config import get_settings
    from tekijin.data.db import get_engine, get_sessionmaker, session_scope
    from tekijin.retrieval.embedding import SentenceTransformerEmbedder

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        use_e5_prefix=settings.embedding_use_e5_prefix,
        query_prefix=settings.embedding_query_prefix,
        passage_prefix=settings.embedding_passage_prefix,
        trust_remote_code=settings.embedding_trust_remote_code,
        revision=settings.embedding_model_revision,
        app_env=settings.app_env,
    )
    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    with session_scope(get_sessionmaker(get_engine())) as session:
        result = seed_circle(session, embedder, load_circle(), now=now)
    sys.stdout.write(f"{result}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())

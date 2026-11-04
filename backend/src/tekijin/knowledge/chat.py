"""Group raw employee chat into conversation units for knowledge extraction (#448).

Raw chat is mostly noise — a single line ("了解です" / "在宅です") carries no
reusable knowledge, and feeding raw messages to System 1 as evidence measured as
near-useless. The value in chat is
in the *exchanges*: a question that gets a substantive answer, an issue that gets
resolved. So the unit of extraction is a **conversation**, not a message.

This module turns ``employee_chat_history`` rows into
:class:`~tekijin.knowledge.extract.ExtractionSource` transcripts by grouping
consecutive messages in the same channel that fall within a time gap. The
:class:`~tekijin.knowledge.extract.CaseExtractor` then distils each transcript
into a case (``問題 → 打ち手/回答 → 結果``), leaning hard on ``extractable=false`` so
chit-chat and logistics store nothing. Chat has no precomputed tags, so topics are
inferred from the model's hints and snapped to the canonical vocabulary by
``extract_and_store(..., infer_topics_from_hints=True)``.

Pure DB reads + in-memory grouping — no LLM, no network — so tests drive it against
the seeded fixtures deterministically.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from tekijin.knowledge.extract import ExtractionSource
from tekijin.models.tables import EmployeeChatHistory

# Channels that are pure noise by construction: company-wide broadcasts and
# chit-chat. Excluded up front so the batch does not spend LLM calls proving they
# are not cases (extractable=false is still the backstop for anything that slips
# through in a signal channel).
DEFAULT_EXCLUDED_CHANNELS: frozenset[str] = frozenset({"雑談", "general"})

_DEFAULT_GAP_MINUTES = 30
_DEFAULT_MIN_MESSAGES = 2
_DEFAULT_MAX_MESSAGES = 40


def _transcript(rows: Sequence[EmployeeChatHistory]) -> str:
    """Render a conversation as speaker-labelled lines the model reads.

    Names are not joined here (the extractor only needs turn-taking), so the
    sender's employee id labels each line — enough for the LLM to see a question
    and the reply that answers it.
    """

    lines = [f"社員{row.sender_employee_id}: {(row.message or '').strip()}" for row in rows]
    return "\n".join(lines)


def _flush(
    rows: list[EmployeeChatHistory], channel: str, min_messages: int
) -> ExtractionSource | None:
    """Turn a buffered run of same-channel messages into one source, or ``None``
    if it is too short to be a conversation."""

    if len(rows) < min_messages:
        return None
    # Key on BOTH the first and last message id, not just the first: the conversation
    # boundary is derived from the grouping params (gap/min/max) and the rows present,
    # so a re-run with different params — or a backdated row landing inside an already
    # extracted window — would otherwise reuse the same source_id for a DIFFERENT
    # transcript and silently overwrite an already-reviewed unit's content (#448
    # review). Including last_id makes a differently-bounded regrouping a NEW unit.
    first_id, last_id = rows[0].id, rows[-1].id
    return ExtractionSource(
        source_type="chat",
        source_id=f"chat_{channel}_{first_id}_{last_id}",
        text=_transcript(rows),
        topics=(),  # chat has no precomputed tags; inferred at extraction time
    )


def group_rows(
    rows: Iterable[EmployeeChatHistory],
    *,
    gap_minutes: int = _DEFAULT_GAP_MINUTES,
    min_messages: int = _DEFAULT_MIN_MESSAGES,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
    exclude_channels: frozenset[str] = DEFAULT_EXCLUDED_CHANNELS,
    limit: int | None = None,
) -> list[ExtractionSource]:
    """Group pre-sorted chat rows into conversation sources (pure; no DB).

    ``rows`` MUST already be ordered by ``(channel, sent_at, id)``. Iterated once and
    lazily, so a streaming cursor can be passed straight through. A conversation is a
    maximal run of consecutive rows in the SAME channel whose successive timestamps
    are within ``gap_minutes`` (a longer silence starts a new one) and which is at
    most ``max_messages`` long (a channel that never goes quiet is chunked so one
    transcript does not balloon). Runs shorter than ``min_messages`` are dropped — a
    lone broadcast is not an exchange. Rows with no text or no timestamp are skipped,
    and ``exclude_channels`` are filtered out. ``limit`` bounds the number of
    conversations returned (PoC / cost control); ``limit=0`` yields none.
    """

    if limit is not None and limit <= 0:
        return []

    gap = dt.timedelta(minutes=gap_minutes)
    sources: list[ExtractionSource] = []
    buffer: list[EmployeeChatHistory] = []
    cur_channel: str | None = None
    prev_at: dt.datetime | None = None

    for row in rows:
        channel = row.channel or ""
        if channel in exclude_channels:
            continue
        sent_at = row.sent_at
        if sent_at is None or not (row.message or "").strip():
            continue
        breaks = (
            channel != cur_channel
            or (prev_at is not None and sent_at - prev_at > gap)
            or len(buffer) >= max_messages
        )
        if breaks and buffer:
            src = _flush(buffer, cur_channel or "", min_messages)
            if src is not None:
                sources.append(src)
                if limit is not None and len(sources) >= limit:
                    return sources
            buffer = []
        cur_channel = channel
        buffer.append(row)
        prev_at = sent_at

    if buffer:
        src = _flush(buffer, cur_channel or "", min_messages)
        if src is not None:
            sources.append(src)
    return sources


def chat_conversation_sources(
    session: Session,
    *,
    gap_minutes: int = _DEFAULT_GAP_MINUTES,
    min_messages: int = _DEFAULT_MIN_MESSAGES,
    max_messages: int = _DEFAULT_MAX_MESSAGES,
    exclude_channels: frozenset[str] = DEFAULT_EXCLUDED_CHANNELS,
    limit: int | None = None,
) -> list[ExtractionSource]:
    """Read chat history ordered by ``(channel, sent_at, id)`` and group it into
    conversation sources (see :func:`group_rows`)."""

    stmt = (
        select(EmployeeChatHistory)
        .where(
            EmployeeChatHistory.message.is_not(None),
            EmployeeChatHistory.sent_at.is_not(None),
        )
        .order_by(
            EmployeeChatHistory.channel,
            EmployeeChatHistory.sent_at,
            EmployeeChatHistory.id,
        )
        # Stream in batches instead of materializing the whole table: group_rows
        # iterates once in order, and with ``limit`` set the unread tail is never
        # fetched. Bounds client memory; the DB still sorts server-side (an index on
        # (channel, sent_at, id) makes that an index scan — tracked as follow-up).
        .execution_options(yield_per=500)
    )
    return group_rows(
        session.scalars(stmt),
        gap_minutes=gap_minutes,
        min_messages=min_messages,
        max_messages=max_messages,
        exclude_channels=exclude_channels,
        limit=limit,
    )

"""Case drafts from solved Discord conversations (#113, R6).

A conversation becomes a case only when it was solved, and the signal comes from
the people in it (``knowledge/solve_signals.py``, shared with Slack):

* the asker writes a thanks that closes their message, or "解決しました"; or
* someone who wrote in the conversation reacts with a check mark.

What is read is the stretch of the conversation that ended at the signal. A
conversation is a thread, or one channel's day (``data/conversations.py``), and a
channel's day can hold several questions; the stretch therefore starts after the
last message an earlier case already cited, and the asker is whoever wrote its
first message. A stretch needs at least two people in it. The draft cites every
message of the stretch, so deleting, hiding or editing any of them sends the case
back to review (``data/knowledge.py``), and a message already cited is never read
into a second case.

The organization's ``knowledge_from_conversations`` setting turns this on (off by
default). It is separate from ``expertise_from_messages``, which decides whether
messages count as evidence of who knows what (ADR-0012).
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Sequence

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from tekijin.data.conversations import conversation_filter, conversation_key
from tekijin.data.knowledge import SourceRef
from tekijin.knowledge.extract import (
    ExtractionReport,
    ExtractionSource,
    Extractor,
    extract_and_store,
)
from tekijin.knowledge.solve_signals import is_closing_utterance
from tekijin.models.tables import (
    ChannelMessage,
    Employee,
    Identity,
    KnowledgeArticle,
    KnowledgeSource,
    Organization,
)

CASE = "case"
#: The longest stretch read into one case (the most recent messages are kept).
WINDOW_MAX = 40
#: How much of each message is kept as its source's quote.
QUOTE_MAX = 500
# Namespace of the per-conversation advisory lock (the Issue number).
_LOCK_CLASS = 113
_UNKNOWN = "(不明)"


def _signal_row(session: Session, org_id: int, message_id: int) -> ChannelMessage | None:
    return session.scalars(
        select(ChannelMessage).where(
            ChannelMessage.org_id == org_id,
            ChannelMessage.message_id == message_id,
            ChannelMessage.hidden_at.is_(None),
        )
    ).first()


def _key(row: ChannelMessage) -> str:
    return conversation_key(row.channel_id, row.thread_id, row.created_at)


def case_window(
    session: Session, *, org_id: int, signal: ChannelMessage
) -> list[tuple[ChannelMessage, str]]:
    """The stretch of ``signal``'s conversation a case would be read from.

    The messages after the last one an earlier case cited, up to and including
    ``signal``, at most :data:`WINDOW_MAX`, each with its author's name.
    """

    rows = session.execute(
        select(ChannelMessage, Employee.name)
        .outerjoin(Employee, Employee.id == ChannelMessage.author_id)
        .where(
            ChannelMessage.org_id == org_id,
            ChannelMessage.hidden_at.is_(None),
            *conversation_filter(_key(signal)),
        )
        .order_by(ChannelMessage.created_at, ChannelMessage.message_id)
    ).all()
    ids = [row.id for row, _ in rows]
    if signal.id not in ids:
        return []
    upto = [(row, name or _UNKNOWN) for row, name in rows[: ids.index(signal.id) + 1]]
    cited = set(
        session.scalars(
            select(KnowledgeSource.channel_message_id)
            .join(KnowledgeArticle, KnowledgeArticle.id == KnowledgeSource.article_id)
            .where(
                KnowledgeArticle.kind == CASE,
                KnowledgeSource.channel_message_id.in_([row.id for row, _ in upto]),
            )
        )
    )
    last = max((i for i, (row, _) in enumerate(upto) if row.id in cited), default=-1)
    return upto[last + 1 :][-WINDOW_MAX:]


def _member_for(session: Session, org_id: int, discord_user_id: str) -> int | None:
    return session.scalar(
        select(Identity.employee_id).where(
            Identity.org_id == org_id,
            Identity.provider == "discord",
            Identity.external_id == discord_user_id,
        )
    )


def is_solved(
    window: Sequence[tuple[ChannelMessage, str]],
    signal: ChannelMessage,
    *,
    reactor_member_id: int | None = None,
) -> bool:
    """Whether the stretch was solved by this signal.

    With ``reactor_member_id`` the signal is a check mark from that member, who
    must have written in the stretch. Without it the signal is ``signal``'s text,
    which must be the asker's closing thanks or "solved", after their question.
    """

    if len(window) < 2 or window[-1][0].id != signal.id:
        return False
    authors = {row.author_id for row, _ in window if row.author_id is not None}
    if len(authors) < 2:
        return False
    if reactor_member_id is not None:
        return reactor_member_id in authors
    asker = window[0][0].author_id
    return signal.author_id == asker and is_closing_utterance(signal.content)


def case_source(org_id: int, window: Sequence[tuple[ChannelMessage, str]]) -> ExtractionSource:
    """The stretch as a record to read: its text, its messages, and who answered."""

    first = window[0][0]
    place = f"#{first.channel_name}"
    if first.thread_name:
        place += f" › {first.thread_name}"
    lines = "\n".join(f"{name}: {row.content}" for row, name in window)
    asker = first.author_id
    answers = Counter(row.author_id for row, _ in window if row.author_id not in (None, asker))
    responder = answers.most_common(1)[0][0] if answers else None
    return ExtractionSource(
        org_id=org_id,
        text=f"Discordの会話({place})。最後に質問した人が解決したと伝えています。\n\n{lines}",
        sources=tuple(
            SourceRef(channel_message_id=row.id, quote=row.content[:QUOTE_MAX]) for row, _ in window
        ),
        author_id=responder,
        kind=CASE,
    )


def draft_conversation_case(
    session: Session,
    *,
    org_id: int,
    message_id: int,
    extractor: Extractor,
    now: dt.datetime,
    reactor_discord_id: str | None = None,
) -> ExtractionReport | None:
    """Make a case draft from the conversation solved at ``message_id``.

    ``message_id`` is the Discord id of the signal: the asker's closing message,
    or the message that got the check mark from ``reactor_discord_id``. ``None``
    when there is no case to read (the message is not stored, the stretch is not
    solved by this signal, or it was read before). Does not look at the
    organization's setting; :func:`capture_conversation_case` does. The caller
    owns the transaction.
    """

    signal = _signal_row(session, org_id, message_id)
    if signal is None:
        return None
    # One capture per conversation at a time, so two signals cannot both read
    # the same stretch. Released at the end of the transaction.
    session.execute(
        text("SELECT pg_advisory_xact_lock(:cls, hashtext(:key))"),
        {"cls": _LOCK_CLASS, "key": f"{org_id}:{_key(signal)}"},
    )
    reactor = None
    if reactor_discord_id is not None:
        reactor = _member_for(session, org_id, reactor_discord_id)
        if reactor is None:
            return None
    window = case_window(session, org_id=org_id, signal=signal)
    if not is_solved(window, signal, reactor_member_id=reactor):
        return None
    read = {row.id: row.content for row, _ in window}

    def unchanged(_source: ExtractionSource) -> bool:
        # A message may have been edited, hidden or deleted while the model read
        # the stretch. Locked, so such a change waits for this write and then
        # sends the stored case back to review itself.
        rows = session.execute(
            select(ChannelMessage.id, ChannelMessage.content, ChannelMessage.hidden_at)
            .where(ChannelMessage.id.in_(list(read)))
            .with_for_update()
        ).all()
        now_read = {mid: content for mid, content, hidden in rows if hidden is None}
        return now_read == read

    return extract_and_store(
        session, [case_source(org_id, window)], extractor, now=now, still_current=unchanged
    )


def conversations_enabled(session: Session, org_id: int) -> bool:
    """The organization's ``knowledge_from_conversations`` setting."""

    return bool(
        session.scalar(
            select(Organization.knowledge_from_conversations).where(Organization.id == org_id)
        )
    )


def capture_conversation_case(
    session: Session,
    *,
    org_id: int,
    message_id: int,
    extractor: Extractor,
    now: dt.datetime,
    reactor_discord_id: str | None = None,
) -> ExtractionReport | None:
    """:func:`draft_conversation_case`, when the organization has it switched on."""

    if not conversations_enabled(session, org_id):
        return None
    return draft_conversation_case(
        session,
        org_id=org_id,
        message_id=message_id,
        extractor=extractor,
        now=now,
        reactor_discord_id=reactor_discord_id,
    )


def closing_candidates(session: Session, *, org_id: int, message_ids: Sequence[int]) -> list[int]:
    """Of these newly stored messages, the ones whose text may close a question.

    A cheap text check, so only these are given to :func:`capture_conversation_case`
    (which checks the author is the asker). In message id order.
    """

    if not message_ids:
        return []
    rows = session.execute(
        select(ChannelMessage.message_id, ChannelMessage.content).where(
            ChannelMessage.org_id == org_id,
            ChannelMessage.message_id.in_(list(message_ids)),
            ChannelMessage.hidden_at.is_(None),
        )
    ).all()
    return sorted(m for m, content in rows if is_closing_utterance(content))

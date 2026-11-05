"""Solve-capture (#476 Screen 02): a ✅ on a hand-off thread → a knowledge draft.

The "解決した" button is pressed by almost no one — the moment a thread is solved,
the person has already moved on. So capture must ride the conversation instead: a
participant reacting ✅ on the pair-channel thread is enough to distil the resolved
Q&A into a knowledge draft (``knowledge_articles``, #112). The draft lands in the
admin's review queue (the "下書き箱") — created eagerly, so "no one pressed anything"
already leaves a draft to review later; a future prompt/button just makes reviewing it faster.

Gating (all must hold, else no-op):
- ``slack_solve_capture_enabled`` is on (OFF by default — dormant).
- the reaction is one of :data:`SOLVE_REACTIONS`.
- the channel is a TEKIJIN pair-channel (``SlackChannelLink``) — these are shared
  private channels the bot created, never DMs, so no DM ever reaches extraction and
  the #404 DM-consent question does not arise here.
- the reactor is the thread's asker or responder (not a bystander in the channel).
- the thread resolves to an accepted hand-off with extractable Q&A content.

The extraction calls the LLM, which routinely runs past Slack's ~3s event budget,
so :func:`schedule_solve_capture` runs it in a daemon thread and the webhook acks
immediately (mirrors ``notify.schedule_pending_handoff``).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import threading

from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from tekijin.config import Settings, get_settings
from tekijin.data.db import session_scope
from tekijin.data.knowledge import (
    ArticleDTO,
    articles_from_question,
    retire_article,
    verify_article,
)
from tekijin.data.messages import thread_parties
from tekijin.data.slack_channel_links import get_channel_link_by_channel_id
from tekijin.data.slack_links import get_slack_link_by_slack_user_id
from tekijin.data.slack_message_anchors import thread_for_message
from tekijin.knowledge.extract import DraftExtractor, ExtractionReport, Extractor, extract_and_store
from tekijin.knowledge.slack_thread import slack_thread_source
from tekijin.slack.client import post_message

logger = logging.getLogger(__name__)

#: Slack reaction names (no colons) that mean "this thread is solved".
SOLVE_REACTIONS = frozenset({"white_check_mark", "heavy_check_mark", "ok"})

#: Interactivity action ids for the in-thread knowledge prompt. keep/discard is the
#: first gate (#476); approve confirms the shown draft (#527 — the review loop runs in
#: Slack, no web hop) and marks it ``verified`` so it can answer questions.
KNOWLEDGE_KEEP_ACTION = "tekijin_knowledge_keep"
KNOWLEDGE_DISCARD_ACTION = "tekijin_knowledge_discard"
KNOWLEDGE_APPROVE_ACTION = "tekijin_knowledge_approve"
KNOWLEDGE_ACTION_IDS = frozenset(
    {KNOWLEDGE_KEEP_ACTION, KNOWLEDGE_DISCARD_ACTION, KNOWLEDGE_APPROVE_ACTION}
)

#: Cap each draft field shown in Slack so one long extraction can't blow past Slack's
#: 3000-char/section limit or bury the buttons below the fold.
_REVIEW_FIELD_MAX = 500

#: Posted (utterance path) when an explicit "解決" produces no capturable case, so the
#: solver is never met with silence (#525). A plain message — NOT the keep/discard
#: prompt — since there is no draft to keep. The extractor's ``extractable=false`` is a
#: correct guardrail against noise; this only makes its outcome visible and actionable.
_NO_CASE_NOTICE = (
    "「解決」を確認しました。ただ、今回の会話からは知識として残せる具体的な内容"
    "（手順・判断の理由・結論）が見つからなかったため、ナレッジ化は行いませんでした。"
    "要点を書き添えて改めてお知らせいただくと、次回から蓄積できます。"
)

# #529: visible lifecycle around the slow LLM extraction. Without the first
# message a solve utterance appears ignored for several seconds; without the
# failure message it can remain stuck at "creating" after a transient error.
CAPTURE_STARTED_TEXT = "解決を確認しました。会話を整理して知識の下書きを作成しています…"
CAPTURE_READY_TEXT = "会話から知識の下書きを作成しました。この会話を知識として残しますか？"
CAPTURE_FAILED_TEXT = (
    "知識の下書きを作成できませんでした。少し時間をおいて、もう一度解決をお知らせください。"
)

# Threads already given a no-case notice, so repeated solve-utterances on a thread that
# never yields a case don't spam it (#525 review). The success path dedups on its
# persisted draft; the no-case path has no draft to check, and a persisted "declined"
# marker would wrongly block a LATER genuine capture once the thread gains real content —
# so the notice is deduped in-process instead. Reset on restart (a restart may re-notify
# once — acceptable for a UX message). The lock makes claim-then-post atomic so two
# concurrent utterances on the same thread post at most one notice.
_no_case_notified: set[int] = set()
_no_case_notified_lock = threading.Lock()

# One extraction-start narrative per thread/process. A skipped non-case retains
# the claim so repeated closing phrases do not spam progress. A transient error
# releases it so an explicit retry can narrate a fresh attempt. Successful captures
# are durably deduped by their knowledge draft before this set is consulted.
_capture_progress_notified: set[int] = set()
_capture_progress_notified_lock = threading.Lock()


def _claim_no_case_notice(thread_id: int) -> bool:
    """True the first time a no-case notice is owed for ``thread_id`` this process, else
    False. Gates only the notice — never extraction — so a later utterance that adds real
    content still re-extracts and can succeed via the (independent) draft path."""

    with _no_case_notified_lock:
        if thread_id in _no_case_notified:
            return False
        _no_case_notified.add(thread_id)
        return True


def _claim_capture_progress(thread_id: int) -> bool:
    with _capture_progress_notified_lock:
        if thread_id in _capture_progress_notified:
            return False
        _capture_progress_notified.add(thread_id)
        return True


def _release_capture_progress(thread_id: int) -> None:
    with _capture_progress_notified_lock:
        _capture_progress_notified.discard(thread_id)


# Conservative solve-utterance markers (substring match). Deliberately RESOLUTION-
# rooted — NOT generic completion/thanks. Bare "できました" is excluded because it
# also matches unrelated completion (資料ができました / 予約ができました). Thanks are
# detected separately by :func:`is_thanks_utterance`: the event handler accepts them
# only from the thread's asker, where thanking the responder is a useful conversation-
# completion signal (#537), while a responder's ordinary closing remains a no-op.
# Product-sensitive: keep tight, and only fires while the feature flag is on.
_SOLVE_UTTERANCES: tuple[str, ...] = (
    "解決しました",
    "解決した",
    "解決です",
    "解決できました",
    "できるようになりました",
    "うまくいきました",
    "直りました",
    "動きました",
)


def is_solve_utterance(text: str) -> bool:
    """True when ``text`` reads as "this is resolved" (a capture trigger, #476)."""
    lowered = text or ""
    return any(marker in lowered for marker in _SOLVE_UTTERANCES)


_THANKS_UTTERANCES: tuple[str, ...] = ("ありがとうございました", "ありがとうございます")


def is_thanks_utterance(text: str) -> bool:
    """True for an explicit Japanese thanks that may close an asker's thread.

    This helper intentionally knows nothing about sender identity. The Slack event
    handler combines it with ``sender_id == parties["asker_id"]`` so a responder's
    polite closing cannot consume the thread's one-shot knowledge prompt (#537).
    """

    # Only a thanks that CLOSES the message is a completion signal. A prefix match
    # such as "ありがとうございます。追加で質問があります" is still an active
    # conversation and must not consume the thread's one-shot prompt.
    value = (text or "").strip().rstrip("。．.!！?？、, \t\r\n")
    return any(value.endswith(marker) for marker in _THANKS_UTTERANCES)


def thread_key(thread_id: int) -> str:
    """The label a capture reports for a thread (logs and callers)."""

    return f"slack_thread_{thread_id}"


def thread_draft(session, thread_id: int) -> ArticleDTO | None:
    """The knowledge made from a thread's question, in any state, or ``None``.

    A retired one counts: a discarded thread stays discarded.
    """

    parties = thread_parties(session, thread_id)
    if parties is None:
        return None
    found = articles_from_question(session, parties["question_id"])
    return found[0] if found else None


def _extract_thread_draft(
    session, thread_id: int, extractor: Extractor, *, parties=None
) -> tuple[str | None, ExtractionReport | None]:
    """Distil one resolved thread into a knowledge draft.

    Returns ``(key, report)``: ``key`` is :func:`thread_key` when a draft was stored,
    else ``None``; ``report`` is ``extract_and_store``'s tally (so a caller can tell
    a genuine "nothing to keep" decision — ``skipped`` — from a transient error —
    ``errored``), or ``None`` when the thread could not even be assembled.
    """

    source = slack_thread_source(session, thread_id, parties=parties)
    if source is None:
        return None, None
    report = extract_and_store(session, [source], extractor, now=_now())
    return (thread_key(thread_id) if report.stored else None), report


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None)


def capture_resolved_thread(
    session_factory: sessionmaker,
    *,
    channel_id: str,
    message_ts: str | None,
    reactor_slack_user_id: str,
    extractor: Extractor | None = None,
    settings: Settings | None = None,
) -> str | None:
    """Distil the reacted-on thread into a knowledge draft.

    ``message_ts`` is the Slack ts of the message the ✅ was placed on. The thread is
    resolved from its recorded per-message anchor (#508) so a reaction on an OLDER
    message on a reused pair channel is attributed to the thread that message
    actually belonged to — not merely the channel's most recent thread. Only when
    the message has no anchor (never mirrored — a bot post, or a message from before
    capture was enabled) does it fall back to ``current_thread_id`` (best-effort).

    Returns :func:`thread_key` when a draft was stored, else ``None`` (any gate
    failed, the thread already has a draft, or the model found nothing to keep).
    A thread is drafted at most once: re-reacting finds the existing draft and does
    nothing. Errors from extraction/storage are isolated by
    :func:`extract_and_store`; this function only opens the transaction.
    """

    settings = settings or get_settings()
    if not settings.slack_solve_capture_enabled:
        return None

    with session_scope(session_factory) as session:
        link = get_channel_link_by_channel_id(session, channel_id)
        if link is None:
            return None  # not a TEKIJIN pair-channel (also excludes every DM)
        # #508: attribute the ✅ to the thread the reacted MESSAGE belonged to (its
        # anchor), not the channel's latest thread; fall back to current only when
        # the message has no anchor.
        thread_id = thread_for_message(session, channel_id, message_ts) if message_ts else None
        if thread_id is None:
            thread_id = link.current_thread_id
        if thread_id is None:
            return None
        reactor = get_slack_link_by_slack_user_id(session, reactor_slack_user_id)
        if reactor is None:
            return None
        parties = thread_parties(session, thread_id)
        if parties is None or reactor.employee_id not in (
            parties["asker_id"],
            parties["responder_id"],
        ):
            return None  # a bystander in the channel cannot trigger capture
        if articles_from_question(session, parties["question_id"]):
            return None  # already drafted (or discarded) for this thread
        extractor = extractor or DraftExtractor(settings=settings)
        # Reaction path stores silently (a ✅ is itself the "keep" gesture — no prompt,
        # and no no-case notice); only the stored id matters here.
        stored, _counts = _extract_thread_draft(session, thread_id, extractor, parties=parties)
        return stored


def schedule_solve_capture(
    session_factory: sessionmaker,
    *,
    channel_id: str,
    message_ts: str | None,
    reactor_slack_user_id: str,
) -> None:
    """Fire-and-forget :func:`capture_resolved_thread` in a daemon thread.

    The extraction invokes the LLM, which routinely runs past Slack's ~3s event
    budget, so the webhook must ack before this completes. The thread swallows and
    logs its own errors so a failed capture never surfaces as an unhandled-thread
    crash (mirrors ``notify.schedule_pending_handoff``).
    """

    def _run() -> None:
        try:
            stored = capture_resolved_thread(
                session_factory,
                message_ts=message_ts,
                channel_id=channel_id,
                reactor_slack_user_id=reactor_slack_user_id,
            )
            if stored is not None:
                logger.info("Solve-capture stored knowledge draft %s", stored)
        except Exception:  # noqa: BLE001 - background thread boundary, must not crash silently
            logger.warning("Slack solve-capture failed", exc_info=True)

    threading.Thread(target=_run, daemon=True, name="slack-solve-capture").start()


def _prompt_blocks(thread_id: int) -> list[dict]:
    """Block Kit for the in-thread "keep / discard" prompt (#476 Screen 02).

    The buttons carry ``thread_id`` in their value, so the interactivity handler
    attributes the click to the exact thread with no ``current_thread_id`` guess
    (the same provenance the anchor gives the reaction path).
    """

    value = json.dumps({"thread_id": thread_id})
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{CAPTURE_READY_TEXT}* 確認は30秒ほどです。",
            },
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "action_id": KNOWLEDGE_KEEP_ACTION,
                    "text": {"type": "plain_text", "text": "残す"},
                    "style": "primary",
                    "value": value,
                },
                {
                    "type": "button",
                    "action_id": KNOWLEDGE_DISCARD_ACTION,
                    "text": {"type": "plain_text", "text": "残さない"},
                    "value": value,
                },
            ],
        },
    ]


def _clip(value: str | None) -> str:
    """A draft field, trimmed to Slack's per-section budget, never blank."""

    text_value = (value or "").strip()
    if not text_value:
        return "（記載なし）"
    if len(text_value) > _REVIEW_FIELD_MAX:
        return text_value[: _REVIEW_FIELD_MAX - 1] + "…"
    return text_value


def review_blocks(*, thread_id: int, article: ArticleDTO) -> list[dict]:
    """Block Kit that shows a draft's content in-thread with 承認する / 破棄する (#527).

    This is the "確認" surface: pressing 残す replaces the keep/discard prompt with this,
    so the reviewer reads the extracted case and approves or discards it without leaving
    Slack. Buttons carry ``thread_id`` (same provenance the keep/discard prompt used)."""

    value = json.dumps({"thread_id": thread_id})
    return [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*この下書きを確認してください*"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*問い*\n{_clip(article.issue)}"},
                {"type": "mrkdwn", "text": f"*答え*\n{_clip(article.resolution)}"},
                {"type": "mrkdwn", "text": f"*場面*\n{_clip(article.environment)}"},
                {"type": "mrkdwn", "text": f"*原因*\n{_clip(article.cause)}"},
            ],
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"出典: {thread_key(thread_id)}"}],
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "action_id": KNOWLEDGE_APPROVE_ACTION,
                    "text": {"type": "plain_text", "text": "承認する"},
                    "style": "primary",
                    "value": value,
                },
                {
                    "type": "button",
                    "action_id": KNOWLEDGE_DISCARD_ACTION,
                    "text": {"type": "plain_text", "text": "破棄する"},
                    "style": "danger",
                    "value": value,
                },
            ],
        },
    ]


def capture_and_prompt(
    session_factory: sessionmaker,
    *,
    channel_id: str,
    thread_id: int,
    extractor: Extractor | None = None,
    settings: Settings | None = None,
) -> str | None:
    """Utterance path (#476): eagerly draft the resolved thread + post an in-thread
    prompt so the asker/responder can keep or discard it.

    ``thread_id`` is already resolved and its sender party-verified by the caller
    (``_handle_message_event``), so no reactor gate is needed here. Deduped on the
    draft's provenance: if a draft for the thread already exists (a prior utterance
    or a ✅ already captured it), this does nothing — so the prompt is posted at most
    once per thread rather than on every "解決しました". The draft is committed BEFORE
    the prompt is posted, so a fast button click always finds it.

    The "no draft yet → create + prompt" check-then-act is made atomic per thread by
    a transaction-scoped advisory lock (#519 review): two solve-utterances landing in
    the same thread within the extractor's round-trip would otherwise both pass the
    existence check and post two independent prompts. The second caller blocks on the
    lock until the first commits, then sees the draft and returns without posting.

    When the model declines the thread as not-a-case (#525), no draft is created but the
    solver still gets one short notice — never silence — deduped per thread in-process
    (:func:`_claim_no_case_notice`) since there is no persisted draft to dedup on. A
    transient extraction *error* (not a decision) closes the visible progress state
    with a retry notice; the next utterance can then retry. The flag-off and
    existing-draft paths remain silent, exactly as before.
    """

    settings = settings or get_settings()
    if not settings.slack_solve_capture_enabled:
        return None

    with session_scope(session_factory) as session:
        # Namespaced (classid=476) so the per-thread lock cannot collide with any
        # other advisory-lock user; released automatically at transaction end.
        session.execute(
            text("SELECT pg_advisory_xact_lock(:cls, :obj)"), {"cls": 476, "obj": thread_id}
        )
        existing = thread_draft(session, thread_id)
        if existing is not None:
            return None  # already captured + prompted for this thread (stay silent)
        progress_claimed = _claim_capture_progress(thread_id)
        if progress_claimed:
            post_message(
                bot_token=settings.slack_bot_token,
                channel_id=channel_id,
                text=CAPTURE_STARTED_TEXT,
            )
        extractor = extractor or DraftExtractor(settings=settings)
        try:
            stored, counts = _extract_thread_draft(session, thread_id, extractor)
        except Exception:
            # A DB/model exception escapes the extraction boundary. The outer
            # scheduler logs it, but the user must not be left at "作成中".
            if progress_claimed:
                post_message(
                    bot_token=settings.slack_bot_token,
                    channel_id=channel_id,
                    text=CAPTURE_FAILED_TEXT,
                )
                _release_capture_progress(thread_id)
            raise

    # Best-effort posts (never raise): a lost post is a lost message, not lost data.
    if stored is None:
        # The model declined the thread as not-a-case (#525): an explicit "解決" must
        # not be met with silence, so tell the solver why nothing was saved. A plain
        # notice — there is no draft to keep, so no keep/discard prompt. Only for a
        # genuine decision (skipped), at most once per thread; a transient error
        # (errored) or an unassemblable thread (counts is None) gets a separate
        # retry notice rather than the factually wrong "見つからなかった" reason.
        if counts is not None and counts.skipped and _claim_no_case_notice(thread_id):
            post_message(
                bot_token=settings.slack_bot_token,
                channel_id=channel_id,
                text=_NO_CASE_NOTICE,
            )
        elif (counts is None or counts.errored) and progress_claimed:
            # Unlike a deliberate non-case, a transient failure may succeed on an
            # explicit retry. Close the visible "creating" state and release only
            # THIS attempt's claim so that retry can narrate itself again (#529).
            post_message(
                bot_token=settings.slack_bot_token,
                channel_id=channel_id,
                text=CAPTURE_FAILED_TEXT,
            )
            _release_capture_progress(thread_id)
        return None

    # A draft exists (committed above, so a fast button click always finds it): offer
    # the in-thread keep/discard prompt for the management review box (#477).
    post_message(
        bot_token=settings.slack_bot_token,
        channel_id=channel_id,
        text=CAPTURE_READY_TEXT,
        blocks=_prompt_blocks(thread_id),
    )
    return stored


def schedule_solve_prompt(
    session_factory: sessionmaker,
    *,
    channel_id: str,
    thread_id: int,
) -> None:
    """Fire-and-forget :func:`capture_and_prompt` in a daemon thread (LLM + Slack
    post both exceed Slack's ~3s event budget)."""

    def _run() -> None:
        try:
            stored = capture_and_prompt(session_factory, channel_id=channel_id, thread_id=thread_id)
            if stored is not None:
                logger.info("Solve-utterance prompted for knowledge draft %s", stored)
        except Exception:  # noqa: BLE001 - background thread boundary, must not crash silently
            logger.warning("Slack solve-prompt failed", exc_info=True)

    threading.Thread(target=_run, daemon=True, name="slack-solve-prompt").start()


def discard_thread_draft(session, thread_id: int) -> bool:
    """Retire a thread's knowledge draft ("残さない"). Returns whether one was found.

    Retiring (not deleting) is durable: a later capture finds the retired draft and
    does not make a new one, so a discarded thread stays discarded."""

    article = thread_draft(session, thread_id)
    if article is None:
        return False
    retire_article(session, article.id, org_id=article.org_id, now=_now())
    return True


def approve_thread_draft(session, thread_id: int, *, verifier_id: int | None) -> bool:
    """Verify a thread's draft from the in-thread "承認する" button (#527).

    A party of the thread confirms what was said in it; the web review screen
    (#112) is where an admin checks drafts from every other source. Returns whether
    a draft with an answer was verified.
    """

    article = thread_draft(session, thread_id)
    if article is None or not article.resolution:
        return False
    verify_article(session, article.id, org_id=article.org_id, verifier_id=verifier_id, now=_now())
    return True

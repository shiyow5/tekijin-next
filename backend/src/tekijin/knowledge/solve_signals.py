"""What marks a conversation as solved, whatever the chat service (#476, #113).

Slack's hand-off threads (``slack/capture.py``) and Discord's conversations
(``knowledge/conversations.py``) both turn a solved exchange into a case draft. The
signals are the same: a check-mark reaction, a resolution-rooted utterance, or a
thanks that closes the asker's message. Only the reaction names differ: Slack sends
names without colons, Discord sends the emoji itself.

Which person may give the signal (the asker, a participant) is the caller's rule;
these helpers only read the text or the emoji.
"""

from __future__ import annotations

#: Slack reaction names (no colons) that mean "this thread is solved".
SLACK_SOLVE_REACTIONS = frozenset({"white_check_mark", "heavy_check_mark", "ok"})

#: Discord unicode emoji that mean the same. "✔" without the variation selector is
#: what some clients send for ✔️.
DISCORD_SOLVE_EMOJI = frozenset({"✅", "✔️", "✔", "☑️", "☑"})

# Conservative solve-utterance markers (substring match). Deliberately RESOLUTION-
# rooted — NOT generic completion/thanks. Bare "できました" is excluded because it
# also matches unrelated completion (資料ができました / 予約ができました). Thanks are
# detected separately by :func:`is_thanks_utterance`: callers accept them only from
# the asker, where thanking the responder is a useful completion signal (#537).
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

_THANKS_UTTERANCES: tuple[str, ...] = ("ありがとうございました", "ありがとうございます")


def is_solve_utterance(text: str) -> bool:
    """True when ``text`` reads as "this is resolved" (a capture trigger, #476)."""

    lowered = text or ""
    return any(marker in lowered for marker in _SOLVE_UTTERANCES)


def is_thanks_utterance(text: str) -> bool:
    """True for an explicit Japanese thanks that may close an asker's conversation.

    Only a thanks that CLOSES the message counts. "ありがとうございます。追加で質問が
    あります" is still an active conversation. The caller checks that the sender is
    the asker, so a responder's polite closing is not a signal (#537).
    """

    value = (text or "").strip().rstrip("。．.!！?？、, \t\r\n")
    return any(value.endswith(marker) for marker in _THANKS_UTTERANCES)


def is_closing_utterance(text: str) -> bool:
    """Either signal: what an asker writes when their question is solved."""

    return is_solve_utterance(text) or is_thanks_utterance(text)


def is_solve_emoji(emoji: str | None) -> bool:
    """True for a Discord check-mark reaction."""

    return bool(emoji) and emoji in DISCORD_SOLVE_EMOJI

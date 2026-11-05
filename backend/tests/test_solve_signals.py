"""The channel-neutral "this is solved" signals shared by Slack and Discord (#113)."""

from __future__ import annotations

from tekijin.knowledge.solve_signals import (
    DISCORD_SOLVE_EMOJI,
    SLACK_SOLVE_REACTIONS,
    is_closing_utterance,
    is_solve_emoji,
    is_solve_utterance,
    is_thanks_utterance,
)
from tekijin.slack import capture


def test_slack_keeps_using_the_same_signals() -> None:
    assert capture.SOLVE_REACTIONS is SLACK_SOLVE_REACTIONS


def test_discord_check_mark_emoji() -> None:
    for emoji in ("✅", "✔️", "☑️", "✔"):
        assert emoji in DISCORD_SOLVE_EMOJI
        assert is_solve_emoji(emoji)
    assert not is_solve_emoji("👍")
    assert not is_solve_emoji("")
    assert not is_solve_emoji(None)


def test_closing_utterance_is_a_solve_or_a_closing_thanks() -> None:
    assert is_solve_utterance("おかげさまで解決しました")
    assert is_thanks_utterance("ありました。付けたら合いました、ありがとうございます")
    assert is_closing_utterance("付けたら直りました")
    assert is_closing_utterance("高梨さんありがとうございます！")
    assert not is_closing_utterance("ありがとうございます。追加で質問があります")
    assert not is_closing_utterance("了解です")
    assert not is_closing_utterance("")

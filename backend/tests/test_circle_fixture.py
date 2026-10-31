"""The fictional circle's fixtures are consistent (#79)."""

from __future__ import annotations

from collections import Counter

import pytest

from tekijin.data.circle import Circle, load_circle, source_text


@pytest.fixture(scope="module")
def circle() -> Circle:
    return load_circle()


def test_the_circle_has_the_size_the_design_asks_for(circle: Circle) -> None:
    assert 15 <= len(circle.members) <= 30
    assert len(circle.documents) == 5
    kinds = Counter(q["kind"] for q in circle.questions)
    assert kinds["answer"] >= 20
    assert kinds["abstain"] >= 5
    assert kinds["person"] >= 5


def test_small_talk_is_included(circle: Circle) -> None:
    chat = [c for c in circle.conversations if c["key"].startswith("chat-")]
    assert len(chat) >= 10


def test_keys_are_unique(circle: Circle) -> None:
    for keys in (
        [m["key"] for m in circle.members],
        [c["key"] for c in circle.conversations],
        [d["key"] for d in circle.documents],
        [q["id"] for q in circle.questions],
    ):
        assert len(keys) == len(set(keys))


def test_a_channel_has_one_conversation_per_day(circle: Circle) -> None:
    days = [(c["channel"], c["date"]) for c in circle.conversations if c["thread"] is None]
    assert len(days) == len(set(days))


def test_every_speaker_and_author_is_a_member(circle: Circle) -> None:
    members = {m["key"] for m in circle.members}
    assert {d["author"] for d in circle.documents} <= members
    for conversation in circle.conversations:
        assert {speaker for speaker, _ in conversation["lines"]} <= members


def test_answers_are_in_their_sources(circle: Circle) -> None:
    for question in circle.questions:
        if question["kind"] != "answer":
            continue
        texts = [source_text(circle, ref) for ref in question["sources"]]
        for phrase in question["contains"]:
            assert any(phrase in text for text in texts), question["id"]


def test_abstain_questions_say_why_and_people_exist(circle: Circle) -> None:
    members = {m["key"] for m in circle.members}
    for question in circle.questions:
        if question["kind"] == "abstain":
            assert question["why"]
        if question["kind"] == "person":
            assert question["members"]
            assert set(question["members"]) <= members


def test_an_unknown_reference_is_rejected(circle: Circle) -> None:
    with pytest.raises(KeyError):
        source_text(circle, "conv:missing")
    with pytest.raises(ValueError):
        source_text(circle, "focus")

"""Ranking an organization's members from evidence, without a database (#107, R5)."""

from __future__ import annotations

import pytest

from tekijin.retrieval.experts import (
    ANSWER,
    CONVERSATION,
    DOCUMENT,
    MAX_REASONS,
    RRF_K,
    SELF_INTRO,
    WEIGHTS,
    Evidence,
    intro_excerpt,
    rank_experts,
)

MEMBERS = {1: ("岩田 颯太", None), 2: ("小野 咲", "観測班"), 3: ("高梨 湊", None)}
NO_FLOOR = dict.fromkeys(WEIGHTS, 0.0)


def _ev(kind: str, member: int, rank: int, *, cosine: float = 0.5, share: float = 1.0, detail=""):
    return Evidence(
        kind=kind,
        member_id=member,
        rank=rank,
        cosine=cosine,
        share=share,
        reason={"type": kind, "detail": detail or f"{kind}-{member}-{rank}"},
    )


def test_score_sums_weight_over_k_plus_rank_per_kind() -> None:
    result = rank_experts(
        [_ev(SELF_INTRO, 1, 1), _ev(DOCUMENT, 1, 2), _ev(SELF_INTRO, 2, 2)],
        MEMBERS,
        floors=NO_FLOOR,
    )

    by_id = {e.member_id: e for e in result.experts}
    assert by_id[1].raw_score == pytest.approx(
        WEIGHTS[SELF_INTRO] / (RRF_K + 1) + WEIGHTS[DOCUMENT] / (RRF_K + 2)
    )
    assert by_id[2].raw_score == pytest.approx(WEIGHTS[SELF_INTRO] / (RRF_K + 2))
    assert [e.member_id for e in result.experts] == [1, 2]


def test_conversation_credit_is_split_by_share() -> None:
    result = rank_experts(
        [_ev(CONVERSATION, 1, 1, share=0.75), _ev(CONVERSATION, 2, 1, share=0.25)],
        MEMBERS,
        floors=NO_FLOOR,
    )

    scores = {e.member_id: e.raw_score for e in result.experts}
    assert scores[1] == pytest.approx(3 * scores[2])


def test_evidence_below_its_kinds_floor_does_not_count() -> None:
    floors = {**NO_FLOOR, SELF_INTRO: 0.4}
    result = rank_experts(
        [_ev(SELF_INTRO, 1, 1, cosine=0.39), _ev(SELF_INTRO, 2, 2, cosine=0.41)],
        MEMBERS,
        floors=floors,
    )

    assert [e.member_id for e in result.experts] == [2]
    assert result.confidence == pytest.approx(0.41)


def test_nothing_above_the_floor_recommends_nobody() -> None:
    result = rank_experts([_ev(ANSWER, 1, 1, cosine=0.1)], MEMBERS, floors={ANSWER: 0.3})

    assert result.experts == []
    assert result.confidence == 0.0


def test_documents_and_conversations_do_not_raise_the_people_confidence() -> None:
    # C4's own hits already are document_confidence; C5 must not read them twice.
    result = rank_experts(
        [_ev(DOCUMENT, 1, 1, cosine=0.55), _ev(CONVERSATION, 2, 1, cosine=0.6)],
        MEMBERS,
        floors=NO_FLOOR,
    )

    assert [e.member_id for e in result.experts] == [1, 2]
    assert result.confidence == 0.0


def test_a_tekijin_answer_raises_the_people_confidence() -> None:
    result = rank_experts([_ev(ANSWER, 1, 1, cosine=0.45)], MEMBERS, floors=NO_FLOOR)

    assert result.confidence == pytest.approx(0.45)


def test_members_outside_the_eligible_set_are_never_ranked() -> None:
    # The asker, opted-out, inactive or declined members are left out of ``members``.
    result = rank_experts([_ev(SELF_INTRO, 9, 1)], MEMBERS, floors=NO_FLOOR)

    assert result.experts == []
    assert result.confidence == 0.0


def test_reasons_are_capped_and_strongest_first() -> None:
    evidence = [
        _ev(CONVERSATION, 1, 5, share=0.5, detail="c"),
        _ev(SELF_INTRO, 1, 1, detail="a"),
        _ev(DOCUMENT, 1, 2, detail="b"),
        _ev(ANSWER, 1, 9, detail="d"),
    ]
    expert = rank_experts(evidence, MEMBERS, floors=NO_FLOOR).experts[0]

    assert len(expert.reasons) == MAX_REASONS
    assert [r["detail"] for r in expert.reasons] == ["a", "b", "d"]


def test_the_same_reason_is_shown_once() -> None:
    evidence = [_ev(DOCUMENT, 1, 1, detail="x"), _ev(DOCUMENT, 1, 2, detail="x")]
    expert = rank_experts(evidence, MEMBERS, floors=NO_FLOOR).experts[0]

    assert [r["detail"] for r in expert.reasons] == ["x"]


def test_ties_break_on_member_id() -> None:
    result = rank_experts([_ev(SELF_INTRO, 3, 1), _ev(SELF_INTRO, 2, 1)], MEMBERS, floors=NO_FLOOR)

    assert [e.member_id for e in result.experts] == [2, 3]


def test_recommendation_has_every_field_consumers_read() -> None:
    expert = rank_experts(
        [_ev(DOCUMENT, 2, 1, detail="『手引き』を書いた")], MEMBERS, floors=NO_FLOOR
    ).experts[0]
    rec = expert.as_recommendation()

    assert set(rec) == {"person_id", "name", "dept", "score", "confidence", "reasons"}
    assert rec["person_id"] == 2
    assert rec["name"] == "小野 咲"
    assert rec["dept"] == "観測班"
    assert rec["confidence"] in {"高", "低"}
    assert rec["reasons"] == [{"type": DOCUMENT, "detail": "『手引き』を書いた"}]
    # The web gauge reads ``score``; it stays inside the scorer's range.
    assert 0.0 < rec["score"] <= 1.8


def test_a_self_introduction_alone_is_low_confidence_and_a_document_high() -> None:
    result = rank_experts([_ev(SELF_INTRO, 1, 1), _ev(DOCUMENT, 2, 1)], MEMBERS, floors=NO_FLOOR)
    labels = {e.member_id: e.as_recommendation()["confidence"] for e in result.experts}

    assert labels == {1: "低", 2: "高"}


def test_the_recommendation_is_a_copy() -> None:
    expert = rank_experts([_ev(DOCUMENT, 2, 1)], MEMBERS, floors=NO_FLOOR).experts[0]
    rec = expert.as_recommendation()
    rec["reasons"][0]["detail"] = "changed"

    assert expert.as_recommendation()["reasons"][0]["detail"] != "changed"


def test_intro_excerpt_drops_the_slide_heading_and_clips() -> None:
    text = "スライド4 岩田颯太(機材係)\n3年。望遠鏡と赤道儀の整備が趣味です。" + "あ" * 100

    excerpt = intro_excerpt(text, origin="slides")

    assert excerpt.startswith("3年。望遠鏡と赤道儀")
    assert len(excerpt) <= 60
    assert excerpt.endswith("…")


def test_intro_excerpt_keeps_a_one_line_join_text() -> None:
    assert intro_excerpt("赤道儀の修理が得意です", origin="join") == "赤道儀の修理が得意です"

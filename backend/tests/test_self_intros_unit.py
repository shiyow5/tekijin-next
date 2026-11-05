"""Splitting a self-introduction document into members' sections (#106)."""

from __future__ import annotations

from tekijin.data.circle import load_circle
from tekijin.data.self_intros import (
    intro_sections,
    is_intro_channel,
    is_intro_document,
    section_member,
    split_sections,
)


def _circle_names() -> dict[int, str]:
    return {index: m["name"] for index, m in enumerate(load_circle().members, start=1)}


def test_the_circle_slides_give_twelve_members() -> None:
    circle = load_circle()
    names = _circle_names()
    sections = intro_sections(circle.bodies["intro-slides"], names)

    assert len(sections) == 12
    iwata = next(i for i, n in names.items() if n == "岩田 颯太")
    assert sections[iwata].startswith("スライド4 岩田颯太(機材係)")
    assert "赤道儀" in sections[iwata]
    # The welcome line above the first slide belongs to no one.
    assert not any("ようこそ" in text for text in sections.values())


def test_markdown_headings_split_people_but_not_their_subheadings() -> None:
    body = (
        "# 2026年度 自己紹介\n\n"
        "全員分をまとめました。\n\n"
        "## 山田 太郎\n"
        "会計をしています。\n"
        "### 得意なこと\n"
        "表計算。\n\n"
        "## 佐藤花子\n"
        "写真を撮ります。\n"
    )
    names = {1: "山田太郎", 2: "佐藤 花子"}

    sections = intro_sections(body, names)

    assert set(sections) == {1, 2}
    assert "表計算" in sections[1]
    assert "写真" in sections[2] and "表計算" not in sections[2]


def test_blank_lines_split_a_document_without_headings() -> None:
    body = "山田太郎です。会計をしています。\n\n佐藤花子です。写真を撮ります。"
    names = {1: "山田 太郎", 2: "佐藤 花子"}

    assert intro_sections(body, names) == {
        1: "山田太郎です。会計をしています。",
        2: "佐藤花子です。写真を撮ります。",
    }


def test_a_section_naming_two_members_is_dropped() -> None:
    body = "スライド1 山田太郎と佐藤花子\n二人で会計をしています。\n\nスライド2 鈴木一郎\n機材係。"
    names = {1: "山田太郎", 2: "佐藤花子", 3: "鈴木一郎"}

    assert set(intro_sections(body, names)) == {3}


def test_two_members_with_the_same_name_get_nothing() -> None:
    body = "スライド1 山田太郎\n会計。\n\nスライド2 鈴木一郎\n機材係。"
    names = {1: "山田 太郎", 2: "山田太郎", 3: "鈴木一郎"}

    assert set(intro_sections(body, names)) == {3}


def test_a_section_naming_no_member_is_dropped() -> None:
    body = "スライド1 だれか\n会計。\n\nスライド2 鈴木一郎\n機材係。"

    assert set(intro_sections(body, {3: "鈴木一郎"})) == {3}
    assert intro_sections("だれのことも書いていない", {3: "鈴木一郎"}) == {}


def test_a_shorter_name_inside_a_longer_one_is_the_longer_member() -> None:
    names = {1: "佐藤", 2: "佐藤花子"}

    assert section_member("スライド1 佐藤花子\n写真。", names) == 2
    assert section_member("スライド2 佐藤\n会計。", names) == 1


def test_the_heading_is_preferred_to_names_mentioned_below() -> None:
    names = {1: "山田太郎", 2: "鈴木一郎"}

    assert section_member("スライド1 山田太郎\n機材は鈴木一郎さんに習いました。", names) == 1


def test_split_sections_keeps_every_slide() -> None:
    body = load_circle().bodies["intro-slides"]
    names = {i: "".join(n.split()) for i, n in _circle_names().items()}

    sections = split_sections(body, names)

    assert len(sections) == 13  # the welcome line, then 12 slides


def test_intro_documents_and_channels() -> None:
    assert is_intro_document("manual", "2026年度 自己紹介スライド")
    assert is_intro_document("google", "自己紹介")
    assert not is_intro_document("manual", "会計の引継ぎ書")
    assert not is_intro_document("manual", None)
    # A Discord conversation's title names its channel; it is not a document.
    assert not is_intro_document("discord", "#自己紹介 2026-04-01")

    for name in ("自己紹介", "自己紹介-2026", "introductions", "Introductions", "👋introductions"):
        assert is_intro_channel(name), name
    for name in ("雑談", "intro", None):
        assert not is_intro_channel(name), name

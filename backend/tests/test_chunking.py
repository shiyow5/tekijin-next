"""Unit tests for splitting an ingested text into chunks (#70, R2)."""

from __future__ import annotations

import pytest

from tekijin.ingest.chunking import split_text


def test_short_text_is_one_chunk() -> None:
    assert split_text("部室の鍵は守衛室で借りる。") == ["部室の鍵は守衛室で借りる。"]


def test_blank_text_has_no_chunks() -> None:
    assert split_text("") == []
    assert split_text(" \n\n  \n") == []


def test_paragraphs_are_packed_up_to_the_limit() -> None:
    paras = ["あ" * 40, "い" * 40, "う" * 40]
    chunks = split_text("\n\n".join(paras), max_chars=90)
    # The first two fit together (40 + 2 + 40 = 82), the third starts a new chunk.
    assert chunks == [f"{paras[0]}\n\n{paras[1]}", paras[2]]


def test_a_long_paragraph_is_split_at_sentence_ends() -> None:
    sentence = "い" * 29 + "。"
    chunks = split_text(sentence * 5, max_chars=70)
    assert chunks == [sentence * 2, sentence * 2, sentence]
    assert all(len(c) <= 70 for c in chunks)


def test_a_sentence_longer_than_the_limit_is_cut_hard() -> None:
    chunks = split_text("x" * 250, max_chars=100)
    assert chunks == ["x" * 100, "x" * 100, "x" * 50]


def test_line_endings_and_surrounding_space_are_normalized() -> None:
    assert split_text("  一行目\r\n二行目  \r\n\r\n\r\n三段落目 ") == ["一行目\n二行目\n\n三段落目"]


def test_the_limit_must_be_positive() -> None:
    with pytest.raises(ValueError):
        split_text("abc", max_chars=0)

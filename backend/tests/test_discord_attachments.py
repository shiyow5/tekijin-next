"""Unit tests for turning ``/tekijin add``'s text and attachment into one note (#74)."""

from __future__ import annotations

import pytest

from tekijin.discord_bot.attachments import (
    MAX_FILE_BYTES,
    AddInputError,
    check_attachment,
    compose_add,
    decode_attachment,
)


def test_text_only() -> None:
    assert compose_add(text="本文", title="題名", file_name=None, file_text=None) == (
        "本文",
        "題名",
    )


def test_file_only_takes_its_name_as_the_title() -> None:
    assert compose_add(text=None, title=None, file_name="引継ぎ書.md", file_text="# 会計") == (
        "# 会計",
        "引継ぎ書",
    )


def test_a_given_title_wins_over_the_file_name() -> None:
    _, title = compose_add(text=None, title="会計", file_name="a.md", file_text="x")
    assert title == "会計"


def test_text_and_file_are_joined() -> None:
    body, _ = compose_add(text="前書き", title=None, file_name="a.txt", file_text="中身")
    assert body == "前書き\n\n中身"


def test_neither_text_nor_file_is_an_error() -> None:
    with pytest.raises(AddInputError, match="本文かファイル"):
        compose_add(text=None, title=None, file_name=None, file_text=None)
    with pytest.raises(AddInputError, match="本文かファイル"):
        compose_add(text="  ", title=None, file_name="a.md", file_text=" \n")


def test_too_long_is_an_error() -> None:
    with pytest.raises(AddInputError, match="2万字"):
        compose_add(text=None, title=None, file_name="a.md", file_text="x" * 20001)


@pytest.mark.parametrize("name", ["memo.txt", "README.md", "notes.MARKDOWN"])
def test_text_files_are_accepted(name: str) -> None:
    check_attachment(name, 100)


@pytest.mark.parametrize("name", ["slides.pdf", "report.docx", "photo.png", "noext"])
def test_other_files_are_refused(name: str) -> None:
    with pytest.raises(AddInputError, match=".txt"):
        check_attachment(name, 100)


def test_a_large_file_is_refused() -> None:
    with pytest.raises(AddInputError, match="200KB"):
        check_attachment("big.md", MAX_FILE_BYTES + 1)


def test_decode_reads_utf8_with_or_without_a_bom() -> None:
    assert decode_attachment("会計".encode()) == "会計"
    assert decode_attachment("﻿会計".encode()) == "会計"


def test_decode_refuses_other_encodings() -> None:
    with pytest.raises(AddInputError, match="UTF-8"):
        decode_attachment("会計".encode("shift_jis"))

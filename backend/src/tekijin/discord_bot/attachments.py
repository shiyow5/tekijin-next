"""Turn ``/tekijin add``'s typed text and attached file into one note (#74).

Only plain text and Markdown are read for now; PDF or Word would need their own
extraction. The checks run in the bot, before anything is downloaded or sent, so a
wrong file gets a reply that says what to fix.
"""

from __future__ import annotations

from pathlib import PurePath

# A long hand-over document in plain text is well under this.
MAX_FILE_BYTES = 200_000
# The backend's limit for one note (api/discord_routes.py, api/source_routes.py).
MAX_BODY_CHARS = 20000
TEXT_SUFFIXES = (".txt", ".md", ".markdown")


class AddInputError(ValueError):
    """What is wrong with the input, worded for the member who typed the command."""


def check_attachment(file_name: str, size: int) -> None:
    if PurePath(file_name).suffix.lower() not in TEXT_SUFFIXES:
        raise AddInputError("添付できるのは、テキスト(.txt)とMarkdown(.md)のファイルだけです。")
    if size > MAX_FILE_BYTES:
        raise AddInputError("ファイルが大きすぎます。200KBまでにしてください。")


def decode_attachment(data: bytes) -> str:
    try:
        # utf-8-sig drops the BOM some Windows editors write.
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AddInputError(
            "ファイルを文字として読めませんでした。UTF-8で保存し直してから添付してください。"
        ) from exc


def compose_add(
    *, text: str | None, title: str | None, file_name: str | None, file_text: str | None
) -> tuple[str, str | None]:
    """The note's body and title: the text, then the file's contents.

    Without a title, the file's name (without its extension) becomes the title.
    """

    parts = [p.strip() for p in (text, file_text) if p and p.strip()]
    if not parts:
        raise AddInputError("本文かファイルのどちらかを指定してください。")
    body = "\n\n".join(parts)
    if len(body) > MAX_BODY_CHARS:
        raise AddInputError("長すぎます。2万字までに分けて追加してください。")
    if not title and file_name:
        title = PurePath(file_name).stem or None
    return body, title

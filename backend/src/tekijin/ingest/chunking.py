"""Split an ingested text into chunks small enough to embed one by one (#70).

The old ``documents`` table embedded title + body as ONE vector, which works for a
short note but not for a manual or a hand-over document: the passage that answers a
question drowns in the rest. Chunks follow the text's own structure — paragraphs
first, then sentences — and only cut mid-sentence when one sentence alone is longer
than the limit.
"""

from __future__ import annotations

import re

# About one screen of Japanese; the local embedder reads 512 tokens, which this stays
# under with the title prefixed.
DEFAULT_MAX_CHARS = 600

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
# Split AFTER a sentence end, keeping the punctuation with its sentence.
_SENTENCE_END = re.compile(r"(?<=[。．！？!?\n])")
_SEPARATOR = "\n\n"


def split_text(text: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Chunks of ``text``, each at most ``max_chars`` long, in reading order.

    Neighbouring paragraphs are packed into one chunk while they fit. A blank text
    has no chunks.
    """

    if max_chars <= 0:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    chunks: list[str] = []
    current = ""
    for piece in _pieces(text, max_chars):
        joined = f"{current}{_SEPARATOR}{piece}" if current else piece
        if len(joined) <= max_chars:
            current = joined
            continue
        if current:
            chunks.append(current)
        current = piece
    if current:
        chunks.append(current)
    return chunks


def _pieces(text: str, max_chars: int) -> list[str]:
    """Paragraphs, with any paragraph over the limit broken into sentence groups."""

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = "\n".join(line.rstrip() for line in lines)
    pieces: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(normalized):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_long(paragraph, max_chars))
    return pieces


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for sentence in (s for s in _SENTENCE_END.split(paragraph) if s):
        while len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if len(current) + len(sentence) <= max_chars:
            current += sentence
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return [p.strip() for p in parts if p.strip()]

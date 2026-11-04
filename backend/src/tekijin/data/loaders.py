"""Fixture loading and value parsing helpers.

Pure, database-free functions that read the synthetic JSON fixtures and coerce
their string dates/timestamps into ``datetime`` objects. Kept separate from the
ORM/seed code so the mapping logic is unit-testable without a database.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

# Relative paths (from ``fixtures_dir``) of every fixture file, keyed by a short
# logical name used throughout the seed.
FIXTURE_FILES: dict[str, str] = {
    "employees": "people/employees.json",
    "profiles": "people/employee_profiles.json",
    "employee_chat": "chat/employee_chat_history.json",
    "questions": "questions/questions.json",
    "answers": "answers/answers.json",
    "documents": "documents/documents.json",
}


def load_json(path: Path) -> list[dict[str, Any]]:
    """Read a JSON array file into a list of dicts."""

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
    return data


def load_fixture(fixtures_dir: Path, name: str) -> list[dict[str, Any]]:
    """Load a single named fixture relative to ``fixtures_dir``."""

    try:
        rel = FIXTURE_FILES[name]
    except KeyError as exc:  # pragma: no cover - guarded by callers/tests
        raise KeyError(f"Unknown fixture '{name}'") from exc
    return load_json(fixtures_dir / rel)


def parse_date(value: str | None) -> dt.date | None:
    """Parse ``YYYY-MM-DD`` into a :class:`date` (``None`` passes through)."""

    if value is None or value == "":
        return None
    return dt.date.fromisoformat(value)


def parse_datetime(value: str | None) -> dt.datetime | None:
    """Parse an ISO date or datetime string into a :class:`datetime`.

    Accepts both ``YYYY-MM-DD`` (documents) and ``YYYY-MM-DDTHH:MM:SS``
    (profiles, chat, reports) so a single parser covers every timestamp column.
    """

    if value is None or value == "":
        return None
    if "T" not in value and " " not in value and len(value) == 10:
        return dt.datetime.fromisoformat(value + "T00:00:00")
    return dt.datetime.fromisoformat(value)

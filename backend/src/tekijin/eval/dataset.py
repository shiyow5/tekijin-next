"""Load the offline evaluation query set (``eval_person.json``, v2).

``eval_person.json`` is the **primary** eval set (fixtures README / Issue #43):
its gold experts were derived from ``projects`` + ``daily_reports`` (both
fixtures removed in #97; the labels are kept as they are) and deliberately NOT
from ``answers`` — the evidence the scorer weighs most — so it
actually measures recommendation quality. (The older ``eval_queries.json`` is
deprecated: its labels came from ``answers``, so an "answers-count" baseline
reproduced all 40, measuring nothing.)

Each row is validated with strict JSON-type checks so a corrupted file fails
loudly instead of silently coercing (``null`` → ``"None"`` etc.) and skewing a
metric. Routes are checked against the closed label set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tekijin.config import get_settings

# Closed set of gold routes: the three A/B/C branches plus ``none`` (abstain, the
# L4 layer — scored by the robustness set, not the A/B/C route metric).
VALID_ROUTES = frozenset({"person", "prior_answer", "document", "none"})

# Difficulty layers (fixtures README): L1 easy … L4 abstain. Closed-set-checked so
# a typo ("l1") fails loudly instead of creating a phantom layer in the breakdown.
VALID_DIFFICULTIES = frozenset({"L1", "L2", "L3", "L4"})


@dataclass(frozen=True)
class EvalQuery:
    """One evaluation query with its gold labels (eval_person.json schema)."""

    id: int
    query: str
    gold_topics: list[str]
    gold_experts: list[int]
    gold_route: str
    difficulty: str
    expect_abstain: bool
    # Independent second gold set (from ``answers``); empty when absent. Used for
    # the anti-circularity check, not the primary metrics.
    gold_experts_alt: list[int]
    # #296: source ids a SELF-ANSWER (#291) should be able to cite (raw doc_id /
    # answer qa_id). Non-empty only on the data-derived routes (document /
    # prior_answer); empty for person/none. Feeds the #297 source-recall metric.
    gold_source: list[str] = field(default_factory=list)


def default_eval_queries_path() -> Path:
    """The bundled primary eval set under the configured fixtures directory."""

    return get_settings().fixtures_dir / "eval" / "eval_person.json"


def load_eval_queries(path: Path | None = None) -> list[EvalQuery]:
    """Load and validate the eval query set (defaults to ``eval_person.json``)."""

    resolved = path or default_eval_queries_path()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"eval query set must be a JSON list, got {type(raw).__name__}")
    return [_parse_query(row, resolved) for row in raw]


def _parse_query(row: object, source: Path) -> EvalQuery:
    if not isinstance(row, dict):
        raise ValueError(f"{source}: each query must be an object, got {type(row).__name__}")
    required = {"id", "query", "gold_topics", "gold_experts", "gold_route", "difficulty"}
    missing = required - row.keys()
    if missing:
        raise ValueError(f"{source}: query is missing keys {sorted(missing)}")

    route = _as_str(row["gold_route"], "gold_route")
    if route not in VALID_ROUTES:
        raise ValueError(f"{source}: gold_route {route!r} is not one of {sorted(VALID_ROUTES)}")

    difficulty = _as_str(row["difficulty"], "difficulty")
    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(
            f"{source}: difficulty {difficulty!r} is not one of {sorted(VALID_DIFFICULTIES)}"
        )

    return EvalQuery(
        id=_as_int(row["id"], "id"),
        query=_as_str(row["query"], "query"),
        gold_topics=[_as_str(t, "gold_topics[]") for t in _as_list(row["gold_topics"])],
        gold_experts=[_as_int(e, "gold_experts[]") for e in _as_list(row["gold_experts"])],
        gold_route=route,
        difficulty=difficulty,
        # ``expect_abstain`` is optional in older rows; default False.
        expect_abstain=_as_bool(row.get("expect_abstain", False), "expect_abstain"),
        gold_experts_alt=[
            _as_int(e, "gold_experts_alt[]") for e in _as_list(row.get("gold_experts_alt", []))
        ],
        # Optional (older rows / person/none routes may omit or empty it).
        gold_source=[_as_str(s, "gold_source[]") for s in _as_list(row.get("gold_source", []))],
    )


def _as_list(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"expected a list, got {type(value).__name__}")
    return value


def _as_str(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected a string, got {type(value).__name__}")
    return value


def _as_int(value: object, field: str) -> int:
    # bool is an int subclass — reject it so ``true`` never becomes ``1``.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}: expected an integer, got {type(value).__name__}")
    return value


def _as_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field}: expected a boolean, got {type(value).__name__}")
    return value

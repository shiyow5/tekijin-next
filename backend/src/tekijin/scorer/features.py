"""Pure scoring-feature functions (no DB, no clock).

Each maps already-gathered primitives (counts, dates) to a scalar,
mostly in ``[0, 1]``, so the weighted sum in :mod:`tekijin.scorer.scorer` stays
interpretable. Time enters only through an injected ``now`` — nothing here reads
the wall clock — so every result is reproducible and unit-testable.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable

from tekijin.scorer.weights import (
    ANSWER_QUALITY_ANSWER_COEF,
    ANSWER_QUALITY_HELPFUL_COEF,
    ANSWER_QUALITY_REUSE_COEF,
    ANSWER_QUALITY_SCALE,
    LOAD_HALF_SATURATION,
    RECENCY_HALF_LIFE_DAYS,
)


def saturate(total: float, scale: float = 1.0) -> float:
    """Map an unbounded non-negative ``total`` into ``[0, 1)`` monotonically.

    ``1 - exp(-total/scale)``: 0 at 0, rising with diminishing returns, so
    stacking ever more evidence never exceeds 1 and the tenth item matters less
    than the first.
    """

    if total <= 0.0:
        return 0.0
    return 1.0 - math.exp(-total / scale)


def age_in_days(now: dt.datetime, when: dt.date | dt.datetime) -> float:
    """Whole days from ``when`` to ``now`` (never negative)."""

    moment = when if isinstance(when, dt.datetime) else dt.datetime(when.year, when.month, when.day)
    delta = (now - moment).total_seconds() / 86400.0
    return max(delta, 0.0)


def decay(age_days: float, half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> float:
    """Exponential relevance decay: ``0.5 ** (age / half_life)`` (fresh -> 1.0)."""

    return 0.5 ** (max(age_days, 0.0) / half_life_days)


def recency(now: dt.datetime, moments: Iterable[dt.date | dt.datetime]) -> float:
    """Decay of the most recent experience; ``0.0`` when there is none."""

    decays = [decay(age_in_days(now, m)) for m in moments]
    return max(decays) if decays else 0.0


def answer_quality(helpful_count: int, reuse_total: int, answer_count: int) -> float:
    """Monotonic quality of a person's on-topic answers, in ``[0, 1)``.

    A helpful answer counts most, reuse adds, and simply having answered gives a
    floor. Saturating so a prolific answerer cannot dominate on volume alone.
    """

    raw = (
        ANSWER_QUALITY_ANSWER_COEF * answer_count
        + ANSWER_QUALITY_HELPFUL_COEF * helpful_count
        + ANSWER_QUALITY_REUSE_COEF * reuse_total
    )
    return saturate(raw, ANSWER_QUALITY_SCALE)


def load(recent_count: int) -> float:
    """Workload penalty in ``[0, 1)`` from recent items (recs + answers).

    ``1 - 0.5 ** (count / LOAD_HALF_SATURATION)``: 0 when idle, 0.5 at the
    half-saturation count, approaching (never reaching) 1 under heavy load.
    """

    if recent_count <= 0:
        return 0.0
    return 1.0 - 0.5 ** (recent_count / LOAD_HALF_SATURATION)


# Evidence source type that drives the confidence tier (#110). Calibrated on the
# misrecommendation slot dump via scripts/research_confidence.py: a past ANSWER on
# the topic is the single signal that separates right from wrong picks (within a
# question, answer あり 0.70 vs なし 0.55). Evidence QUANTITY (edge_weight /
# evidence_count) is non-predictive and was even inverted, so it is deliberately
# dropped.
_CONFIDENCE_ANSWER = "answer"


def confidence_label(source_types: Iterable[str]) -> str:
    """Deterministic 高/低 from the KINDS of evidence a person has (#110).

    ``source_types`` is every evidence item's ``source_type`` for this person on
    the question's topics (order/duplicates irrelevant — only the set matters).

    * **高** — has answered on the topic.
    * **低** — no past answer on the topic; a 直接相談 retrospective alone does not
      prove the person answers *here*. This is the tier a "低 は棄却" policy acts on.

    The #110 calibration (docs/benchmarks/confidence.md §4-5) also had a **中** tier
    for an answerer who carried project membership — being staffed near a product
    co-occurred with wrong picks. Project evidence was removed in #95, so no
    candidate reaches 中 any more; R5 recalibrates the tiers on its new evidence.
    """

    if _CONFIDENCE_ANSWER in set(source_types):
        return "高"
    return "低"

"""C5: deterministic route decision (確信度 × 閾値).

Chooses how to resolve the question — no LLM, no randomness. Crucially it routes
on **absolute cosine similarity** (each channel's top dense ``1 - distance``,
supplied by C4 as ``*_confidence``), NOT on RRF fusion scores: an RRF score is a
sum of ``1/(k+rank)`` and has no absolute meaning across queries, so a fixed
threshold against it is meaningless. Cosine similarity is comparable, so the
thresholds below are real "how close is it?" gates. This also resolves the
dense-similarity floor deferred from #29.

Threshold rationale is **model-specific** — cosine absolute values depend on the
embedding. These constants are calibrated to Nemotron-3-Embed-1B, whose cosines
are heavily compressed (observed 0.025–0.542 across all three channels on the
eval corpus, not the 0.6–0.95 spread of e5/BERT-style encoders). See
``docs/adr/0004`` and ``fixtures/synthetic/eval/route_calibration.json`` (#90;
recalibrated on the 66-item basis in #191; **calibration data regenerated on the
current 87-row corpus in #434 — the thresholds themselves did not move**).

Every figure below is labelled with the basis it was measured on, because the
eval corpus grew underneath these constants. On the CURRENT corpus the routed-set
accuracy is **0.833 (60/72) vs a 0.681 majority baseline**; on the 66-item basis
it read 0.818 vs 0.742.

* ``DOCUMENT_SIM`` = 0.28 — a document is on-topic enough to be the demotion
  target. Recalibrated 0.30→0.28 in #191, where 0.30 dipped document recall to
  4/10 and pushed the single-route share to 0.95 (the collapse ceiling) while
  0.28 restored recall to 5/10 and lifted accuracy 0.803→0.818. Re-swept on the
  87-row corpus (#434) the decision is unchanged and the margins are wider:

  ==============  ==========  ================  =========
  DOCUMENT_SIM    accuracy    document recall   collapse
  ==============  ==========  ================  =========
  **0.28**        **0.833**   **11/16**         **0.874**
  0.30            0.819       10/16             0.885
  0.32            0.806        9/16             0.897
  ==============  ==========  ================  =========

  The three lowest document golds (conf 0.166/0.177/0.189 — the same three rows
  on both bases) cannot be recovered without collapsing accuracy below the
  majority baseline — that is the corpus-count-routing job (#116/#119), not a
  threshold job.
* ``PERSON_WEAK_SIM`` = 0.40 — profile match below this counts as weak, letting a
  document take over (sits inside the observed 0.053–0.473 people range).
* ``PRIOR_ANSWER_SIM`` = 0.55 — **deliberately above the observed answer-cosine
  max (0.542): prior_answer never fires with Nemotron.** ``answer_confidence``
  cannot separate this route — person-gold rows reach 0.542 while prior_answer
  gold tops out at 0.410, so any firing threshold mislabels person first. Prior-
  answer detection is therefore disabled here; corpus-count routing (#119/#327)
  was measured as the alternative and REJECTED — no config Pareto-improves the
  baseline (ADR-0007), so this route stays dormant. This is a #90
  stopgap, not the intended design.

Routes:
* ``prior_answer`` — a past QA is *very* close: present who answered before, then
  hand off to that responder (flowchart PA → C6). **Dormant under Nemotron** —
  see the PRIOR_ANSWER_SIM note above (#119).
* ``document`` — the person signal is weak but a document is strongly on-topic:
  point at where it lives (答えは作らない).
* ``person`` — the main line and the default/fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tekijin.agent.state import Route

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tekijin.agent.state import RetrievalResult

# Calibrated to Nemotron-3-Embed-1B (#90). See the module docstring for the
# rationale behind each band and why prior_answer is dormant.
#
# Above the observed answer-cosine max (0.542): prior_answer never fires with
# Nemotron because answer_confidence cannot separate it. Corpus-count routing was
# tried and rejected (#327 / ADR-0007); this route stays dormant.
PRIOR_ANSWER_SIM = 0.55
# A document must be on-topic enough to be the demotion target. Recalibrated
# 0.30→0.28 on the 66-item basis (#191); see the module docstring.
DOCUMENT_SIM = 0.28
# Below this profile similarity the person signal counts as weak (a document may
# then take over). All three are cosine-similarity constants, tunable on eval.
PERSON_WEAK_SIM = 0.40

PERSON: Route = "person"
PRIOR_ANSWER: Route = "prior_answer"
DOCUMENT: Route = "document"


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: Route
    reason: str
    confidence: float


def decide_route(
    retrieval: RetrievalResult,
    *,
    prior_answer_sim: float = PRIOR_ANSWER_SIM,
    document_sim: float = DOCUMENT_SIM,
    person_weak_sim: float = PERSON_WEAK_SIM,
    prior_answer_reuse_min: int | None = None,
    prior_answer_relevance_floor: float = 0.15,
) -> RouteDecision:
    """Pick ``person`` / ``prior_answer`` / ``document`` from channel confidences.

    Deterministic: depends only on the three absolute similarities and whether
    candidate people exist — never on iteration order. The default landing spot
    is always ``person``.

    ``prior_answer_reuse_min`` (default ``None`` = OFF) turns on **corpus-count
    routing** for prior_answer (#119/#327): since Nemotron's answer cosine cannot
    separate the route (``PRIOR_ANSWER_SIM`` is above the observed max), route on
    whether the top retrieved past answer is a REUSED/canonical answer instead —
    ``reuse_count`` is the very signal ``route_for`` used to define the gold. When
    set, a top past answer with ``reuse_count >= prior_answer_reuse_min`` and
    ``answer_confidence >= prior_answer_relevance_floor`` (a low noise floor, not a
    discriminator) fires prior_answer before the dormant cosine gate below.
    """

    answer_conf = float(retrieval.get("answer_confidence", 0.0))
    document_conf = float(retrieval.get("document_confidence", 0.0))
    people_conf = float(retrieval.get("people_confidence", 0.0))
    candidate_people = retrieval.get("candidate_people") or []
    past_answers = retrieval.get("past_answers") or []

    # #119/#327: corpus-count routing for prior_answer (OFF unless a reuse floor is
    # supplied). Fires on a reused/canonical top answer rather than on cosine, which
    # cannot separate this route under Nemotron. The relevance floor only screens
    # out clearly-unrelated top hits (the retriever already returns query-matched
    # answers); reuse_count does the discrimination.
    if prior_answer_reuse_min is not None and past_answers:
        top_answer = max(past_answers, key=lambda p: p.get("score", 0.0))
        reuse = int(top_answer.get("reuse_count", 0) or 0)
        if reuse >= prior_answer_reuse_min and answer_conf >= prior_answer_relevance_floor:
            return RouteDecision(
                PRIOR_ANSWER,
                f"よく再利用される過去回答が該当（再利用 {reuse} 回）。回答者を提示します。",
                answer_conf,
            )

    # prior_answer needs BOTH a near-duplicate past QA AND actual past answers to
    # hand off to. answer_confidence already only counts questions that have
    # answers, so these agree; the past_answers check is an explicit safety gate.
    if answer_conf >= prior_answer_sim and past_answers:
        return RouteDecision(
            PRIOR_ANSWER,
            f"類似の過去QAが非常に近い（類似度 {answer_conf:.2f}）。回答者を主線として提示します。",
            answer_conf,
        )
    # Demote to document only when the person signal is weak (no near-duplicate
    # answer AND weak profile match) and a document is strongly on-topic. Reachable
    # even with candidate people present, when their profile match is weak. The
    # ``answer_conf < prior_answer_sim`` term is NOT redundant given the gate
    # above: we can reach here with a high answer_conf but empty past_answers, and
    # in that case a strong-answer query should stay on the person line, not demote.
    if (
        document_conf >= document_sim
        and people_conf < person_weak_sim
        and answer_conf < prior_answer_sim
    ):
        reason = f"組織の文書が強く該当（類似度 {document_conf:.2f}）。文書の場所を示します。"
        return RouteDecision(DOCUMENT, reason, document_conf)
    if candidate_people:
        return RouteDecision(
            PERSON,
            "候補となる担当者が見つかりました。主線（人）で取り次ぎます。",
            max(people_conf, answer_conf),
        )
    return RouteDecision(
        PERSON,
        "確度の高い手がかりはありませんが、既定どおり人へ取り次ぎます。",
        0.0,
    )

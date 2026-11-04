"""Tunable constants for the C6 expertise scorer.

Every magic number the scorer depends on lives here, frozen and documented, so
the score is reproducible and the weights can be swapped without touching logic
(technical-spec §5: "重みは評価セットで調整する。手で決めた値をそのまま出さない").
Pass a different :class:`Weights` to :class:`~tekijin.scorer.scorer.ExpertiseScorer`
to override the defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Weights:
    """Linear weights for ``score = w1·topic_fit + w2·recency + w3·answer_quality
    + w4·proximity − w5·load (+ w6·question_fit when a qsim map is supplied)``.

    Defaults: topic_fit dominates (it is the "who actually knows this" signal);
    answer_quality is the next strongest (proven, helpful answers); recency and
    proximity are supporting nudges; load is a middling penalty — strong enough
    to spread work (risk ②, 負荷集中) yet never enough to bury a true expert. The
    ``question_fit`` term (#405) is dormant unless the scorer is handed a
    ``question_similarity`` map, so the default score is unchanged.
    """

    topic_fit: float = 0.45
    recency: float = 0.15
    answer_quality: float = 0.20
    proximity: float = 0.10
    load: float = 0.20
    # #405: additive weight for the question↔past-answer similarity term (qsim).
    # NOTE: 1.0 is an EMPIRICAL coefficient, not a theoretical co-equal bound — on
    # the eval corpus a matching answer's qsim lands in the same range as a strong
    # topic_fit contribution (~0.45), which is why the two behave co-equally there.
    # It is NOT normalised: qsim's theoretical max is 1.0 (> topic_fit's 0.45 cap),
    # so if the embedding model is swapped and qsim's distribution shifts upward the
    # term could quietly dominate — re-calibrate on the eval when changing the
    # embedder (scripts/research_c6_qsim.py sweeps this weight).
    # topic_fit sees only the topic TAG and saturates at 2-3 evidence pieces
    # (ADR-0006), so it cannot re-rank on the specific QUESTION — and when C1
    # predicts the wrong topic it scores the gold expert against the wrong tag and
    # drops them. qsim is the max cosine of the question against the person's past
    # answers; adding it lifts the expert whose answers actually match the question
    # regardless of the (possibly wrong) topic label. It is applied ONLY when the
    # scorer is given a ``question_similarity`` map (feature-gated by
    # ``question_fit_enabled``); the default path never adds it, so develop stays
    # byte-identical. Calibrated on scripts/research_c6_qsim.py: at 1.0 the qsim
    # term is co-equal with topic_fit and lifts real-C1 Hit@3 0.742->0.807, with
    # the gain concentrated on the rows where C1 mispredicts the topic (Hit@3 on
    # those rows 0.444->0.778). C5 does not read the scorer, so this never changes
    # routing (person recall is preserved by construction).
    question_fit: float = 1.0


DEFAULT_WEIGHTS = Weights()

# #405: minimum qsim for the question-fit REASON to be shown. The score always adds
# ``question_fit * qsim`` continuously (that continuous signal is what the eval
# measured), but the assertive reason "質問内容が過去の回答と一致" must not appear for a
# noise-level cosine — mirroring the noise floors elsewhere (prior_answer 0.15,
# knowledge 0.20). Below this, the tiny score nudge is unexplained but immaterial.
QUESTION_FIT_REASON_FLOOR = 0.15

# Evidence base_scores (doc15 / db-schema.md): helpful answer 1.0 > project lead
# 0.8 > past answer 0.7 > certification 0.6 > project member 0.5 > self-declared
# skill 0.3. These are summed per (person, topic) and saturated into an edge
# weight; a decline is NOT negative evidence (it lowers availability, not skill).
BASE_SCORE_HELPFUL_ANSWER = 1.0
BASE_SCORE_PROJECT_LEAD = 0.8
BASE_SCORE_ANSWER = 0.7
BASE_SCORE_CERTIFICATION = 0.6
BASE_SCORE_PROJECT_MEMBER = 0.5
BASE_SCORE_SKILL = 0.3
# 直接相談のふりかえり (#247): the asker's written summary of a face-to-face
# consultation. It sits BELOW a self-declared skill (0.3) because it is HEARSAY —
# one person paraphrasing another's words — yet it records an actual consultation
# the asker explicitly tagged with a topic and rated. Capped so a chatty pair
# cannot saturate topic_fit: 4 × 0.25 = 1.0, comparable to one helpful
# answer. A retrospective marked ``unresolved`` contributes NOTHING (and never
# subtracts — 断り≠非専門), so the cap counts only the ones that did help.
BASE_SCORE_OFFLINE_CONSULT = 0.25
OFFLINE_CONSULT_EVIDENCE_CAP = 4

# ``resolution`` values that count as expertise evidence. ``unresolved`` is
# deliberately absent: it is recorded (for the accumulation metrics) but is not
# evidence either way.
OFFLINE_CONSULT_POSITIVE_RESOLUTIONS = ("resolved", "partial")

# Half-life of experience relevance: a 6-month-old project/answer counts half as
# much toward ``recency`` (technical-spec §5: 半減期6か月の時間減衰).
RECENCY_HALF_LIFE_DAYS = 182.5

# Evidence source_types that decay with time and so feed ``recency``. A
# certification does not become less true as it ages, so it is deliberately
# excluded (technical-spec §5 ties recency to projects/answers).
RECENCY_SOURCE_TYPES = ("project", "answer")

# ``load`` saturates: this many recent items (recommendations + answers in the
# window) map to a 0.5 penalty; more approaches, but never reaches, 1.0.
LOAD_HALF_SATURATION = 5.0

# ``answer_quality`` saturation scale (larger raw quality -> closer to 1.0).
ANSWER_QUALITY_SCALE = 3.0

# ``answer_quality`` raw-score coefficients: a helpful answer counts most, reuse
# adds, and simply having answered gives a floor (all summed, then saturated).
ANSWER_QUALITY_ANSWER_COEF = 0.3
ANSWER_QUALITY_HELPFUL_COEF = 0.7
ANSWER_QUALITY_REUSE_COEF = 0.2

# Approximate days per month, for the human-readable "約Nか月前" recency reason.
DAYS_PER_MONTH = 30.4

# The ``load`` window: how many days back "recent" reaches (technical-spec §5:
# 直近7日).
LOAD_WINDOW_DAYS = 7

# Proximity tiers (technical-spec §5: 同支店 > 同エリア > 全社).
PROXIMITY_SAME_BRANCH = 1.0
PROXIMITY_SAME_REGION = 0.5
PROXIMITY_COMPANY_WIDE = 0.2

# Region grouping for the middle proximity tier. Branches are cities; this maps
# each to a coarse region so "same area" sits between "same branch" and
# "company-wide". Extend as branches are added.
REGION_OF_BRANCH: dict[str, str] = {
    "本社": "関東",
    "東京": "関東",
    "名古屋": "中部",
    "大阪": "関西",
    "福岡": "九州",
}

# The closed branch vocabulary C1 must choose from when the asker names a location
# (#83). Derived from the region map so the two can never disagree: a branch C1
# emits that the scorer does not know would silently match nobody. Ordered for a
# stable JSON Schema enum (guided decoding), like TOPIC_VOCABULARY (#64).
BRANCH_VOCABULARY: tuple[str, ...] = tuple(sorted(REGION_OF_BRANCH))

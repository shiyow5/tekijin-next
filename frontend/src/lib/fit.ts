/**
 * Fit-gauge magnitudes for recommendation cards (#222 / recalibrated #240).
 *
 * The gauge shows **適合度 (fit strength)**, derived from the candidate's composite
 * score — NOT the qualitative 高/中/低 confidence label. The two are deliberately
 * decoupled (#240): after #110 the label reflects the *kind of evidence* (a topic
 * with no prior answers is 低 even for a strong candidate), so anchoring the gauge
 * to the label made a well-qualified #1 read as a 33%-capped poor match on any
 * never-asked topic. Now the gauge reads the actual fit and the label rides along
 * as a separate evidence badge (ConfidenceGauge shows both).
 *
 * The raw `score` is a weighted sum (scorer/weights.py). The base positive weights
 * (topic_fit .45 + recency .15 + answer_quality .20) sum to 0.80,
 * BUT the question-fit term (#405, `question_fit_enabled` default ON) adds
 * `question_fit(1.0) · qsim` on top, where qsim ∈ [0, 1]. So the real composite
 * ceiling is 0.80 + 1.0 = 1.80, not 0.80 (#498). Normalising against the base made any
 * candidate with a non-trivial qsim saturate at 100% — the "適合度が100ばかり" bug.
 * We normalise against the qsim-inclusive ceiling so the gauge de-saturates and
 * differentiates candidates again.
 *
 * TRADE-OFF (interim): because the frontend only receives the FINAL score, it
 * cannot separate the base term from qsim, so a genuinely strong topic expert with
 * NO matching past answer (base ~0.8, qsim 0) now reads ~44% rather than ~100%.
 * The proper fix is to have the backend — which knows the components and the active
 * weights — compute and send a normalised fit percent (follow-up to #498), removing
 * this single-ceiling guess entirely. The exact ceiling is theoretical, not eval-
 * calibrated; recalibrate on the eval when the embedder / weights change.
 */

/** Base positive scorer weights: topic_fit .45 + recency .15 + answer_quality .20
 * (scorer/weights.py). */
const BASE_POSITIVE_WEIGHTS = 0.8;
/** #405 question-fit weight (scorer/weights.py `question_fit`); qsim ∈ [0, 1] is
 * added as `question_fit · qsim`, so its max contribution is this value. */
const QUESTION_FIT_WEIGHT = 1.0;

/** Composite fit ceiling: the max score a candidate can reach with the question-fit
 * term (#405) included. Normalise against this so qsim does not saturate the gauge. */
export const MAX_COMPOSITE_SCORE = BASE_POSITIVE_WEIGHTS + QUESTION_FIT_WEIGHT;

/** Fraction of the ring each qualitative level fills; the ConfidenceGauge fallback
 * when no fit percent is supplied. Kept for that fallback only — the gauge magnitude
 * is normally the score-derived fit, not this. */
export const LEVEL_FRACTION: Record<string, number> = { 高: 1, 中: 0.66, 低: 0.33 };

export function levelFraction(level: string): number {
  return LEVEL_FRACTION[level] ?? 0.5;
}

/** Qualitative band for the score-derived fit percent.
 *
 * Keep these boundaries aligned with the legacy fallback magnitudes above:
 * 0–33=低, 34–66=中, 67–100=高. This label describes the SAME value as the
 * number in the gauge; evidence confidence remains a separate axis (#540).
 */
export function fitLevel(percent: number): "高" | "中" | "低" {
  const clamped = Math.max(0, Math.min(percent, 100));
  if (clamped >= 67) return "高";
  if (clamped >= 34) return "中";
  return "低";
}

export interface FitInput {
  score: number;
  confidence: string;
}

/** Absolute fit percent (0–100) for one composite score, normalised to the fit
 * ceiling and clamped. Independent of the confidence label (#240). */
export function fitPercent(score: number): number {
  return Math.round(Math.max(0, Math.min(score / MAX_COMPOSITE_SCORE, 1)) * 100);
}

/**
 * Absolute fit percents for a list of candidates.
 *
 * Each card reads its own score-derived fit — decoupled from the confidence label,
 * so a strong #1 on a never-asked topic (label 低) still shows a high gauge, while
 * a genuinely weak top pick reads low instead of being overstated as 100%.
 */
export function fitPercents(candidates: readonly FitInput[]): number[] {
  return candidates.map((c) => fitPercent(c.score));
}

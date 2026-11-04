/**
 * Labels and helpers for recommendation `reasons` (the "why this person" chips).
 *
 * Reason `type` values come from the scorer (model-definition): answers / load /
 * recency / question_fit. Certifications, skills and projects stopped being
 * evidence in #95. Unknown types fall back to the raw string so a new scorer
 * signal still renders.
 */

import type { Reason } from "@/lib/api-types";

export const REASON_LABELS: Record<string, string> = {
  answers: "過去回答",
  load: "現在の負荷",
  recency: "直近の活動",
  question_fit: "質問と過去の回答の近さ",
};

export function reasonLabel(type: string): string {
  return REASON_LABELS[type] ?? type;
}

/**
 * The `answers` reason's detail — the past-answer evidence summary the backend
 * provides verbatim (e.g. "類似の質問に過去5件回答（うち有用と評価3件）"). Returns
 * `null` when there is no such reason, so the caller can show a placeholder. The
 * backend does NOT expose a raw reuse count, so we never synthesise one.
 */
export function answersEvidence(reasons: readonly Reason[]): string | null {
  return reasons.find((r) => r.type === "answers")?.detail ?? null;
}

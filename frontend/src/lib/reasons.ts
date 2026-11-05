/**
 * Labels and helpers for recommendation `reasons` (the "why this person" chips).
 *
 * Reason `type` values come from the scorer (model-definition): answers / load /
 * recency / question_fit. Certifications, skills and projects stopped being
 * evidence in #95. An organization's member is recommended from evidence instead
 * (#107): self_intro / document / conversation / answer, whose `detail` already
 * names its source (「『機材の使い方と管理マニュアル』を書いた」). A referral
 * (#518) is `referral`. Unknown types fall back to the raw string so a new scorer
 * signal still renders.
 */

import type { Reason } from "@/lib/api-types";

export const REASON_LABELS: Record<string, string> = {
  answers: "過去回答",
  load: "現在の負荷",
  recency: "直近の活動",
  question_fit: "質問と過去の回答の近さ",
  self_intro: "自己紹介",
  document: "文書",
  conversation: "会話",
  answer: "TEKIJINでの回答",
  referral: "担当者からの推薦",
};

export function reasonLabel(type: string): string {
  return REASON_LABELS[type] ?? type;
}

/**
 * The text shown after a reason's label, or `null` when it would only repeat it.
 * The backend writes a self-introduction as 「自己紹介: …」 so it reads on its own in
 * Discord, where no label is shown; next to the 自己紹介 label the prefix goes.
 */
export function reasonDetail(reason: Reason): string | null {
  const label = reasonLabel(reason.type);
  const detail = reason.detail.trim();
  if (!detail || detail === label) {
    return null;
  }
  for (const separator of [": ", "：", ":"]) {
    if (detail.startsWith(label + separator)) {
      return detail.slice(label.length + separator.length).trim() || null;
    }
  }
  return detail;
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

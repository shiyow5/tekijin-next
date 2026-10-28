/**
 * Small presentation helpers for the processing screen.
 */

/**
 * Render a 0..1 confidence float as a whole-percent string (e.g. 0.85 -> "85%").
 * Values are clamped to the 0..100 range so a stray out-of-range float never
 * shows a nonsensical percentage.
 */
export function formatConfidence(confidence: number): string {
  if (!Number.isFinite(confidence)) {
    return "—";
  }
  const percent = Math.round(confidence * 100);
  const clamped = Math.min(100, Math.max(0, percent));
  return `${clamped}%`;
}

/**
 * Who a source came from, for the 資料 screens (#76). A Discord conversation has
 * many authors and is shown as such; a source without an author was added by the
 * admin account.
 */
export function sourceOrigin(source: { provider: string; author_name: string | null }): string {
  if (source.provider === "discord") return "Discordの会話";
  return `追加: ${source.author_name ?? "管理者"}`;
}

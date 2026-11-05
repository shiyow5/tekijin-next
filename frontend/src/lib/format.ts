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
 * admin account. A shared Google file (#99) is told apart by its link: Docs live
 * under /document/, Slides under /presentation/. A Notion page (#101) is "Notion".
 */
export function sourceOrigin(source: {
  provider: string;
  author_name: string | null;
  url?: string | null;
}): string {
  if (source.provider === "discord") return "Discordの会話";
  if (source.provider === "notion") return "Notion";
  if (source.provider === "google") {
    if (source.url?.includes("docs.google.com/presentation/")) return "Googleスライド";
    if (source.url?.includes("docs.google.com/document/")) return "Googleドキュメント";
    return "Google";
  }
  return `追加: ${source.author_name ?? "管理者"}`;
}

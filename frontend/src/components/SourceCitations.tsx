import type { SourceCitation } from "@/lib/api-types";
import Link from "next/link";

/**
 * 出典チップ列（#291 自己回答の検証性）。`self_answered` / `document` 終端が返す
 * `message.citations` を、ライブの ProcessingScreen とリロード後の ResultScreen の
 * 両方で同一に描画するための共有コンポーネント（片方だけ描画してドリフトする #382
 * レビュー指摘の再発防止）。`document` は内部文書ビューアへ、`qa` はナレッジ詳細
 * （#293 part2, `/knowledge/[id]`）へリンク — `source_id` は自己回答の citation と
 * 同じ実体（`Answer.id`）を指すので、それぞれの詳細ページがそのまま解決できる。
 * `knowledge`（#357）は詳細ページが無いのでラベルチップ。型に無い種類（保存済みの履歴に残る
 * 旧 `daily` など）は、存在しないページへリンクしないよう「出典」のラベルチップにする。
 */
export function SourceCitations({
  citations,
  sessionId,
}: {
  citations?: SourceCitation[];
  sessionId?: string;
}) {
  if (!citations || citations.length === 0) {
    return null;
  }
  return (
    <div className="mt-md">
      <p className="text-xs font-bold text-on-surface-variant">出典</p>
      <ul className="mt-xs flex flex-wrap gap-xs">
        {citations.map((citation) => (
          <li key={`${citation.kind}:${citation.source_id}`}>
            {citation.kind === "source" ? (
              // #81: something the organization ingested; its detail page is /sources.
              <Link
                href={`/sources/${encodeURIComponent(citation.source_id.replace(/^src_/, ""))}`}
                className="inline-flex min-h-[44px] items-center gap-xs rounded-full border border-outline-variant bg-surface px-md py-sm text-sm font-bold text-primary transition-colors hover:bg-surface-container"
              >
                <span aria-hidden="true">📄</span>
                {citation.title || "資料"}
              </Link>
            ) : citation.kind === "knowledge" ? (
              // #357/#366: a structured knowledge unit. No detail page yet (that
              // arrives with #354), so a label chip — but a DISTINCT one: 「過去の回答」
              // would be a false claim about where the answer came from.
              <span className="inline-flex min-h-[44px] items-center gap-xs rounded-full border border-outline-variant bg-surface px-md py-sm font-bold text-on-surface-variant text-sm">
                <span aria-hidden="true">📚</span>
                ナレッジ {citation.source_id}
              </span>
            ) : citation.kind === "document" ? (
              <Link
                href={`/documents/${encodeURIComponent(citation.source_id)}${
                  sessionId ? `?from=${encodeURIComponent(sessionId)}` : ""
                }`}
                className="inline-flex min-h-[44px] items-center gap-xs rounded-full border border-outline-variant bg-surface px-md py-sm text-sm font-bold text-primary transition-colors hover:bg-surface-container"
              >
                <span aria-hidden="true">📄</span>
                {citation.source_id}
              </Link>
            ) : citation.kind === "qa" ? (
              <Link
                href={`/knowledge/${encodeURIComponent(citation.source_id)}`}
                className="inline-flex min-h-[44px] items-center gap-xs rounded-full border border-outline-variant bg-surface px-md py-sm text-sm font-bold text-primary transition-colors hover:bg-surface-container"
              >
                <span aria-hidden="true">💬</span>
                過去の回答 {citation.source_id}
              </Link>
            ) : (
              // A kind the type does not list (e.g. a "daily" citation stored before
              // #91 removed daily reports). It has no page, so never link it.
              <span className="inline-flex min-h-[44px] items-center gap-xs rounded-full border border-outline-variant bg-surface px-md py-sm font-bold text-on-surface-variant text-sm">
                出典 {citation.source_id}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

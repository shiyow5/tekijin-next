"use client";

/**
 * One source's full text (#74), opened from the 資料 list.
 *
 * Shows what was stored — the body as sent, not the chunks it was split into —
 * with who added it and a link to where the original lives. The member who added
 * it (or the admin) can delete it here too; the API enforces the same rule.
 */

import { PageBackLink } from "@/components/PageBackLink";
import { ApiError, deleteSource, getSource } from "@/lib/api-client";
import type { SourceDetail } from "@/lib/api-types";
import { formatDateJst } from "@/lib/datetime";
import { sourceOrigin } from "@/lib/format";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

const NOT_FOUND = "資料が見つかりません。";

type DetailState =
  | { phase: "loading" }
  | { phase: "ready"; source: SourceDetail }
  | { phase: "error"; message: string };

export function SourceDetailScreen({ sourceId }: { sourceId: string }) {
  const router = useRouter();
  const [state, setState] = useState<DetailState>({ phase: "loading" });
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const id = /^\d+$/.test(sourceId) ? Number(sourceId) : null;

  useEffect(() => {
    if (id === null) {
      setState({ phase: "error", message: NOT_FOUND });
      return;
    }
    let active = true;
    getSource(id)
      .then((source) => {
        if (active) setState({ phase: "ready", source });
      })
      .catch((error: unknown) => {
        if (!active) return;
        const notFound = error instanceof ApiError && error.status === 404;
        setState({
          phase: "error",
          message: notFound ? NOT_FOUND : "資料を読み込めませんでした。",
        });
      });
    return () => {
      active = false;
    };
  }, [id]);

  async function remove(source: SourceDetail) {
    if (!window.confirm(`「${source.title ?? "題名なし"}」を削除します。よろしいですか？`)) return;
    setDeleteError(null);
    try {
      await deleteSource(source.id);
      router.push("/sources");
    } catch {
      setDeleteError("削除できませんでした。時間をおいて、もう一度お試しください。");
    }
  }

  return (
    <section className="mx-auto flex w-full max-w-3xl flex-col gap-lg px-md py-lg">
      <PageBackLink
        href="/sources"
        label="資料の一覧へ戻る"
        className="self-center md:self-start"
      />
      {state.phase === "loading" ? (
        <p className="text-on-surface-variant text-sm">読み込み中…</p>
      ) : state.phase === "error" ? (
        <p role="alert" className="text-error text-sm">
          {state.message}
        </p>
      ) : (
        <article className="flex flex-col gap-md rounded-xl border border-outline-variant bg-surface-container-lowest p-md">
          <div className="flex items-start justify-between gap-sm">
            <h1 className="min-w-0 break-words font-bold text-on-surface text-xl">
              {state.source.title ?? "（題名なし）"}
            </h1>
            {state.source.can_delete ? (
              <button
                type="button"
                onClick={() => remove(state.source)}
                className="shrink-0 rounded-md px-sm py-xs text-error text-sm hover:underline"
              >
                削除
              </button>
            ) : null}
          </div>
          <div className="flex flex-wrap items-center gap-x-md gap-y-xs text-on-surface-variant text-xs">
            <span>{sourceOrigin(state.source)}</span>
            <span>更新日: {formatDateJst(state.source.updated_at) ?? "—"}</span>
            <span>断片: {state.source.chunks}</span>
            {state.source.url ? (
              <a
                href={state.source.url}
                target="_blank"
                rel="noopener noreferrer"
                aria-label="元の文書を開く"
                className="text-primary hover:underline"
              >
                元の文書
              </a>
            ) : null}
          </div>
          {deleteError ? (
            <p role="alert" className="text-error text-sm">
              {deleteError}
            </p>
          ) : null}
          <div className="whitespace-pre-wrap break-words border-outline-variant border-t pt-md text-on-surface text-sm leading-relaxed">
            {state.source.body}
          </div>
        </article>
      )}
    </section>
  );
}

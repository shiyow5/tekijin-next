"use client";

/**
 * One knowledge article (#112), opened from the 知識 list or a chat citation.
 *
 * Everyone who may read it sees the question, the answer and where it came from.
 * The organization's admin (`can_review`) also edits the fields and decides:
 * 確認済みにする (it may then answer questions), 差し戻す (back to 作業中), or
 * 引退させる (optionally naming the article that replaces it). The API enforces
 * the same rule; this screen only hides what a member cannot do.
 */

import { PageBackLink } from "@/components/PageBackLink";
import { ApiError, getArticle, patchArticle, reviewArticle } from "@/lib/api-client";
import type { ArticleDetail, ArticleSource, KnowledgeArticle } from "@/lib/api-types";
import { formatDateJst } from "@/lib/datetime";
import { KIND_LABEL, STATE_LABEL } from "@/lib/knowledge";
import Link from "next/link";
import { type FormEvent, useEffect, useState } from "react";

const NOT_FOUND = "知識が見つかりません。";
// Mirrors the API's limit (api/knowledge_routes.py).
const MAX_FIELD = 2000;

type DetailState =
  | { phase: "loading" }
  | { phase: "ready"; article: ArticleDetail }
  | { phase: "error"; message: string };

interface Draft {
  kind: KnowledgeArticle["kind"];
  issue: string;
  environment: string;
  resolution: string;
  cause: string;
}

function draftOf(article: ArticleDetail): Draft {
  return {
    kind: article.kind,
    issue: article.issue,
    environment: article.environment ?? "",
    resolution: article.resolution ?? "",
    cause: article.cause ?? "",
  };
}

function failure(error: unknown, fallback: string): string {
  return error instanceof ApiError && error.message ? error.message : fallback;
}

const FIELDS: { key: keyof Omit<Draft, "kind">; label: string }[] = [
  { key: "issue", label: "問い" },
  { key: "environment", label: "場面" },
  { key: "resolution", label: "答え" },
  { key: "cause", label: "原因" },
];

function SourceEntry({ source }: { source: ArticleSource }) {
  const label =
    source.title ??
    (source.kind === "message"
      ? "Discordの発言"
      : source.kind === "question"
        ? "依頼への回答"
        : "資料");
  const name =
    source.kind === "source" && source.source_item_id != null ? (
      <Link href={`/sources/${source.source_item_id}`} className="text-primary hover:underline">
        {label}
      </Link>
    ) : source.url ? (
      <a
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        className="text-primary hover:underline"
      >
        {label}
      </a>
    ) : (
      <span>{label}</span>
    );
  return (
    <li className="flex flex-col gap-xs text-sm">
      {name}
      {source.quote ? (
        <q className="whitespace-pre-wrap break-words text-on-surface-variant text-xs [quotes:none]">
          「{source.quote}」
        </q>
      ) : null}
    </li>
  );
}

function ReviewPanel({
  article,
  onChange,
}: {
  article: ArticleDetail;
  onChange: (article: ArticleDetail) => void;
}) {
  const [draft, setDraft] = useState<Draft>(() => draftOf(article));
  const [replacement, setReplacement] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run(action: () => Promise<ArticleDetail>, done: string) {
    setBusy(true);
    setNotice(null);
    setError(null);
    try {
      const updated = await action();
      onChange(updated);
      setDraft(draftOf(updated));
      setNotice(done);
    } catch (err) {
      setError(failure(err, "操作できませんでした。時間をおいて、もう一度お試しください。"));
    } finally {
      setBusy(false);
    }
  }

  function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void run(
      () =>
        patchArticle(article.id, {
          kind: draft.kind,
          issue: draft.issue,
          environment: draft.environment,
          resolution: draft.resolution,
          cause: draft.cause,
        }),
      "保存しました。",
    );
  }

  const replacementId = /^\d+$/.test(replacement.trim()) ? Number(replacement.trim()) : null;
  const inputClass =
    "w-full rounded-md border border-outline bg-surface px-sm py-xs text-on-surface text-sm";
  const buttonClass =
    "rounded-lg px-md py-sm font-bold text-sm transition-colors disabled:opacity-50";

  return (
    <div className="flex flex-col gap-md rounded-xl border border-outline-variant bg-surface-container-lowest p-md">
      <h2 className="font-bold text-lg text-on-surface">確認</h2>
      <form onSubmit={save} className="flex flex-col gap-sm">
        <label className="flex flex-col gap-xs text-on-surface-variant text-xs">
          種類
          <select
            value={draft.kind}
            onChange={(e) => setDraft({ ...draft, kind: e.target.value as Draft["kind"] })}
            className={inputClass}
          >
            {Object.entries(KIND_LABEL).map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
        {FIELDS.map(({ key, label }) => (
          <label key={key} className="flex flex-col gap-xs text-on-surface-variant text-xs">
            {label}
            <textarea
              value={draft[key]}
              rows={key === "resolution" ? 5 : 2}
              maxLength={MAX_FIELD}
              onChange={(e) => setDraft({ ...draft, [key]: e.target.value })}
              className={inputClass}
            />
          </label>
        ))}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={busy || draft.issue.trim() === ""}
            className={`${buttonClass} border border-outline text-on-surface hover:bg-surface-container-low`}
          >
            保存する
          </button>
        </div>
      </form>

      <div className="flex flex-wrap items-end gap-sm border-outline-variant border-t pt-md">
        <button
          type="button"
          disabled={busy || !article.resolution || article.state === "verified"}
          onClick={() => run(() => reviewArticle(article.id, "verify", {}), "確認済みにしました。")}
          className={`${buttonClass} bg-primary text-on-primary hover:bg-primary-container`}
        >
          確認済みにする
        </button>
        <button
          type="button"
          disabled={busy || article.state === "wip" || article.state === "retired"}
          onClick={() => run(() => reviewArticle(article.id, "send-back", {}), "差し戻しました。")}
          className={`${buttonClass} border border-outline text-on-surface hover:bg-surface-container-low`}
        >
          差し戻す
        </button>
        <label className="flex flex-col gap-xs text-on-surface-variant text-xs">
          置き換え先の知識の番号（任意）
          <input
            value={replacement}
            inputMode="numeric"
            onChange={(e) => setReplacement(e.target.value)}
            className={`${inputClass} w-32`}
          />
        </label>
        <button
          type="button"
          disabled={busy || article.state === "retired"}
          onClick={() =>
            run(
              () =>
                reviewArticle(
                  article.id,
                  "retire",
                  replacementId !== null ? { supersededBy: replacementId } : {},
                ),
              "引退させました。",
            )
          }
          className={`${buttonClass} text-error hover:underline`}
        >
          引退させる
        </button>
      </div>
      {!article.resolution ? (
        <p className="text-on-surface-variant text-xs">答えが空のあいだは確認済みにできません。</p>
      ) : null}
      {notice ? <output className="text-on-surface text-sm">{notice}</output> : null}
      {error ? (
        <p role="alert" className="text-error text-sm">
          {error}
        </p>
      ) : null}
    </div>
  );
}

export function ArticleDetailScreen({ articleId }: { articleId: string }) {
  const [state, setState] = useState<DetailState>({ phase: "loading" });
  const id = /^\d+$/.test(articleId) ? Number(articleId) : null;

  useEffect(() => {
    if (id === null) {
      setState({ phase: "error", message: NOT_FOUND });
      return;
    }
    let active = true;
    getArticle(id)
      .then((article) => {
        if (active) setState({ phase: "ready", article });
      })
      .catch((error: unknown) => {
        if (!active) return;
        const notFound = error instanceof ApiError && error.status === 404;
        setState({
          phase: "error",
          message: notFound ? NOT_FOUND : "知識を読み込めませんでした。",
        });
      });
    return () => {
      active = false;
    };
  }, [id]);

  return (
    <section className="mx-auto flex w-full max-w-3xl flex-col gap-lg px-md py-lg">
      <PageBackLink
        href="/articles"
        label="知識の一覧へ戻る"
        className="self-center md:self-start"
      />
      {state.phase === "loading" ? (
        <p className="text-on-surface-variant text-sm">読み込み中…</p>
      ) : state.phase === "error" ? (
        <p role="alert" className="text-error text-sm">
          {state.message}
        </p>
      ) : (
        <>
          <article className="flex flex-col gap-md rounded-xl border border-outline-variant bg-surface-container-lowest p-md">
            <div className="flex flex-wrap items-center gap-xs text-xs">
              <span className="rounded-full bg-surface-container px-sm py-xs text-on-surface-variant">
                {KIND_LABEL[state.article.kind]}
              </span>
              <span className="rounded-full bg-surface-container px-sm py-xs text-on-surface-variant">
                {STATE_LABEL[state.article.state]}
              </span>
            </div>
            <h1 className="min-w-0 break-words font-bold text-on-surface text-xl">
              {state.article.issue}
            </h1>
            {state.article.review_note ? (
              <p
                role="note"
                className="rounded-lg bg-surface-container px-sm py-xs text-on-surface text-sm"
              >
                確認に戻った理由: {state.article.review_note}
              </p>
            ) : null}
            <dl className="flex flex-col gap-sm text-sm">
              {(
                [
                  ["答え", state.article.resolution],
                  ["場面", state.article.environment],
                  ["原因", state.article.cause],
                ] as const
              ).map(([label, value]) =>
                value ? (
                  <div key={label} className="flex flex-col gap-xs">
                    <dt className="font-bold text-on-surface-variant text-xs">{label}</dt>
                    <dd className="whitespace-pre-wrap break-words text-on-surface">{value}</dd>
                  </div>
                ) : null,
              )}
            </dl>
            <div className="flex flex-wrap items-center gap-x-md gap-y-xs text-on-surface-variant text-xs">
              <span>回答に使われた回数: {state.article.reuse_count}</span>
              {state.article.author_name ? <span>作成: {state.article.author_name}</span> : null}
              {state.article.verified_at ? (
                <span>
                  確認: {state.article.verifier_name ?? "管理者"}（
                  {formatDateJst(state.article.verified_at) ?? "—"}）
                </span>
              ) : null}
              {state.article.superseded_by ? (
                <Link
                  href={`/articles/${state.article.superseded_by}`}
                  className="text-primary hover:underline"
                >
                  置き換え先の知識
                </Link>
              ) : null}
            </div>
            {state.article.sources.length > 0 ? (
              <div className="flex flex-col gap-xs border-outline-variant border-t pt-md">
                <h2 className="font-bold text-on-surface-variant text-xs">出典</h2>
                <ul className="flex flex-col gap-sm">
                  {state.article.sources.map((source, index) => (
                    // Sources have no id of their own on the wire; their order is stable.
                    // biome-ignore lint/suspicious/noArrayIndexKey: stable server order
                    <SourceEntry key={index} source={source} />
                  ))}
                </ul>
              </div>
            ) : null}
          </article>
          {state.article.can_review ? (
            <ReviewPanel
              article={state.article}
              onChange={(article) => setState({ phase: "ready", article })}
            />
          ) : null}
        </>
      )}
    </section>
  );
}

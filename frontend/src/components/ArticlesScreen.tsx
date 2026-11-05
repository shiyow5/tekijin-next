"use client";

/**
 * 知識の画面 (#112, R6): the organization's verified knowledge, and for the
 * admin, the drafts waiting to be checked.
 *
 * Every member reads verified knowledge. The admin (`can_review` from the API)
 * also gets a 確認待ち tab listing drafts in the order the API gives — the ones
 * questions would have used most come first. Opening an article leads to its page,
 * where the admin edits and decides.
 */

import { PageBackLink } from "@/components/PageBackLink";
import { ApiError, getArticles } from "@/lib/api-client";
import type { KnowledgeArticle } from "@/lib/api-types";
import { formatDateJst } from "@/lib/datetime";
import { KIND_LABEL, STATE_LABEL } from "@/lib/knowledge";
import Link from "next/link";
import { type FormEvent, useCallback, useEffect, useState } from "react";

const PAGE_SIZE = 50;

type View = "verified" | "review";

type ListState =
  | { phase: "loading" }
  | { phase: "ready"; items: KnowledgeArticle[]; total: number }
  | { phase: "error"; message: string };

function ArticleRow({ article, view }: { article: KnowledgeArticle; view: View }) {
  return (
    <li className="flex flex-col gap-xs rounded-xl border border-outline-variant bg-surface-container-lowest p-md">
      <div className="flex flex-wrap items-center gap-xs text-xs">
        <span className="rounded-full bg-surface-container px-sm py-xs text-on-surface-variant">
          {KIND_LABEL[article.kind]}
        </span>
        {view === "review" ? (
          <span className="rounded-full bg-surface-container px-sm py-xs text-on-surface-variant">
            {STATE_LABEL[article.state]}
          </span>
        ) : null}
      </div>
      <h3 className="min-w-0 break-words font-bold text-base text-on-surface">
        <Link href={`/articles/${article.id}`} className="hover:underline">
          {article.issue}
        </Link>
      </h3>
      {article.resolution ? (
        <p className="line-clamp-3 whitespace-pre-wrap break-words text-on-surface text-sm">
          {article.resolution}
        </p>
      ) : (
        <p className="text-on-surface-variant text-sm">（答えがまだありません）</p>
      )}
      <div className="flex flex-wrap items-center gap-x-md gap-y-xs text-on-surface-variant text-xs">
        {view === "review" ? (
          <span>この知識で答えられた質問: {article.demand_count}</span>
        ) : (
          <span>回答に使われた回数: {article.reuse_count}</span>
        )}
        <span>更新日: {formatDateJst(article.updated_at) ?? "—"}</span>
        {article.author_name ? <span>作成: {article.author_name}</span> : null}
      </div>
    </li>
  );
}

export function ArticlesScreen() {
  const [view, setView] = useState<View>("verified");
  const [query, setQuery] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [list, setList] = useState<ListState>({ phase: "loading" });
  const [canReview, setCanReview] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);

  const load = useCallback((nextView: View, q: string) => {
    setList({ phase: "loading" });
    return getArticles({ view: nextView, q: q || undefined, limit: PAGE_SIZE })
      .then(({ items, total, can_review }) => {
        setCanReview(can_review);
        setList({ phase: "ready", items, total });
      })
      .catch((error: unknown) =>
        setList({
          phase: "error",
          message:
            error instanceof ApiError && error.status === 403
              ? error.message
              : "知識の一覧を読み込めませんでした。",
        }),
      );
  }, []);

  useEffect(() => {
    void load(view, submitted);
  }, [load, view, submitted]);

  function search(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitted(query.trim());
  }

  async function loadMore() {
    if (list.phase !== "ready") return;
    setLoadingMore(true);
    try {
      const page = await getArticles({
        view,
        q: submitted || undefined,
        offset: list.items.length,
        limit: PAGE_SIZE,
      });
      setList({ phase: "ready", items: [...list.items, ...page.items], total: page.total });
    } catch {
      setList({ phase: "error", message: "続きを読み込めませんでした。" });
    } finally {
      setLoadingMore(false);
    }
  }

  const tabClass = (active: boolean) =>
    `rounded-md px-md py-xs text-sm font-bold ${
      active ? "bg-primary text-on-primary" : "text-on-surface hover:bg-surface-container-low"
    }`;
  const empty =
    view === "review" ? "確認を待っている下書きはありません。" : "まだ確認済みの知識がありません。";

  return (
    <section className="mx-auto flex w-full max-w-3xl flex-col gap-lg px-md py-lg">
      <PageBackLink href="/" label="ホームへ戻る" className="self-center md:self-start" />
      <h1 className="text-center font-bold text-2xl text-on-surface">知識</h1>
      <p className="text-on-surface-variant text-sm">
        組織の管理者が確認した知識です。TEKIJINはこの知識を使って質問に答えます。
      </p>

      {canReview ? (
        <div role="tablist" aria-label="知識の種類" className="flex gap-xs">
          <button
            type="button"
            role="tab"
            aria-selected={view === "verified"}
            onClick={() => setView("verified")}
            className={tabClass(view === "verified")}
          >
            確認済み
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={view === "review"}
            onClick={() => setView("review")}
            className={tabClass(view === "review")}
          >
            確認待ち
          </button>
        </div>
      ) : null}

      <form onSubmit={search} className="flex gap-sm">
        <label className="sr-only" htmlFor="article-search">
          知識を探す
        </label>
        <input
          id="article-search"
          type="search"
          value={query}
          maxLength={200}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="ことばで探す"
          className="min-w-0 flex-1 rounded-md border border-outline bg-surface px-sm py-xs text-on-surface text-sm"
        />
        <button
          type="submit"
          className="rounded-lg bg-primary px-md py-xs font-bold text-on-primary text-sm"
        >
          探す
        </button>
      </form>

      {list.phase === "loading" ? (
        <p className="text-on-surface-variant text-sm">読み込み中…</p>
      ) : list.phase === "error" ? (
        <p role="alert" className="text-error text-sm">
          {list.message}
        </p>
      ) : list.items.length === 0 ? (
        <p className="text-on-surface-variant text-sm">{empty}</p>
      ) : (
        <>
          <ul className="flex flex-col gap-sm">
            {list.items.map((article) => (
              <ArticleRow key={article.id} article={article} view={view} />
            ))}
          </ul>
          {list.items.length < list.total ? (
            <button
              type="button"
              onClick={loadMore}
              disabled={loadingMore}
              className="self-center rounded-md border border-outline px-md py-xs text-on-surface text-sm transition-colors hover:bg-surface-container-low disabled:opacity-50"
            >
              もっと見る（残り{list.total - list.items.length}件）
            </button>
          ) : null}
        </>
      )}
    </section>
  );
}

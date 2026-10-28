"use client";

/**
 * 資料の画面 (#72, R2): the organization's documents, and a form to add one.
 *
 * Everything TEKIJIN ingests — a note sent with `/tekijin add` in Discord, or a
 * document added here — is listed together, newest first. A document longer than
 * Discord's 4000-character option goes in here, pasted or read from a text /
 * Markdown file. With a URL, re-adding the same URL updates the document instead
 * of adding a copy (the backend decides; this screen only reports what happened).
 *
 * Only the member who added a document (or the admin) sees its delete button; the
 * API enforces the same rule.
 */

import { PageBackLink } from "@/components/PageBackLink";
import { ApiError, deleteSource, getSources, postSource } from "@/lib/api-client";
import type { SourceCreateResponse, SourceItem } from "@/lib/api-types";
import { formatDateJst } from "@/lib/datetime";
import { type ChangeEvent, type FormEvent, useCallback, useEffect, useState } from "react";

// Mirrors the API's limit (api/source_routes.py).
const MAX_BODY_LENGTH = 20000;

type ListState =
  | { phase: "loading" }
  | { phase: "ready"; items: SourceItem[] }
  | { phase: "error"; message: string };

interface Draft {
  title: string;
  url: string;
  body: string;
}

const EMPTY_DRAFT: Draft = { title: "", url: "", body: "" };

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof ApiError && error.message ? error.message : fallback;
}

function addedMessage(result: SourceCreateResponse): string {
  if (result.status === "unchanged") {
    return "同じ内容がすでに登録されています。中身は変わっていないので、そのままにしました。";
  }
  const verb = result.status === "added" ? "追加しました" : "更新しました";
  return `${verb}（${result.chunks}個の断片に分けて保存）。`;
}

function readFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ""));
    reader.onerror = () => reject(reader.error);
    reader.readAsText(file);
  });
}

function SourceRow({
  item,
  onDelete,
}: {
  item: SourceItem;
  onDelete: (item: SourceItem) => void;
}) {
  const title = item.title ?? "（題名なし）";
  return (
    <li className="flex flex-col gap-xs rounded-xl border border-outline-variant bg-surface-container-lowest p-md">
      <div className="flex items-start justify-between gap-sm">
        <h3 className="min-w-0 break-words font-bold text-base text-on-surface">{title}</h3>
        {item.can_delete ? (
          <button
            type="button"
            onClick={() => onDelete(item)}
            aria-label={`「${title}」を削除`}
            className="shrink-0 rounded-md px-sm py-xs text-error text-sm hover:underline"
          >
            削除
          </button>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center gap-x-md gap-y-xs text-on-surface-variant text-xs">
        <span>追加: {item.author_name ?? "管理者"}</span>
        <span>更新日: {formatDateJst(item.updated_at) ?? "—"}</span>
        <span>断片: {item.chunks}</span>
        {item.url ? (
          <a
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            aria-label="元の文書を開く"
            className="text-primary hover:underline"
          >
            元の文書
          </a>
        ) : null}
      </div>
    </li>
  );
}

export function SourcesScreen() {
  const [list, setList] = useState<ListState>({ phase: "loading" });
  const [draft, setDraft] = useState<Draft>(EMPTY_DRAFT);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);

  const reload = useCallback(() => {
    return getSources()
      .then((items) => setList({ phase: "ready", items }))
      .catch((error: unknown) =>
        setList({
          phase: "error",
          message: errorMessage(error, "資料の一覧を読み込めませんでした。"),
        }),
      );
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const tooLong = draft.body.length > MAX_BODY_LENGTH;
  const canSubmit = draft.body.trim() !== "" && !tooLong && !submitting;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setNotice(null);
    setFormError(null);
    try {
      const result = await postSource({
        title: draft.title.trim() || null,
        url: draft.url.trim() || null,
        body: draft.body,
      });
      setNotice(addedMessage(result));
      setDraft(EMPTY_DRAFT);
      await reload();
    } catch (error) {
      setFormError(
        errorMessage(error, "追加できませんでした。時間をおいて、もう一度お試しください。"),
      );
    } finally {
      setSubmitting(false);
    }
  }

  async function loadFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const body = await readFile(file);
    const name = file.name.replace(/\.[^.]+$/, "");
    setDraft((prev) => ({ ...prev, body, title: prev.title || name }));
  }

  async function remove(item: SourceItem) {
    if (!window.confirm(`「${item.title ?? "題名なし"}」を削除します。よろしいですか？`)) return;
    setNotice(null);
    setFormError(null);
    try {
      await deleteSource(item.id);
      await reload();
    } catch (error) {
      setFormError(
        errorMessage(error, "削除できませんでした。時間をおいて、もう一度お試しください。"),
      );
    }
  }

  const inputClass =
    "w-full rounded-md border border-outline bg-surface px-sm py-xs text-on-surface text-sm";

  return (
    <section className="mx-auto flex w-full max-w-3xl flex-col gap-lg px-md py-lg">
      <PageBackLink href="/" label="ホームへ戻る" className="self-center md:self-start" />
      <h1 className="text-center font-bold text-2xl text-on-surface">資料</h1>
      <p className="text-on-surface-variant text-sm">
        マニュアルや引継ぎ書などを追加すると、TEKIJINが質問に答えるときの材料になります。Discordの
        <code className="mx-xs">/tekijin add</code>
        で追加したものも、ここに並びます。
      </p>

      <form
        onSubmit={submit}
        className="flex flex-col gap-sm rounded-xl border border-outline-variant bg-surface-container-lowest p-md"
      >
        <label className="flex flex-col gap-xs text-on-surface-variant text-xs">
          題名（任意）
          <input
            value={draft.title}
            maxLength={200}
            onChange={(e) => setDraft((p) => ({ ...p, title: e.target.value }))}
            className={inputClass}
          />
        </label>
        <label className="flex flex-col gap-xs text-on-surface-variant text-xs">
          置き場所のURL（任意）
          <input
            type="url"
            value={draft.url}
            maxLength={2000}
            placeholder="https://"
            onChange={(e) => setDraft((p) => ({ ...p, url: e.target.value }))}
            className={inputClass}
          />
        </label>
        <label className="flex flex-col gap-xs text-on-surface-variant text-xs">
          本文
          <textarea
            value={draft.body}
            rows={10}
            onChange={(e) => setDraft((p) => ({ ...p, body: e.target.value }))}
            className={`${inputClass} font-mono`}
          />
        </label>
        <div className="flex flex-wrap items-center justify-between gap-sm text-xs">
          <label className="flex items-center gap-xs text-on-surface-variant">
            ファイルから読み込む
            <input
              type="file"
              accept=".txt,.md,text/plain,text/markdown"
              onChange={loadFile}
              className="text-xs"
            />
          </label>
          <span className={tooLong ? "text-error" : "text-on-surface-variant"}>
            {draft.body.length.toLocaleString()} / {MAX_BODY_LENGTH.toLocaleString()}字
          </span>
        </div>
        {tooLong ? (
          <p className="text-error text-sm">
            本文が長すぎます。2万字までに分けて追加してください。
          </p>
        ) : null}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={!canSubmit}
            className="rounded-lg bg-primary px-md py-sm font-bold text-on-primary text-sm transition-colors hover:bg-primary-container disabled:opacity-50"
          >
            追加する
          </button>
        </div>
        {notice ? <output className="text-on-surface text-sm">{notice}</output> : null}
        {formError ? (
          <p role="alert" className="text-error text-sm">
            {formError}
          </p>
        ) : null}
      </form>

      <h2 className="font-bold text-lg text-on-surface">組織の資料</h2>
      {list.phase === "loading" ? (
        <p className="text-on-surface-variant text-sm">読み込み中…</p>
      ) : list.phase === "error" ? (
        <p role="alert" className="text-error text-sm">
          {list.message}
        </p>
      ) : list.items.length === 0 ? (
        <p className="text-on-surface-variant text-sm">まだ資料がありません。</p>
      ) : (
        <ul className="flex flex-col gap-sm">
          {list.items.map((item) => (
            <SourceRow key={item.id} item={item} onDelete={remove} />
          ))}
        </ul>
      )}
    </section>
  );
}

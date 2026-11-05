"use client";

/**
 * Answer screen (product-spec 画面4 / ui_template _3) — the "asked" side.
 *
 * The responder receives the question already filled in with the asker's context
 * and the reasons they were chosen, plus the generated draft (下書き機能の受益者).
 * Three equal-size choices keep declining a first-class option (F-09): 引き受ける /
 * 今は難しい / 自分より適任がいる. Accepting connects the asker to this person, so the
 * copy says "引き受ける"/"お繋ぎします", never "回答をお届け" (#176). On accept the
 * responder may ALSO type an answer (optional), captured as a reusable ``answers``
 * row so the accumulation loop closes (#274). The reuse count at the bottom is the
 * 見返り (F-13).
 *
 * Data + actions come from {@link useHandoff}; this component is presentation and
 * wiring only. A `sessionId` is the single input (from the route param).
 */

import { ConsultMethodBadge } from "@/components/ConsultMethodBadge";
import { PageBackLink } from "@/components/PageBackLink";
import type { HandoffAction } from "@/hooks/useHandoff";
import { useHandoff } from "@/hooks/useHandoff";
import type { HandoffResponse, Reason } from "@/lib/api-types";
import { reasonDetail, reasonLabel } from "@/lib/reasons";
import Link from "next/link";
import { useEffect, useState } from "react";

export interface AnswerScreenProps {
  sessionId: string;
  /** Show the route-level return link for a standalone deep link. The inbox's
   * embedded detail pane has its own responsive list navigation. */
  showBackLink?: boolean;
  /** Fired once the responder's outcome lands, so an embedding page (the inbox's
   * merged list+detail view) can refresh its list without a full navigation. */
  onDone?: (action: HandoffAction) => void;
}

// One equal-size contract shared by all three actions, so declining is never a
// second-class choice (product-spec 画面4). Only the color weight differs.
const BASE_BTN =
  "flex-1 min-h-[56px] rounded-xl px-md py-sm font-bold text-base shadow-sm transition-colors disabled:cursor-not-allowed disabled:opacity-50";
const PRIMARY_BTN = `${BASE_BTN} bg-primary text-on-primary hover:bg-primary-container`;
const DECLINE_BTN = `${BASE_BTN} border border-outline text-on-surface hover:bg-surface-container-low`;

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <section className="mx-auto flex w-full max-w-2xl flex-col gap-md py-lg text-center">
      {children}
    </section>
  );
}

function BackLink() {
  // The responder came from their inbox, so return there (label matches the
  // destination — #126). Client-side navigation via next/link.
  return (
    <div className="flex justify-center">
      <Link
        href="/inbox"
        className="min-h-[48px] rounded-full bg-primary px-lg py-sm font-bold text-on-primary shadow-md transition-colors hover:bg-primary-container"
      >
        受信箱へ戻る
      </Link>
    </div>
  );
}

function ChatCta({ href, label }: { href: string; label: string }) {
  return (
    <div className="flex justify-center">
      <Link
        href={href}
        className="min-h-[48px] rounded-full border border-primary px-lg py-sm font-bold text-primary shadow-sm transition-colors hover:bg-primary-container hover:text-on-primary-container"
      >
        {label}
      </Link>
    </div>
  );
}

function ReasonList({ reasons }: { reasons: Reason[] }) {
  if (reasons.length === 0) {
    return <p className="text-on-surface-variant text-sm">根拠を確認中…</p>;
  }
  return (
    <ul className="flex flex-col gap-xs">
      {reasons.map((reason) => (
        <li
          key={`${reason.type}-${reason.detail}`}
          className="flex items-start gap-xs text-on-surface-variant text-sm"
        >
          <span aria-hidden="true" className="text-primary">
            ✓
          </span>
          <span>
            <span className="font-medium text-on-surface">{reasonLabel(reason.type)}</span>
            {reasonDetail(reason) ? `：${reasonDetail(reason)}` : null}
          </span>
        </li>
      ))}
    </ul>
  );
}

const DONE_HEADING: Record<HandoffAction, string> = {
  answer: "お引き受けありがとうございます",
  defer: "承知しました",
  refer: "承知しました",
};

// The app captures the accept/decline, not the answer text itself, so the copy
// must not promise a delivered answer. "引き受ける" connects the asker to this
// person; "今は難しい"/"自分より適任がいる" both reroute to the next candidate
// automatically (a named referral is #76) (#176).
const DONE_BODY: Record<HandoffAction, string> = {
  answer: "質問者にあなたをお繋ぎします。この後、直接ご回答ください。ご協力ありがとうございます。",
  defer: "別の候補者を自動でお探しします。無理のない範囲でご協力ください。",
  refer: "別の候補者を自動でお探しします。ご対応ありがとうございました。",
};

export function AnswerScreen({ sessionId, showBackLink = false, onDone }: AnswerScreenProps) {
  const { phase, handoff, action, errorKind, submitError, submit } = useHandoff(sessionId);
  // Optional answer text, captured on accept as reusable knowledge (#274). Empty
  // is fine — accepting without it records the hand-off but no answers row.
  const [answerBody, setAnswerBody] = useState("");

  // onDone fires once per completed outcome (keyed on phase/action); an
  // identity change in the caller's callback must not re-fire it.
  // biome-ignore lint/correctness/useExhaustiveDependencies: <explained above>
  useEffect(() => {
    if (phase === "done" && action) onDone?.(action);
  }, [phase, action]);

  if (phase === "loading") {
    return (
      <Centered>
        {showBackLink ? <PageBackLink href="/inbox" label="受信箱へ戻る" /> : null}
        <h1 className="font-bold text-2xl text-on-surface">質問を読み込み中…</h1>
        <p className="text-on-surface-variant">届いた質問の内容を取得しています。</p>
      </Centered>
    );
  }

  if (phase === "error") {
    const message =
      errorKind === "gone"
        ? "この依頼はすでに対応が完了したか、受付を終了しています。"
        : "質問の読み込みに失敗しました。時間をおいて再度お試しください。";
    return (
      <Centered>
        <h1 className="font-bold text-2xl text-on-surface">表示できませんでした</h1>
        <div
          role="alert"
          className="rounded-xl border border-outline-variant bg-surface-container p-md text-on-surface-variant"
        >
          {message}
        </div>
        {/* "gone" is this issue's other gap: reloading after an accept 404s here
            with no way back to the conversation. The chat list finds it without
            needing the (now-consumed) session id (#224). */}
        {errorKind === "gone" ? <ChatCta href="/chat" label="チャット一覧を見る" /> : null}
        {showBackLink ? <BackLink /> : null}
      </Centered>
    );
  }

  if (phase === "done" && action) {
    return (
      <Centered>
        <h1 className="font-bold text-2xl text-primary">{DONE_HEADING[action]}</h1>
        <p className="text-on-surface-variant">{DONE_BODY[action]}</p>
        {action === "answer" &&
        handoff?.recommendation_id != null &&
        handoff.consult_method !== "direct" ? (
          <ChatCta href={`/chat?thread=${handoff.recommendation_id}`} label="チャットを開く" />
        ) : null}
        {showBackLink ? <BackLink /> : null}
      </Centered>
    );
  }

  // ready / submitting: the handoff is loaded.
  const hf = handoff as HandoffResponse;
  const submitting = phase === "submitting";
  const askerName = hf.asker.name ?? "質問者";
  const askerMeta = [hf.asker.dept, hf.asker.id].filter(Boolean).join(" / ");
  const slots = [
    hf.products.length > 0 ? `【製品】${hf.products.join("・")}` : null,
    hf.topics.length > 0 ? `【分野】${hf.topics.join("・")}` : null,
    hf.situation ? `【状況】${hf.situation}` : null,
    hf.missing.length > 0 ? `【未確認】${hf.missing.join("・")}` : null,
  ].filter(Boolean);

  return (
    <section className="mx-auto flex w-full max-w-2xl flex-col gap-md py-lg">
      {showBackLink ? <PageBackLink href="/inbox" label="受信箱へ戻る" /> : null}
      <header className="flex items-baseline justify-between gap-sm">
        <h1 className="font-bold text-2xl text-on-surface">あなたに届いた質問</h1>
        {hf.responder ? (
          <span className="text-on-surface-variant text-sm">{hf.responder.name}</span>
        ) : null}
      </header>

      <article className="flex flex-col gap-sm rounded-xl border border-outline-variant bg-surface-container-lowest p-md shadow-sm">
        <div className="flex flex-wrap items-center gap-sm text-on-surface-variant text-sm">
          <span className="font-medium text-on-surface">{askerName}さん</span>
          {askerMeta ? <span>（{askerMeta}）</span> : null}
          {/* Shown BEFORE the accept/decline buttons: "直接相談" never opens a
              chat, so it changes what accepting commits you to (#245). */}
          <ConsultMethodBadge method={hf.consult_method ?? "chat"} />
        </div>
        <h2 className="font-bold text-lg text-on-surface leading-snug">{hf.question}</h2>
        {slots.length > 0 ? (
          <ul className="flex flex-wrap gap-x-md gap-y-xs text-on-surface-variant text-sm">
            {slots.map((slot) => (
              <li key={slot}>{slot}</li>
            ))}
          </ul>
        ) : null}
      </article>

      <section className="flex flex-col gap-xs">
        <h2 className="font-bold text-on-surface text-sm">あなたが選ばれた理由</h2>
        <ReasonList reasons={hf.responder?.reasons ?? []} />
      </section>

      <section className="flex flex-col gap-xs">
        <h2 className="font-bold text-on-surface text-sm">依頼内容（下書き）</h2>
        <p className="whitespace-pre-wrap rounded-xl border border-outline-variant bg-surface-container-low p-md text-on-surface text-sm">
          {hf.draft || "（下書きは準備中です）"}
        </p>
      </section>

      <section className="flex flex-col gap-xs">
        <label htmlFor="answer-body" className="font-bold text-on-surface text-sm">
          回答内容（任意）
        </label>
        <textarea
          id="answer-body"
          value={answerBody}
          onChange={(e) => setAnswerBody(e.target.value)}
          disabled={submitting}
          rows={4}
          placeholder="ここに回答を書くと、今後の似た質問に再利用できる知識として蓄積されます。空欄のまま引き受けても構いません。"
          className="w-full resize-y rounded-xl border border-outline-variant bg-surface-container-lowest p-md text-on-surface text-sm placeholder:text-on-surface-variant disabled:cursor-not-allowed disabled:opacity-50"
        />
      </section>

      {submitError ? (
        <p role="alert" className="text-error text-sm">
          {submitError}
        </p>
      ) : null}

      <div className="flex flex-col gap-sm sm:flex-row">
        <button
          type="button"
          className={PRIMARY_BTN}
          disabled={submitting}
          onClick={() => submit("answer", answerBody)}
        >
          引き受ける
        </button>
        <button
          type="button"
          className={DECLINE_BTN}
          disabled={submitting}
          onClick={() => submit("defer")}
        >
          今は難しい
        </button>
        <button
          type="button"
          className={DECLINE_BTN}
          disabled={submitting}
          onClick={() => submit("refer")}
        >
          自分より適任がいる
        </button>
      </div>

      <p className="text-center text-on-surface-variant text-sm">
        あなたの回答は、これまで <span className="font-bold text-on-surface">{hf.reuse_count}</span>{" "}
        件再利用されています
        {hf.helpful_answer_count > 0 ? `（有用評価 ${hf.helpful_answer_count} 件）` : null}。
      </p>
    </section>
  );
}

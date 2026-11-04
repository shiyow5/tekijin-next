"use client";

/**
 * 直接相談のふりかえりフォーム (#247).
 *
 * A "direct" consultation (#245) happens face to face, so — unlike a chat
 * hand-off — nothing is written down and the knowledge it carried is lost. This
 * form is the asker's write-up of it, and what it records becomes topic evidence
 * for the responder (`base_score` well below a past answer, because it is the
 * asker paraphrasing someone else).
 *
 * The topic list comes from `GET /topics` rather than a copy in this file: the
 * C6 scorer JOINs on those exact strings, so a drifted local list would silently
 * produce evidence that matches nothing (#116).
 */

import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getTopics, postConsultRetrospective } from "@/lib/api-client";
import type { ConsultResolution } from "@/lib/api-types";

const RESOLUTIONS: { value: ConsultResolution; label: string }[] = [
  { value: "resolved", label: "解決した" },
  { value: "partial", label: "部分的に解決した" },
  { value: "unresolved", label: "解決しなかった" },
];

export interface RetrospectiveFormProps {
  questionId: string;
  /** The person who answered, in the external "E###" form. */
  responderId: string;
  responderName?: string | null;
  /** Called after a successful submission (e.g. to refresh the parent view). */
  onRecorded?: () => void;
}

export function RetrospectiveForm({
  questionId,
  responderId,
  responderName,
  onRecorded,
}: RetrospectiveFormProps) {
  const [topics, setTopics] = useState<string[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [asked, setAsked] = useState("");
  const [answer, setAnswer] = useState("");
  const [resolution, setResolution] = useState<ConsultResolution>("resolved");
  const [submitting, setSubmitting] = useState(false);
  const [recorded, setRecorded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Guards a double submit within one tick: `submitting` only disables the button
  // on the NEXT render, so two clicks in the same batch both pass `canSubmit`. The
  // API answers the second one 409 (one write-up per question), so without this the
  // user would see a failure message for a write that actually succeeded.
  const inFlight = useRef(false);

  useEffect(() => {
    let cancelled = false;
    getTopics()
      .then((list) => {
        if (!cancelled) {
          setTopics(list);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError("トピック一覧を取得できませんでした。時間をおいて開き直してください。");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const toggleTopic = useCallback((topic: string) => {
    setSelected((current) =>
      current.includes(topic) ? current.filter((t) => t !== topic) : [...current, topic],
    );
  }, []);

  const trimmedAnswer = answer.trim();
  const canSubmit = selected.length > 0 && trimmedAnswer.length > 0 && !submitting;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canSubmit || inFlight.current) {
      return;
    }
    inFlight.current = true;
    setSubmitting(true);
    setError(null);
    try {
      await postConsultRetrospective({
        question_id: questionId,
        responder_id: responderId,
        topics: selected,
        // Optional field (#247 の項目2): send null rather than "" so the column
        // records "not written", not "written as empty".
        asked: asked.trim() || null,
        answer_body: trimmedAnswer,
        resolution,
      });
      setRecorded(true);
      onRecorded?.();
    } catch (err) {
      // Deliberately keep every field: this is a recollection of a conversation
      // that exists nowhere else, so losing it to a transient failure is the one
      // outcome this form must never produce.
      //
      // Two of these are NOT transient, so telling the user to retry would send
      // them at something that can never succeed: 503 = the feature is switched
      // off (the kill switch), 409 = this consultation is already written up.
      const status = err instanceof ApiError ? err.status : undefined;
      if (status === 503) {
        setError("ふりかえりの記録は現在停止しています。時間をおいて開き直してください。");
      } else if (status === 409) {
        setError("この相談のふりかえりは、すでに記録されています。");
      } else {
        setError("記録できませんでした。もう一度お試しください。");
      }
    } finally {
      inFlight.current = false;
      setSubmitting(false);
    }
  }

  if (recorded) {
    return (
      <div className="w-full rounded-xl border border-primary-fixed bg-surface-container-low p-md">
        <p className="font-bold text-on-surface">ふりかえりを記録しました</p>
        <p className="mt-xs text-on-surface-variant text-sm">
          次に同じことで困った人が、この内容から適任者にたどり着けるようになります。
        </p>
      </div>
    );
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="w-full rounded-xl border border-primary-fixed bg-surface-container-low p-md"
    >
      <p className="font-bold text-on-surface">直接相談のふりかえり</p>
      <p className="mt-xs text-on-surface-variant text-sm">
        {responderName ? `${responderName}さんに相談した内容を残してください。` : null}
        対面での相談は記録が残らないため、ここに書いた内容だけが次の人に引き継がれます。
      </p>

      <fieldset className="mt-md">
        <legend className="font-bold text-on-surface text-sm">
          相談内容（トピック）<span className="text-error">*</span>
        </legend>
        <div className="mt-sm flex flex-wrap gap-xs">
          {topics.map((topic) => {
            const isOn = selected.includes(topic);
            return (
              <button
                key={topic}
                type="button"
                aria-pressed={isOn}
                onClick={() => toggleTopic(topic)}
                className={`min-h-[36px] rounded-full border px-sm py-[6px] text-sm transition-colors ${
                  isOn
                    ? "border-primary bg-primary text-on-primary"
                    : "border-outline-variant bg-surface-container-lowest text-on-surface-variant hover:border-primary"
                }`}
              >
                {topic}
              </button>
            );
          })}
        </div>
      </fieldset>

      <label className="mt-md block font-bold text-on-surface text-sm" htmlFor="retro-asked">
        何を聞いたか（任意）
      </label>
      <textarea
        id="retro-asked"
        value={asked}
        onChange={(e) => setAsked(e.target.value)}
        rows={2}
        placeholder="依頼文に書ききれなかった前提など"
        className="mt-xs w-full rounded-lg border border-outline-variant bg-surface-container-lowest px-sm py-2 text-on-surface outline-none focus:border-primary"
      />

      <label className="mt-md block font-bold text-on-surface text-sm" htmlFor="retro-answer">
        得られた回答・アドバイス<span className="text-error">*</span>
      </label>
      <textarea
        id="retro-answer"
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
        rows={5}
        placeholder="教わった内容を、次の人が読んで分かる粒度で"
        className="mt-xs w-full rounded-lg border border-outline-variant bg-surface-container-lowest px-sm py-2 text-on-surface outline-none focus:border-primary"
      />

      <fieldset className="mt-md">
        <legend className="font-bold text-on-surface text-sm">
          解決度<span className="text-error">*</span>
        </legend>
        <div className="mt-sm flex flex-col gap-xs sm:flex-row sm:gap-md">
          {RESOLUTIONS.map((option) => (
            <label key={option.value} className="flex items-center gap-xs text-on-surface text-sm">
              <input
                type="radio"
                name="retro-resolution"
                value={option.value}
                checked={resolution === option.value}
                onChange={() => setResolution(option.value)}
              />
              {option.label}
            </label>
          ))}
        </div>
        {resolution === "unresolved" ? (
          <p className="mt-xs text-on-surface-variant text-xs">
            記録は残りますが、相談相手の評価を下げることはありません。
          </p>
        ) : null}
      </fieldset>

      {error ? (
        <p role="alert" className="mt-sm text-error text-sm">
          {error}
        </p>
      ) : null}

      <button
        type="submit"
        disabled={!canSubmit}
        className="mt-md min-h-[44px] rounded-full bg-primary px-lg py-sm font-bold text-on-primary transition-colors hover:bg-primary-container disabled:cursor-not-allowed disabled:opacity-50"
      >
        {submitting ? "記録中..." : "記録する"}
      </button>
    </form>
  );
}

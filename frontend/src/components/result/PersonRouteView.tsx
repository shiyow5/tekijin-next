"use client";

/**
 * Main-line result (route === "person"): the person is the answer, so this
 * leads with the candidate cards and a sendable draft. Up to three candidates
 * are shown; the asker may pick any of them as the recipient, not just the top
 * pick (#200) — picking one calls `POST /handoff/select`, which reorders the
 * durable hand-off target and regenerates the draft for that person. Only the
 * top card is `expanded`; the comparison signal 現在の負荷 is surfaced on
 * every card by `CandidateCard` itself (#204).
 * "この内容で依頼する" then persists the (possibly further-edited) draft text
 * to the pending hand-off (POST /handoff/draft), so the responder reads the
 * edited text — it is a real send, not a UI-only transition (#174).
 *
 * "AIに下書きを作り直してもらう" calls `POST /handoff/redraft` (#260): the graph
 * loops back through C7 for the same target and re-emits a fresh draft over the
 * stream. It remounts the editor (discarding the asker's local edits) so the
 * regenerated text replaces what they had.
 *
 * "この人には聞かない" on the current send target calls `POST /handoff/exclude`
 * (#260): it declines the top pick and reroutes to a freshly-scored next
 * candidate, which arrives over the open `/events` stream and remounts this view
 * (keyed by the new top). The control is offered ONLY on the selected target —
 * the backend rejects excluding a non-target candidate (422) — so the reroute
 * always follows the person the asker actually named (no mis-send).
 *
 * "AIの理解を訂正する" calls `POST /handoff/correct` (#260): the asker adds a
 * supplement, the whole pipeline re-runs from C1, and we navigate back to the
 * processing screen where the shared stream shows the re-think (which may ask a
 * fresh clarification or land on a new result).
 */

import { ReferenceAnswer } from "@/components/ReferenceAnswer";
import { CandidateCard } from "@/components/result/CandidateCard";
import { ConsultMethodDialog } from "@/components/result/ConsultMethodDialog";
import { DraftEditor } from "@/components/result/DraftEditor";
import { QuestionStructurePanel } from "@/components/result/QuestionStructurePanel";
import {
  correctInterpretation,
  excludeHandoffCandidate,
  regenerateHandoffDraft,
  selectHandoffCandidate,
  updateHandoffDraft,
} from "@/lib/api-client";
import type { ConsultMethod, Recommendation, ReferenceData } from "@/lib/api-types";
import { fitPercents } from "@/lib/fit";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

export interface PersonRouteViewProps {
  recommendations: Recommendation[];
  reason?: string;
  draft: string;
  /** #413: additive cited answer to show above the candidates (optional). */
  reference?: ReferenceData;
  /** Session id for the confirm POST; null when rendered outside a session. */
  sessionId: string | null;
}

const MAX_CANDIDATES = 3;
const SEND_ERROR = "送信に失敗しました。時間をおいて、もう一度お試しください。";
const SELECT_ERROR = "候補の切り替えに失敗しました。時間をおいて、もう一度お試しください。";
const EXCLUDE_ERROR = "候補の選び直しに失敗しました。時間をおいて、もう一度お試しください。";
const REDRAFT_ERROR = "下書きの作り直しに失敗しました。時間をおいて、もう一度お試しください。";
const CORRECT_ERROR = "解釈の訂正に失敗しました。時間をおいて、もう一度お試しください。";

export function PersonRouteView({
  recommendations,
  reason,
  draft,
  reference,
  sessionId,
}: PersonRouteViewProps) {
  const router = useRouter();
  const candidates = recommendations.slice(0, MAX_CANDIDATES);
  // Absolute fit % from each candidate's score, decoupled from the 高/中/低
  // confidence label (#240): a strong #1 on a never-asked topic (label 低) still
  // reads high, and the label rides along as a separate evidence badge.
  const fitValues = fitPercents(candidates);
  const [selectedPersonId, setSelectedPersonId] = useState(candidates[0]?.person_id ?? null);
  const [localDraft, setLocalDraft] = useState(draft);
  const [selecting, setSelecting] = useState(false);
  const [excluding, setExcluding] = useState(false);
  const [redrafting, setRedrafting] = useState(false);
  const [correcting, setCorrecting] = useState(false);
  const [correctOpen, setCorrectOpen] = useState(false);
  const [correctText, setCorrectText] = useState("");
  // Bumping this remounts the DraftEditor (keyed on it), which discards the
  // asker's local edits and re-seeds from the AI draft — the point of "作り直し".
  const [redraftNonce, setRedraftNonce] = useState(0);
  // Bumped when the asker folds the structured question into the draft (#475), so
  // the editor remounts and shows the combined text (same remount pattern as 作り直し).
  const [structureNonce, setStructureNonce] = useState(0);
  const [sending, setSending] = useState(false);
  const [sentTo, setSentTo] = useState<string | null>(null);
  // Holds the draft text between pressing "この内容で依頼する" and picking a
  // consultation method in the popup; null means the popup is closed (#245).
  const [pendingDraft, setPendingDraft] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const topPersonId = candidates[0]?.person_id ?? null;
  // A later-arriving `draft` SSE event (recommend can land before draft) updates
  // the editor baseline — but ONLY while the top pick is still the selected
  // recipient. After a reselect, the SSE `draft` is the (stale) top-pick draft;
  // the reselected candidate's draft came from handleSelect's response, so folding
  // the SSE baseline back in would clobber it. That clobber was the #258 flake:
  // a `draft` event re-firing after a reselect reverted the editor to the top's
  // text. Gating on `selectedPersonId === topPersonId` makes the two writers
  // (SSE baseline vs reselect response) mutually exclusive.
  useEffect(() => {
    if (selectedPersonId === topPersonId) {
      setLocalDraft(draft);
    }
  }, [draft, selectedPersonId, topPersonId]);

  // A decline→reroute remounts this view (keyed by the top candidate in
  // ResultScreen) while a confirm POST may still be in flight. Drop the post-await
  // state updates if that happened, matching the project's async-guard convention.
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  async function handleSelect(personId: string) {
    if (selecting || excluding || redrafting || personId === selectedPersonId || !sessionId) return;
    setSelecting(true);
    setError(null);
    try {
      const result = await selectHandoffCandidate({ session_id: sessionId, person_id: personId });
      if (mounted.current) {
        setSelectedPersonId(personId);
        setLocalDraft(result.draft);
      }
    } catch {
      // 404 = the hand-off was already answered/closed; 409 = a clarification is
      // owed; 422 = the candidate is no longer among the shown recommendations.
      if (mounted.current) setError(SELECT_ERROR);
    } finally {
      if (mounted.current) setSelecting(false);
    }
  }

  async function handleExclude(personId: string) {
    if (excluding || selecting || redrafting || !sessionId) return;
    setExcluding(true);
    setError(null);
    try {
      await excludeHandoffCandidate({ session_id: sessionId, person_id: personId });
      // Success: the reroute is queued server-side; the freshly-scored next
      // candidate + draft arrive over the open /events stream, which remounts
      // this view (keyed by the new top candidate) and clears `excluding`. Keep
      // the UI disabled meanwhile so the asker can't double-exclude.
    } catch {
      // 404 = the hand-off was already answered/closed; 409 = a clarification is
      // owed; 422 = this person is no longer the send target (a reroute moved on).
      if (mounted.current) {
        setError(EXCLUDE_ERROR);
        setExcluding(false);
      }
    }
  }

  async function handleRedraft() {
    if (redrafting || selecting || excluding || sending || !sessionId) return;
    setRedrafting(true);
    setError(null);
    // Remount the editor now so the asker's edits are visibly discarded and the
    // AI draft is shown; the regenerated text then streams in over /events (for a
    // real model it differs; the deterministic stub reproduces the same draft).
    setRedraftNonce((n) => n + 1);
    try {
      await regenerateHandoffDraft({ session_id: sessionId });
    } catch {
      // 404 = the hand-off was already answered/closed; 409 = a clarification is owed.
      if (mounted.current) setError(REDRAFT_ERROR);
    } finally {
      if (mounted.current) setRedrafting(false);
    }
  }

  async function handleCorrect() {
    const supplement = correctText.trim();
    if (
      correcting ||
      redrafting ||
      selecting ||
      excluding ||
      sending ||
      !sessionId ||
      !supplement
    ) {
      return;
    }
    setCorrecting(true);
    setError(null);
    try {
      await correctInterpretation({ session_id: sessionId, supplement });
      // The whole pipeline re-runs from C1 over the shared stream; go back to the
      // processing screen where the re-think (a fresh clarification or a new
      // result) is rendered. Guard the unmount-crossing navigation.
      if (mounted.current) router.push(`/session/${sessionId}`);
    } catch {
      // 404 = the hand-off was already answered/closed; 409 = a clarification is owed.
      if (mounted.current) {
        setError(CORRECT_ERROR);
        setCorrecting(false);
      }
    }
  }

  async function handleSend(text: string, consultMethod: ConsultMethod) {
    if (!sessionId) {
      setError(SEND_ERROR);
      return;
    }
    setSending(true);
    setError(null);
    try {
      await updateHandoffDraft({
        session_id: sessionId,
        draft: text,
        consult_method: consultMethod,
      });
      const selected = candidates.find((c) => c.person_id === selectedPersonId);
      if (mounted.current) setSentTo(selected?.name ?? candidates[0]?.name ?? "ご担当者");
    } catch {
      // 404 = the hand-off was already answered/closed; 409 = a clarification is
      // owed; either way the send can't land, so surface a retryable error.
      if (mounted.current) setError(SEND_ERROR);
    } finally {
      if (mounted.current) setSending(false);
    }
  }

  // #475: fold the (edited) structured question into the draft and remount the
  // editor so the combined text shows. Appends below the polite letter rather than
  // replacing it, so the greeting/addressee the template produced is kept.
  function handleApplyStructure(block: string) {
    if (!block) return;
    setLocalDraft((prev) => (prev.trim() ? `${prev.trim()}\n\n${block}` : block));
    setStructureNonce((n) => n + 1);
  }

  function handleChooseConsultMethod(consultMethod: ConsultMethod) {
    if (pendingDraft === null) return;
    const text = pendingDraft;
    setPendingDraft(null);
    void handleSend(text, consultMethod);
  }

  if (sentTo !== null) {
    return (
      <section className="mx-auto flex w-full max-w-2xl flex-col gap-md py-lg text-center">
        <h1 className="font-bold text-2xl text-primary">依頼を送りました</h1>
        <p className="text-on-surface-variant">
          {sentTo}さんに、この内容でお繋ぎしました。返信があるとお知らせします。
        </p>
        <div className="flex justify-center">
          <a
            href="/"
            className="min-h-[48px] rounded-full bg-primary px-lg py-sm font-bold text-on-primary shadow-md transition-colors hover:bg-primary-container"
          >
            新しい質問をする
          </a>
        </div>
      </section>
    );
  }

  return (
    <section className="mx-auto flex w-full max-w-5xl flex-col gap-md py-lg">
      <header className="flex flex-col gap-xs">
        <h1 className="font-bold text-2xl text-on-surface">この質問は、人に聞くのが確実です</h1>
        <p className="text-on-surface-variant">
          {reason || "直近で同様の質問に答えた方の知見が役立ちそうです。"}
        </p>
      </header>

      {/* #413: past-answer citation shown above the candidates — a supporting
          reference, not a replacement for the hand-off. */}
      <ReferenceAnswer reference={reference} sessionId={sessionId} />

      {candidates.length > 0 ? (
        <div className="grid grid-cols-1 gap-md md:grid-cols-3">
          {candidates.map((candidate, index) => (
            <CandidateCard
              key={candidate.person_id}
              candidate={candidate}
              rank={index + 1}
              expanded={index === 0}
              selected={candidate.person_id === selectedPersonId}
              onSelect={sessionId ? handleSelect : undefined}
              onExclude={
                sessionId && candidate.person_id === selectedPersonId ? handleExclude : undefined
              }
              fitPercent={fitValues[index]}
            />
          ))}
        </div>
      ) : (
        // Graceful fallback: a reconnect at the send interrupt can replay the
        // draft without the candidates. Keep the draft sendable rather than
        // dead-ending.
        <p className="rounded-lg border border-outline-variant bg-surface-container-low p-md text-on-surface-variant text-sm">
          宛先候補を再取得しています。この下書きはそのまま送れます。
        </p>
      )}

      {error ? (
        <p
          role="alert"
          className="rounded-lg border border-error-container bg-error-container p-sm text-on-error-container text-sm"
        >
          {error}
        </p>
      ) : null}

      {excluding ? (
        <output className="block rounded-lg border border-outline-variant bg-surface-container-low p-sm text-on-surface-variant text-sm">
          別の方を選び直しています…
        </output>
      ) : null}

      {/* #475 Screen 01: on-demand "質問を整理する" — reshape the raw question into
          起きていること / 環境 / 試したこと / 詰まっている点, editable, then fold into
          the draft. Off the C1 critical path (generated only when the asker asks). */}
      {sessionId ? (
        <QuestionStructurePanel
          sessionId={sessionId}
          disabled={sending || selecting || excluding || redrafting || correcting}
          onApply={handleApplyStructure}
        />
      ) : null}

      {sessionId ? (
        <div className="flex justify-end">
          <button
            type="button"
            onClick={handleRedraft}
            disabled={redrafting || selecting || excluding || sending}
            className="min-h-[32px] rounded-lg px-sm py-1 font-medium text-primary text-xs underline decoration-dotted underline-offset-2 transition-colors hover:text-primary-container disabled:cursor-not-allowed disabled:opacity-50"
          >
            {redrafting ? "作り直しています…" : "AIに下書きを作り直してもらう"}
          </button>
        </div>
      ) : null}

      <DraftEditor
        key={`${selectedPersonId}:${redraftNonce}:${structureNonce}`}
        initialDraft={localDraft}
        disabled={sending || selecting || excluding || redrafting || correcting}
        onSend={(text) => setPendingDraft(text)}
      />

      {sessionId ? (
        <div className="rounded-xl border border-outline-variant border-dashed bg-surface-container-lowest p-md">
          {correctOpen ? (
            <div className="flex flex-col gap-sm">
              <label
                htmlFor="interpretation-correction"
                className="font-medium text-on-surface text-sm"
              >
                AIの理解が違う場合は、補足して質問し直せます
              </label>
              <textarea
                id="interpretation-correction"
                value={correctText}
                onChange={(e) => setCorrectText(e.target.value)}
                disabled={correcting}
                maxLength={2000}
                placeholder="例：対象は営業部ではなく情報システム部です／製品はUTMではなくVPNです"
                className="h-24 w-full resize-none rounded-lg border border-outline-variant bg-surface p-sm text-on-surface text-sm outline-none focus:border-primary"
              />
              <div className="flex justify-end gap-sm">
                <button
                  type="button"
                  onClick={() => setCorrectOpen(false)}
                  disabled={correcting}
                  className="min-h-[36px] rounded-lg px-md py-1 font-medium text-on-surface-variant text-sm transition-colors hover:text-on-surface disabled:opacity-50"
                >
                  やめる
                </button>
                <button
                  type="button"
                  onClick={handleCorrect}
                  disabled={correcting || correctText.trim().length === 0}
                  className="inline-flex min-h-[36px] items-center rounded-lg bg-secondary-container px-md py-1 font-medium text-on-secondary-container text-sm transition-colors hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {correcting ? "質問し直しています…" : "補足して質問し直す"}
                </button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              onClick={() => setCorrectOpen(true)}
              className="font-medium text-on-surface-variant text-sm underline decoration-dotted underline-offset-2 transition-colors hover:text-on-surface"
            >
              AIの理解が違いますか？補足して質問し直す
            </button>
          )}
        </div>
      ) : null}

      {pendingDraft !== null ? (
        <ConsultMethodDialog
          onChoose={handleChooseConsultMethod}
          onCancel={() => setPendingDraft(null)}
          disabled={sending}
        />
      ) : null}
    </section>
  );
}

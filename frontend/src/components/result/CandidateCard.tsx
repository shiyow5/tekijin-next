/**
 * One recommended person on the main-line result (product-spec 画面3).
 *
 * Shows the person id ("E###"), name, department, the score-derived fit percent,
 * the separately-labelled evidence confidence, and the evidence `reasons`. The
 * raw `score` is a weighted internal ranking value (not a percentage) and is never
 * shown. The top-ranked card is `expanded` (full reason detail); lower
 * ranks show compact reason labels — EXCEPT the comparison signal 現在の負荷
 * (load), whose value is shown on every card so the asker can actually compare
 * 2nd/3rd against the top pick (#204).
 *
 * `onSelect` lets the asker hand off to any of the shown candidates, not only
 * the top pick (#200).
 */

import { ConfidenceGauge } from "@/components/result/ConfidenceGauge";
import type { Recommendation } from "@/lib/api-types";
import { REVEAL_CLASS, revealStyle } from "@/lib/motion";
import { reasonLabel } from "@/lib/reasons";

export interface CandidateCardProps {
  candidate: Recommendation;
  rank: number;
  expanded: boolean;
  /** Highlight this card (the confirmed recipient / top pick). */
  selected: boolean;
  /**
   * Recipient-selection handler (#200/#A1). When omitted the card is
   * display-only (no "選択する" button) — used when there is no session to act
   * against (e.g. a static/replayed view).
   */
  onSelect?: (personId: string) => void;
  /**
   * Exclusion handler (#260, "この人には聞かない"). Provided only for the current
   * send target: excluding reroutes to a freshly-scored next candidate. When
   * omitted the control is hidden (a non-target card cannot be excluded).
   */
  onExclude?: (personId: string) => void;
  /**
   * Absolute fit percentage from this candidate's own score (#240), normalised to
   * the scorer's fit ceiling. Decoupled from the 高/中/低 confidence label, so a
   * strong candidate on a never-asked topic still reads high. Drives the gauge's
   * ring + number. Omitted → the gauge falls back to the level's own magnitude.
   */
  fitPercent?: number;
}

function avatarInitial(name: string): string {
  return name.slice(0, 1);
}

export function CandidateCard({
  candidate,
  rank,
  expanded,
  selected,
  onSelect,
  onExclude,
  fitPercent,
}: CandidateCardProps) {
  return (
    <article
      style={revealStyle(rank - 1)}
      className={`flex h-full flex-col rounded-xl border bg-surface-container-lowest p-md shadow-sm transition-colors ${REVEAL_CLASS} ${
        selected ? "border-primary" : "border-outline-variant"
      }`}
    >
      <div className="mb-sm flex items-start gap-sm">
        <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-surface-variant font-bold text-lg text-on-surface-variant">
          {avatarInitial(candidate.name)}
        </div>
        <div className="min-w-0 flex-1">
          <h2 className="break-words font-bold text-lg text-on-surface leading-tight">
            {candidate.name}
          </h2>
          <p className="truncate text-on-surface-variant text-xs">
            {[candidate.dept, candidate.person_id].filter(Boolean).join(" / ")}
          </p>
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-xs">
          <span aria-hidden="true" className="text-on-surface-variant text-xs">
            適合度
          </span>
          <ConfidenceGauge percent={fitPercent} />
        </div>
      </div>

      {candidate.reasons.length > 0 ? (
        <ul className="mb-md flex flex-col gap-xs">
          {candidate.reasons.map((reason) => {
            // Current load is a comparison signal: show its value on every card,
            // not only the expanded top one (#204).
            const showDetail = expanded || reason.type === "load";
            return (
              <li
                key={`${reason.type}-${reason.detail}`}
                className="flex items-start gap-xs text-on-surface-variant text-sm"
              >
                <span aria-hidden="true" className="text-primary">
                  ✓
                </span>
                <span>
                  <span className="font-medium text-on-surface">{reasonLabel(reason.type)}</span>
                  {showDetail && reason.detail ? `：${reason.detail}` : null}
                </span>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="mb-md text-on-surface-variant text-sm">根拠を確認中…</p>
      )}

      {onSelect ? (
        <div className="mt-auto flex flex-col gap-xs">
          <button
            type="button"
            onClick={() => onSelect(candidate.person_id)}
            aria-pressed={selected}
            className={`min-h-[40px] rounded-lg px-md py-2 font-medium text-sm transition-colors ${
              selected
                ? "bg-primary text-on-primary"
                : "border border-primary text-primary hover:bg-surface-container-low"
            }`}
          >
            {selected ? "選択中" : "選択する"}
          </button>
          {/* Exclusion is offered only on the current send target (#260): declining
              them reroutes to a freshly-scored next candidate. */}
          {onExclude ? (
            <button
              type="button"
              onClick={() => onExclude(candidate.person_id)}
              className="min-h-[32px] rounded-lg px-md py-1 font-medium text-on-surface-variant text-xs underline decoration-dotted underline-offset-2 transition-colors hover:text-on-surface"
            >
              この人には聞かない
            </button>
          ) : null}
        </div>
      ) : rank === 1 ? (
        <span className="mt-auto rounded-lg bg-secondary-container px-md py-2 text-center font-medium text-on-secondary-container text-sm">
          この方に依頼します
        </span>
      ) : null}
    </article>
  );
}

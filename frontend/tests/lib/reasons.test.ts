import type { Reason } from "@/lib/api-types";
import { REASON_LABELS, answersEvidence, reasonDetail, reasonLabel } from "@/lib/reasons";
import { describe, expect, it } from "vitest";

describe("reasonLabel", () => {
  it("maps known reason types to Japanese labels", () => {
    expect(reasonLabel("answers")).toBe(REASON_LABELS.answers);
    expect(reasonLabel("answers")).toBe("過去回答");
    expect(reasonLabel("load")).toBe("現在の負荷");
    expect(reasonLabel("recency")).toBe("直近の活動");
  });

  it("labels the question_fit reason instead of showing the raw type", () => {
    expect(reasonLabel("question_fit")).toBe("質問と過去の回答の近さ");
  });

  it("has no labels for the evidence removed in #95", () => {
    for (const removed of ["cert", "self", "skill", "project"]) {
      expect(REASON_LABELS[removed]).toBeUndefined();
    }
  });

  it("falls back to the raw type for unknown reasons", () => {
    expect(reasonLabel("mystery")).toBe("mystery");
  });
});

describe("an organization member's evidence (#107)", () => {
  it("labels the four kinds of evidence and a referral", () => {
    expect(reasonLabel("self_intro")).toBe("自己紹介");
    expect(reasonLabel("document")).toBe("文書");
    expect(reasonLabel("conversation")).toBe("会話");
    expect(reasonLabel("answer")).toBe("TEKIJINでの回答");
    expect(reasonLabel("referral")).toBe("担当者からの推薦");
  });

  it("drops a detail's prefix when it repeats the label", () => {
    expect(reasonDetail({ type: "self_intro", detail: "自己紹介: 星座の神話が好きです" })).toBe(
      "星座の神話が好きです",
    );
    expect(reasonDetail({ type: "referral", detail: "担当者からの推薦" })).toBeNull();
  });

  it("keeps every other detail as it is", () => {
    const detail = "『機材の使い方と管理マニュアル』を書いた";
    expect(reasonDetail({ type: "document", detail })).toBe(detail);
    expect(reasonDetail({ type: "load", detail: "今週の対応件数: 少なめ" })).toBe(
      "今週の対応件数: 少なめ",
    );
    expect(reasonDetail({ type: "answers", detail: "" })).toBeNull();
  });
});

describe("answersEvidence", () => {
  it("returns the answers reason detail verbatim (real backend format)", () => {
    const reasons: Reason[] = [
      { type: "recency", detail: "直近1か月以内の関連実績あり" },
      { type: "answers", detail: "類似の質問に過去5件回答（うち有用と評価3件）" },
    ];
    expect(answersEvidence(reasons)).toBe("類似の質問に過去5件回答（うち有用と評価3件）");
  });

  it("returns null when there is no answers reason", () => {
    expect(answersEvidence([{ type: "load", detail: "今週の対応件数: 少なめ" }])).toBeNull();
  });

  it("never synthesises a reuse count from other reason types", () => {
    // The backend does not expose a raw reuse count; only the `answers` detail
    // is used, so a non-answers reason yields null (no fabricated number).
    expect(answersEvidence([{ type: "recency", detail: "直近の関連実績: 約3か月前" }])).toBeNull();
  });
});

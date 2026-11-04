import { MAX_COMPOSITE_SCORE, fitLevel, fitPercent, fitPercents, levelFraction } from "@/lib/fit";
import { describe, expect, it } from "vitest";

describe("levelFraction", () => {
  it("maps the qualitative levels and defaults unknown to half", () => {
    expect(levelFraction("高")).toBe(1);
    expect(levelFraction("中")).toBe(0.66);
    expect(levelFraction("低")).toBe(0.33);
    expect(levelFraction("不明")).toBe(0.5);
  });
});

describe("fitLevel", () => {
  it("uses the same 33/66 boundaries as the displayed fit magnitude", () => {
    expect(fitLevel(0)).toBe("低");
    expect(fitLevel(33)).toBe("低");
    expect(fitLevel(34)).toBe("中");
    expect(fitLevel(66)).toBe("中");
    expect(fitLevel(67)).toBe("高");
    expect(fitLevel(100)).toBe("高");
  });

  it("clamps out-of-range input before assigning a band", () => {
    expect(fitLevel(-1)).toBe("低");
    expect(fitLevel(101)).toBe("高");
  });
});

describe("fitPercent", () => {
  it("normalises a composite score against the qsim-inclusive ceiling (1.8 -> 100%)", () => {
    // #498: the ceiling now includes the #405 question-fit term, so a full score of
    // 1.8 is 100%; a base-only score of 0.8 no longer saturates.
    expect(MAX_COMPOSITE_SCORE).toBe(1.8);
    expect(fitPercent(MAX_COMPOSITE_SCORE)).toBe(100);
    expect(fitPercent(0.8)).toBe(44); // base-only strong — de-saturated (was 100)
    expect(fitPercent(0.45)).toBe(25);
    expect(fitPercent(0)).toBe(0);
  });

  it("clamps: a score above the ceiling caps at 100, a negative score floors at 0", () => {
    expect(fitPercent(2.5)).toBe(100); // above the 1.8 ceiling -> clamp 100
    expect(fitPercent(-0.5)).toBe(0);
  });
});

describe("fitPercents", () => {
  it("returns each candidate's absolute fit, independent of order", () => {
    // Normalised against the 1.8 ceiling (#498): 0.9->50, 0.45->25, 0.18->10.
    expect(
      fitPercents([
        { score: 0.9, confidence: "低" },
        { score: 0.45, confidence: "高" },
        { score: 0.18, confidence: "中" },
      ]),
    ).toEqual([50, 25, 10]);
  });

  it("is decoupled from the confidence label — a 低 top still reads high (#240)", () => {
    // A strong candidate on a never-asked topic: label 低 but a fit gauge above the
    // old label-anchored 33% cap.
    const strongLow = fitPercents([{ score: 0.74, confidence: "低" }]);
    const strongHigh = fitPercents([{ score: 0.74, confidence: "高" }]);
    expect(strongLow).toEqual(strongHigh); // label does not change the gauge
    // NOTE (#498): under the qsim-inclusive 1.8 ceiling, 0.74 reads 41% — still
    // above 33%, but the margin is now thin (was 82% under the 0.9 ceiling). This
    // is the acknowledged single-ceiling trade-off; if the backend qsim/weights
    // distribution drifts, a strong-but-qsim-less candidate could dip toward the
    // old #240 under-read. The proper fix is backend-computed fit (#500), which
    // removes this frontend re-derivation entirely.
    expect(strongLow[0]).toBeGreaterThan(33); // no longer capped at 33
  });

  it("differentiates candidates by their own score, not the shared label", () => {
    const out = fitPercents([
      { score: 0.9, confidence: "高" },
      { score: 0.81, confidence: "高" },
      { score: 0.54, confidence: "高" },
    ]);
    // #498: 0.9->50, 0.81->45, 0.54->30 against the 1.8 ceiling — still monotonic.
    expect(out).toEqual([50, 45, 30]);
    expect(new Set(out).size).toBeGreaterThan(1);
  });

  it("returns an empty array for no candidates", () => {
    expect(fitPercents([])).toEqual([]);
  });
});

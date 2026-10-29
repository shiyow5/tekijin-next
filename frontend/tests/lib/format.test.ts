import { formatConfidence, sourceOrigin } from "@/lib/format";
import { describe, expect, it } from "vitest";

describe("formatConfidence", () => {
  it("renders a 0..1 float as a whole percent", () => {
    expect(formatConfidence(0.85)).toBe("85%");
    expect(formatConfidence(0)).toBe("0%");
    expect(formatConfidence(1)).toBe("100%");
  });

  it("rounds to the nearest percent", () => {
    expect(formatConfidence(0.626)).toBe("63%");
  });

  it("clamps out-of-range values", () => {
    expect(formatConfidence(1.5)).toBe("100%");
    expect(formatConfidence(-0.2)).toBe("0%");
  });

  it("returns a dash for non-finite input", () => {
    expect(formatConfidence(Number.NaN)).toBe("—");
    expect(formatConfidence(Number.POSITIVE_INFINITY)).toBe("—");
    expect(formatConfidence(Number.NEGATIVE_INFINITY)).toBe("—");
  });
});

describe("sourceOrigin (#76)", () => {
  it("names a Discord conversation, an author, or the admin", () => {
    expect(sourceOrigin({ provider: "discord", author_name: null })).toBe("Discordの会話");
    expect(sourceOrigin({ provider: "manual", author_name: "アリス" })).toBe("追加: アリス");
    expect(sourceOrigin({ provider: "manual", author_name: null })).toBe("追加: 管理者");
  });
});

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

  it("names a shared Google document or slide deck (#99)", () => {
    const doc = "https://docs.google.com/document/d/abc/edit";
    const slides = "https://docs.google.com/presentation/d/abc/edit";
    expect(sourceOrigin({ provider: "google", author_name: null, url: doc })).toBe(
      "Googleドキュメント",
    );
    expect(sourceOrigin({ provider: "google", author_name: null, url: slides })).toBe(
      "Googleスライド",
    );
    expect(sourceOrigin({ provider: "google", author_name: null, url: null })).toBe("Google");
  });

  it("names a Notion page (#101)", () => {
    const url = "https://www.notion.so/abc";
    expect(sourceOrigin({ provider: "notion", author_name: null, url })).toBe("Notion");
  });
});

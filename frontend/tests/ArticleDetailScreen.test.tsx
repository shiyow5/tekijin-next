import { ArticleDetailScreen } from "@/components/ArticleDetailScreen";
import { ApiError } from "@/lib/api-client";
import type { ArticleDetail } from "@/lib/api-types";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const getArticleMock = vi.fn();
const patchArticleMock = vi.fn();
const reviewArticleMock = vi.fn();

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ApiError: actual.ApiError,
    getArticle: (...args: unknown[]) => getArticleMock(...args),
    patchArticle: (...args: unknown[]) => patchArticleMock(...args),
    reviewArticle: (...args: unknown[]) => reviewArticleMock(...args),
  };
});

const DRAFT: ArticleDetail = {
  id: 7,
  kind: "fact",
  issue: "会費はいくら",
  environment: null,
  resolution: "月500円",
  cause: null,
  state: "draft",
  reuse_count: 0,
  demand_count: 2,
  last_used_at: null,
  superseded_by: null,
  author_name: "アリス",
  verifier_name: null,
  created_at: "2026-11-01T10:00:00",
  updated_at: "2026-11-01T10:00:00",
  verified_at: null,
  sources: [
    {
      kind: "source",
      source_item_id: 3,
      title: "会計の引継ぎ",
      url: "https://example.com/kaikei",
      quote: "会費は月500円",
    },
    { kind: "message", title: "#会計", url: "https://discord.com/channels/1/2/3", quote: null },
  ],
  can_review: true,
};

beforeEach(() => {
  getArticleMock.mockReset().mockResolvedValue(DRAFT);
  patchArticleMock.mockReset();
  reviewArticleMock.mockReset();
});

describe("ArticleDetailScreen (#112)", () => {
  it("shows the article and where it came from", async () => {
    getArticleMock.mockResolvedValue({ ...DRAFT, state: "verified", can_review: false });
    render(<ArticleDetailScreen articleId="7" />);

    expect(await screen.findByRole("heading", { name: "会費はいくら" })).toBeInTheDocument();
    expect(screen.getByText("月500円")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "会計の引継ぎ" })).toHaveAttribute(
      "href",
      "/sources/3",
    );
    expect(screen.getByText("「会費は月500円」")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "#会計" })).toHaveAttribute(
      "href",
      "https://discord.com/channels/1/2/3",
    );
    // A member reads only: no review controls.
    expect(screen.queryByRole("button", { name: "確認済みにする" })).toBeNull();
    expect(screen.queryByLabelText("答え")).toBeNull();
  });

  it("lets the admin edit and verify", async () => {
    patchArticleMock.mockResolvedValue({ ...DRAFT, resolution: "月600円" });
    reviewArticleMock.mockResolvedValue({ ...DRAFT, resolution: "月600円", state: "verified" });
    render(<ArticleDetailScreen articleId="7" />);

    const answer = await screen.findByLabelText("答え");
    fireEvent.change(answer, { target: { value: "月600円" } });
    fireEvent.click(screen.getByRole("button", { name: "保存する" }));
    await waitFor(() =>
      expect(patchArticleMock).toHaveBeenCalledWith(
        7,
        expect.objectContaining({ resolution: "月600円", issue: "会費はいくら" }),
      ),
    );

    fireEvent.click(screen.getByRole("button", { name: "確認済みにする" }));
    await waitFor(() => expect(reviewArticleMock).toHaveBeenCalledWith(7, "verify", {}));
    expect(await screen.findByText("確認済み")).toBeInTheDocument();
  });

  it("sends back and retires", async () => {
    reviewArticleMock.mockResolvedValueOnce({ ...DRAFT, state: "wip" });
    render(<ArticleDetailScreen articleId="7" />);

    fireEvent.click(await screen.findByRole("button", { name: "差し戻す" }));
    await waitFor(() => expect(reviewArticleMock).toHaveBeenCalledWith(7, "send-back", {}));

    reviewArticleMock.mockResolvedValueOnce({ ...DRAFT, state: "retired", superseded_by: 9 });
    fireEvent.change(screen.getByLabelText("置き換え先の知識の番号（任意）"), {
      target: { value: "9" },
    });
    fireEvent.click(screen.getByRole("button", { name: "引退させる" }));
    await waitFor(() =>
      expect(reviewArticleMock).toHaveBeenLastCalledWith(7, "retire", { supersededBy: 9 }),
    );
  });

  it("cannot verify without an answer", async () => {
    getArticleMock.mockResolvedValue({ ...DRAFT, resolution: null, state: "wip" });
    render(<ArticleDetailScreen articleId="7" />);
    expect(await screen.findByRole("button", { name: "確認済みにする" })).toBeDisabled();
  });

  it("reports the server's refusal", async () => {
    reviewArticleMock.mockRejectedValue(
      new ApiError(409, "答えが空の知識は確認済みにできません。"),
    );
    render(<ArticleDetailScreen articleId="7" />);
    fireEvent.click(await screen.findByRole("button", { name: "確認済みにする" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "答えが空の知識は確認済みにできません。",
    );
  });

  it("says when the article does not exist", async () => {
    getArticleMock.mockRejectedValue(new ApiError(404, "not found"));
    render(<ArticleDetailScreen articleId="7" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("知識が見つかりません。");
  });

  it("sends every edited field", async () => {
    patchArticleMock.mockResolvedValue(DRAFT);
    render(<ArticleDetailScreen articleId="7" />);
    fireEvent.change(await screen.findByLabelText("種類"), { target: { value: "decision" } });
    fireEvent.change(screen.getByLabelText("問い"), { target: { value: "会費は？" } });
    fireEvent.change(screen.getByLabelText("場面"), { target: { value: "新年度" } });
    fireEvent.change(screen.getByLabelText("原因"), { target: { value: "値上げ" } });
    fireEvent.click(screen.getByRole("button", { name: "保存する" }));
    await waitFor(() =>
      expect(patchArticleMock).toHaveBeenCalledWith(7, {
        kind: "decision",
        issue: "会費は？",
        environment: "新年度",
        resolution: "月500円",
        cause: "値上げ",
      }),
    );
    expect(await screen.findByText("保存しました。")).toBeInTheDocument();
  });

  it("rejects an id that is not a number", async () => {
    render(<ArticleDetailScreen articleId="abc" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("知識が見つかりません。");
    expect(getArticleMock).not.toHaveBeenCalled();
  });
});

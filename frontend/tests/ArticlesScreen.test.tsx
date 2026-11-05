import { ArticlesScreen } from "@/components/ArticlesScreen";
import type { KnowledgeArticle } from "@/lib/api-types";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const getArticlesMock = vi.fn();

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ApiError: actual.ApiError,
    getArticles: (...args: unknown[]) => getArticlesMock(...args),
  };
});

const VERIFIED: KnowledgeArticle = {
  id: 1,
  kind: "procedure",
  issue: "部室の鍵はどこ",
  environment: "合宿の前",
  resolution: "部室の箱に入っている",
  cause: null,
  state: "verified",
  reuse_count: 3,
  demand_count: 0,
  last_used_at: "2026-11-04T10:00:00",
  superseded_by: null,
  author_name: "アリス",
  verifier_name: "ボブ",
  created_at: "2026-11-01T10:00:00",
  updated_at: "2026-11-01T10:00:00",
  verified_at: "2026-11-02T10:00:00",
};

const DRAFT: KnowledgeArticle = {
  ...VERIFIED,
  id: 2,
  kind: "fact",
  issue: "会費はいくら",
  resolution: "月500円",
  state: "draft",
  reuse_count: 0,
  demand_count: 4,
  verifier_name: null,
  verified_at: null,
};

beforeEach(() => {
  getArticlesMock.mockReset();
});

describe("ArticlesScreen (#112)", () => {
  it("lists verified knowledge for a member, without the review tab", async () => {
    getArticlesMock.mockResolvedValue({ items: [VERIFIED], total: 1, can_review: false });
    render(<ArticlesScreen />);

    const link = await screen.findByRole("link", { name: "部室の鍵はどこ" });
    expect(link).toHaveAttribute("href", "/articles/1");
    expect(screen.getByText("部室の箱に入っている")).toBeInTheDocument();
    expect(screen.getByText("手順")).toBeInTheDocument();
    expect(screen.getByText("回答に使われた回数: 3")).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "確認待ち" })).toBeNull();
    expect(getArticlesMock).toHaveBeenCalledWith(expect.objectContaining({ view: "verified" }));
  });

  it("shows the admin the review queue, most wanted first", async () => {
    getArticlesMock.mockImplementation(({ view }: { view: string }) =>
      Promise.resolve(
        view === "review"
          ? { items: [DRAFT], total: 1, can_review: true }
          : { items: [VERIFIED], total: 1, can_review: true },
      ),
    );
    render(<ArticlesScreen />);

    fireEvent.click(await screen.findByRole("tab", { name: "確認待ち" }));
    expect(await screen.findByRole("link", { name: "会費はいくら" })).toBeInTheDocument();
    expect(screen.getByText("この知識で答えられた質問: 4")).toBeInTheDocument();
    expect(getArticlesMock).toHaveBeenLastCalledWith(expect.objectContaining({ view: "review" }));
  });

  it("searches", async () => {
    getArticlesMock.mockResolvedValue({ items: [], total: 0, can_review: false });
    render(<ArticlesScreen />);
    await screen.findByText("まだ確認済みの知識がありません。");

    fireEvent.change(screen.getByLabelText("知識を探す"), { target: { value: "鍵" } });
    fireEvent.click(screen.getByRole("button", { name: "探す" }));
    await waitFor(() =>
      expect(getArticlesMock).toHaveBeenLastCalledWith(expect.objectContaining({ q: "鍵" })),
    );
  });

  it("reports a failure to load", async () => {
    getArticlesMock.mockRejectedValue(new Error("boom"));
    render(<ArticlesScreen />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "知識の一覧を読み込めませんでした。",
    );
  });

  it("loads more and switches back to verified", async () => {
    getArticlesMock.mockImplementation(({ offset }: { offset?: number }) =>
      Promise.resolve(
        offset
          ? { items: [DRAFT], total: 2, can_review: true }
          : { items: [VERIFIED], total: 2, can_review: true },
      ),
    );
    render(<ArticlesScreen />);
    fireEvent.click(await screen.findByRole("button", { name: "もっと見る（残り1件）" }));
    expect(await screen.findByRole("link", { name: "会費はいくら" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: "確認待ち" }));
    fireEvent.click(screen.getByRole("tab", { name: "確認済み" }));
    await waitFor(() =>
      expect(getArticlesMock).toHaveBeenLastCalledWith(
        expect.objectContaining({ view: "verified" }),
      ),
    );
  });

  it("drops a slower response for a tab that is no longer shown", async () => {
    let finishReview: (value: unknown) => void = () => {};
    getArticlesMock.mockImplementation(({ view }: { view: string }) =>
      view === "review"
        ? new Promise((resolve) => {
            finishReview = resolve;
          })
        : Promise.resolve({ items: [VERIFIED], total: 1, can_review: true }),
    );
    render(<ArticlesScreen />);
    fireEvent.click(await screen.findByRole("tab", { name: "確認待ち" }));
    fireEvent.click(screen.getByRole("tab", { name: "確認済み" }));
    expect(await screen.findByRole("link", { name: "部室の鍵はどこ" })).toBeInTheDocument();

    finishReview({ items: [DRAFT], total: 1, can_review: true });
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.queryByRole("link", { name: "会費はいくら" })).toBeNull();
    expect(screen.getByRole("link", { name: "部室の鍵はどこ" })).toBeInTheDocument();
  });
});

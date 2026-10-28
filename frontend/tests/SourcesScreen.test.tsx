import { SourcesScreen } from "@/components/SourcesScreen";
import type { SourceItem } from "@/lib/api-types";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getSourcesMock = vi.fn();
const postSourceMock = vi.fn();
const deleteSourceMock = vi.fn();

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ApiError: actual.ApiError,
    getSources: (...args: unknown[]) => getSourcesMock(...args),
    postSource: (...args: unknown[]) => postSourceMock(...args),
    deleteSource: (...args: unknown[]) => deleteSourceMock(...args),
  };
});

const MINE: SourceItem = {
  id: 1,
  provider: "manual",
  title: "会計の引継ぎ",
  url: "https://example.com/kaikei",
  author_name: "アリス",
  updated_at: "2026-10-28T12:00:00",
  chunks: 2,
  can_delete: true,
};
const THEIRS: SourceItem = {
  id: 2,
  provider: "manual",
  title: null,
  url: null,
  author_name: "ボブ",
  updated_at: "2026-10-27T09:00:00",
  chunks: 1,
  can_delete: false,
};

beforeEach(() => {
  getSourcesMock.mockReset().mockResolvedValue({ items: [MINE, THEIRS], total: 2 });
  postSourceMock.mockReset();
  deleteSourceMock.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

function fill(label: string, value: string) {
  fireEvent.change(screen.getByLabelText(label), { target: { value } });
}

describe("SourcesScreen (#72)", () => {
  it("lists the organization's sources", async () => {
    render(<SourcesScreen />);

    expect(await screen.findByText("会計の引継ぎ")).toBeInTheDocument();
    expect(screen.getByText("（題名なし）")).toBeInTheDocument();
    expect(screen.getByText(/アリス/)).toBeInTheDocument();
    const link = screen.getByRole("link", { name: "元の文書を開く" });
    expect(link).toHaveAttribute("href", "https://example.com/kaikei");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    // Only the member's own source can be deleted.
    expect(screen.getAllByRole("button", { name: /を削除/ })).toHaveLength(1);
  });

  it("pages past the first response", async () => {
    getSourcesMock
      .mockResolvedValueOnce({ items: [MINE], total: 2 })
      .mockResolvedValueOnce({ items: [THEIRS], total: 2 });
    render(<SourcesScreen />);

    fireEvent.click(await screen.findByRole("button", { name: "もっと見る（残り1件）" }));

    expect(await screen.findByText("（題名なし）")).toBeInTheDocument();
    expect(getSourcesMock).toHaveBeenLastCalledWith({ offset: 1, limit: 50 });
    expect(screen.queryByRole("button", { name: /もっと見る/ })).not.toBeInTheDocument();
  });

  it("shows an error when the next page fails", async () => {
    getSourcesMock
      .mockResolvedValueOnce({ items: [MINE], total: 2 })
      .mockRejectedValueOnce(new Error("boom"));
    render(<SourcesScreen />);

    fireEvent.click(await screen.findByRole("button", { name: /もっと見る/ }));

    expect(await screen.findByText("続きを読み込めませんでした。")).toBeInTheDocument();
  });

  it("shows an empty state", async () => {
    getSourcesMock.mockResolvedValue({ items: [], total: 0 });
    render(<SourcesScreen />);

    expect(await screen.findByText("まだ資料がありません。")).toBeInTheDocument();
  });

  it("shows why the list could not be loaded", async () => {
    const { ApiError } = await import("@/lib/api-client");
    getSourcesMock.mockRejectedValue(
      new ApiError(403, "組織に属していないため、資料を扱えません。"),
    );
    render(<SourcesScreen />);

    expect(
      await screen.findByText("組織に属していないため、資料を扱えません。"),
    ).toBeInTheDocument();
  });

  it("adds a source and reloads the list", async () => {
    postSourceMock.mockResolvedValue({ status: "added", id: 3, chunks: 2 });
    render(<SourcesScreen />);
    await screen.findByText("会計の引継ぎ");

    fill("題名（任意）", " 合宿の手順 ");
    fill("本文", "宿の予約は3か月前までに行う。");
    fireEvent.click(screen.getByRole("button", { name: "追加する" }));

    await waitFor(() =>
      expect(postSourceMock).toHaveBeenCalledWith({
        title: "合宿の手順",
        body: "宿の予約は3か月前までに行う。",
        url: null,
      }),
    );
    expect(await screen.findByText(/追加しました/)).toBeInTheDocument();
    expect(getSourcesMock).toHaveBeenCalledTimes(2);
    expect(screen.getByLabelText("本文")).toHaveValue("");
  });

  it("says when the same content was already there", async () => {
    postSourceMock.mockResolvedValue({ status: "unchanged", id: 1, chunks: 2 });
    render(<SourcesScreen />);
    await screen.findByText("会計の引継ぎ");

    fill("本文", "同じ本文");
    fill("置き場所のURL（任意）", "https://example.com/kaikei");
    fireEvent.click(screen.getByRole("button", { name: "追加する" }));

    expect(await screen.findByText(/変わっていない/)).toBeInTheDocument();
    expect(postSourceMock.mock.calls[0][0].url).toBe("https://example.com/kaikei");
  });

  it("keeps the input and shows the error when adding fails", async () => {
    postSourceMock.mockRejectedValue(new Error("boom"));
    render(<SourcesScreen />);
    await screen.findByText("会計の引継ぎ");

    fill("本文", "残しておきたい本文");
    fireEvent.click(screen.getByRole("button", { name: "追加する" }));

    expect(await screen.findByText(/追加できませんでした/)).toBeInTheDocument();
    expect(screen.getByLabelText("本文")).toHaveValue("残しておきたい本文");
  });

  it("does not submit a blank body", async () => {
    render(<SourcesScreen />);
    await screen.findByText("会計の引継ぎ");

    expect(screen.getByRole("button", { name: "追加する" })).toBeDisabled();
    fill("本文", "   ");
    expect(screen.getByRole("button", { name: "追加する" })).toBeDisabled();
  });

  it("reads a text file into the body and its name into the title", async () => {
    render(<SourcesScreen />);
    await screen.findByText("会計の引継ぎ");

    const file = new File(["# 引継ぎ\n通帳は金庫"], "引継ぎ書.md", { type: "text/markdown" });
    fireEvent.change(screen.getByLabelText("ファイルから読み込む"), { target: { files: [file] } });

    await waitFor(() => expect(screen.getByLabelText("本文")).toHaveValue("# 引継ぎ\n通帳は金庫"));
    expect(screen.getByLabelText("題名（任意）")).toHaveValue("引継ぎ書");
  });

  it("deletes a source after confirmation", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    deleteSourceMock.mockResolvedValue({ status: "deleted" });
    render(<SourcesScreen />);

    fireEvent.click(await screen.findByRole("button", { name: "「会計の引継ぎ」を削除" }));

    await waitFor(() => expect(deleteSourceMock).toHaveBeenCalledWith(1));
    await waitFor(() => expect(getSourcesMock).toHaveBeenCalledTimes(2));
  });

  it("does nothing when the deletion is cancelled", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<SourcesScreen />);

    fireEvent.click(await screen.findByRole("button", { name: "「会計の引継ぎ」を削除" }));

    expect(deleteSourceMock).not.toHaveBeenCalled();
  });

  it("shows an error when the deletion fails", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    deleteSourceMock.mockRejectedValue(new Error("boom"));
    render(<SourcesScreen />);

    fireEvent.click(await screen.findByRole("button", { name: "「会計の引継ぎ」を削除" }));

    expect(await screen.findByText(/削除できませんでした/)).toBeInTheDocument();
  });
});

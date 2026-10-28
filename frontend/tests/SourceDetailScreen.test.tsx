import { SourceDetailScreen } from "@/components/SourceDetailScreen";
import type { SourceDetail } from "@/lib/api-types";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getSourceMock = vi.fn();
const deleteSourceMock = vi.fn();
const pushMock = vi.fn();

vi.mock("@/lib/api-client", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-client")>("@/lib/api-client");
  return {
    ApiError: actual.ApiError,
    getSource: (...args: unknown[]) => getSourceMock(...args),
    deleteSource: (...args: unknown[]) => deleteSourceMock(...args),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
}));

const DETAIL: SourceDetail = {
  id: 4,
  provider: "manual",
  title: "会計の引継ぎ",
  url: "https://example.com/kaikei",
  author_name: "アリス",
  updated_at: "2026-10-28T12:00:00",
  chunks: 2,
  can_delete: true,
  body: "通帳は部室の金庫。\n\n印鑑は会計が持つ。",
};

beforeEach(() => {
  getSourceMock.mockReset().mockResolvedValue(DETAIL);
  deleteSourceMock.mockReset();
  pushMock.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("SourceDetailScreen (#74)", () => {
  it("shows the source's body and details", async () => {
    render(<SourceDetailScreen sourceId="4" />);

    expect(await screen.findByRole("heading", { name: "会計の引継ぎ" })).toBeInTheDocument();
    expect(getSourceMock).toHaveBeenCalledWith(4);
    expect(screen.getByText(/通帳は部室の金庫。/)).toBeInTheDocument();
    expect(screen.getByText(/アリス/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "元の文書を開く" })).toHaveAttribute(
      "href",
      "https://example.com/kaikei",
    );
  });

  it("names an untitled source and hides delete when not allowed", async () => {
    getSourceMock.mockResolvedValue({ ...DETAIL, title: null, url: null, can_delete: false });
    render(<SourceDetailScreen sourceId="4" />);

    expect(await screen.findByRole("heading", { name: "（題名なし）" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "削除" })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "元の文書を開く" })).not.toBeInTheDocument();
  });

  it("says when the source does not exist", async () => {
    const { ApiError } = await import("@/lib/api-client");
    getSourceMock.mockRejectedValue(new ApiError(404, "資料が見つかりません。"));
    render(<SourceDetailScreen sourceId="99" />);

    expect(await screen.findByText("資料が見つかりません。")).toBeInTheDocument();
  });

  it("does not ask for an id that is not a number", async () => {
    render(<SourceDetailScreen sourceId="abc" />);

    expect(await screen.findByText("資料が見つかりません。")).toBeInTheDocument();
    expect(getSourceMock).not.toHaveBeenCalled();
  });

  it("shows a generic error when loading fails", async () => {
    getSourceMock.mockRejectedValue(new Error("boom"));
    render(<SourceDetailScreen sourceId="4" />);

    expect(await screen.findByText("資料を読み込めませんでした。")).toBeInTheDocument();
  });

  it("deletes and returns to the list", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    deleteSourceMock.mockResolvedValue({ status: "deleted" });
    render(<SourceDetailScreen sourceId="4" />);

    fireEvent.click(await screen.findByRole("button", { name: "削除" }));

    await waitFor(() => expect(deleteSourceMock).toHaveBeenCalledWith(4));
    expect(pushMock).toHaveBeenCalledWith("/sources");
  });

  it("stays when the deletion is cancelled or fails", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<SourceDetailScreen sourceId="4" />);
    fireEvent.click(await screen.findByRole("button", { name: "削除" }));
    expect(deleteSourceMock).not.toHaveBeenCalled();

    confirm.mockReturnValue(true);
    deleteSourceMock.mockRejectedValue(new Error("boom"));
    fireEvent.click(screen.getByRole("button", { name: "削除" }));

    expect(await screen.findByText(/削除できませんでした/)).toBeInTheDocument();
    expect(pushMock).not.toHaveBeenCalled();
  });
});

import type { CurrentUserContextValue } from "@/components/CurrentUserProvider";
import { InboxScreen } from "@/components/InboxScreen";
import type { HandoffResponse, InboxItem } from "@/lib/api-types";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const useCurrentUserMock = vi.fn<() => CurrentUserContextValue>();
vi.mock("@/components/CurrentUserProvider", () => ({
  useCurrentUser: () => useCurrentUserMock(),
}));

const getInboxMock = vi.fn();
const getHandoffMock = vi.fn();
const postAnswerMock = vi.fn();
const advanceSessionMock = vi.fn();
vi.mock("@/lib/api-client", () => ({
  getInbox: (...args: unknown[]) => getInboxMock(...args),
  getHandoff: (...args: unknown[]) => getHandoffMock(...args),
  postAnswer: (...args: unknown[]) => postAnswerMock(...args),
  advanceSession: (...args: unknown[]) => advanceSessionMock(...args),
  ApiError: class ApiError extends Error {
    readonly status: number;
    constructor(status: number, message: string) {
      super(message);
      this.name = "ApiError";
      this.status = status;
    }
  },
}));

function asUser(id: string | null, name?: string): CurrentUserContextValue {
  return {
    employees: [],
    currentUserId: id,
    currentUser: id && name ? { id, name, dept: "技術部" } : null,
    setCurrentUserId: vi.fn(),
    loading: false,
    error: false,
    reload: vi.fn(),
    canSwitch: false,
  };
}

const ITEM: InboxItem = {
  session_id: "sess-42",
  question_id: "api_q1",
  question: "UTM の移行時に気をつけることは？",
  topics: ["ネットワーク", "セキュリティ"],
  asker: { id: "E010", name: "藤田 悠斗", dept: "第3営業部" },
  consult_method: "chat",
  created_at: "2026-08-23T09:30:00",
};

const ITEM2: InboxItem = {
  session_id: "sess-43",
  question_id: "api_q2",
  question: "VPN の帯域制限について",
  topics: ["ネットワーク"],
  asker: { id: "E011", name: "森田 恵", dept: "広報部" },
  // The two items differ so the list badge is visibly per-item (#245).
  consult_method: "direct",
  created_at: "2026-08-23T10:00:00",
};

function handoffFor(item: InboxItem): HandoffResponse {
  return {
    session_id: item.session_id,
    question: item.question,
    asker: item.asker,
    topics: item.topics,
    products: [],
    missing: [],
    responder: {
      person_id: "E001",
      name: "高梨 健太",
      dept: "技術部",
      score: 0.9,
      confidence: "高",
      reasons: [{ type: "answers", detail: "類似の質問に過去3件回答" }],
    },
    draft: `${item.asker.name}さん向けの下書きです。`,
    reuse_count: 3,
    helpful_answer_count: 1,
  };
}

beforeEach(() => {
  useCurrentUserMock.mockReset();
  getInboxMock.mockReset();
  getHandoffMock.mockReset();
  postAnswerMock.mockReset();
  advanceSessionMock.mockReset();
  postAnswerMock.mockResolvedValue({ session_id: "sess-42", status: "resumed" });
  advanceSessionMock.mockResolvedValue(undefined);
  // Since #246 the inbox renders the detail pane (AnswerScreen) for the
  // auto-selected first item, so `getHandoff` fires even in tests that only
  // assert on the list. A default keeps such a call harmless whenever it
  // lands; tests that care still override it. (#303)
  getHandoffMock.mockResolvedValue(handoffFor(ITEM));
});

// No `afterEach(vi.restoreAllMocks)` here on purpose. There is no `vi.spyOn` in
// this file, so it would have nothing to restore — but it *does* strip the mock
// implementations, and since #246 the inbox renders the detail pane, whose
// `getHandoff` effect can still land after a test that only asserted on the
// list has handed control back. A stripped mock returns undefined there and
// `getHandoff(...).then` throws. `beforeEach` resets every mock explicitly, so
// isolation between tests does not depend on restoring. (#303)

describe("InboxScreen", () => {
  it("shows loading until the current user resolves", () => {
    useCurrentUserMock.mockReturnValue(asUser(null));
    render(<InboxScreen />);
    expect(screen.getByRole("link", { name: "ホームへ戻る" })).toHaveAttribute("href", "/");
    expect(screen.getByText("読み込み中…")).toBeInTheDocument();
    expect(getInboxMock).not.toHaveBeenCalled();
  });

  it("fetches the inbox and shows the first item's detail without an extra click", async () => {
    useCurrentUserMock.mockReturnValue(asUser("E001", "高梨 健太"));
    getInboxMock.mockResolvedValue([ITEM]);
    getHandoffMock.mockResolvedValue(handoffFor(ITEM));
    render(<InboxScreen />);

    await waitFor(() => expect(getInboxMock).toHaveBeenCalledWith("E001"));
    // The `who` line comes from `useCurrentUser()`, so it is on screen from the
    // first render. The list is what the fetch produces, so wait on THAT.
    expect(screen.getByText("高梨 健太 さん宛てに届いた質問です。")).toBeInTheDocument();
    const listItem = await screen.findByRole("button", { name: /藤田 悠斗 さんからの質問/ });

    // Scope the rest to the list item. Since #246 this page renders the list AND
    // the detail pane, so the question text exists TWICE — once here, once as
    // the detail's heading — and a bare `getByText` throws "Found multiple
    // elements". It is a race, too: whether the second one exists yet depends on
    // when the handoff fetch lands. Same defect the e2e hit in #283/#284.
    expect(listItem).toHaveTextContent(ITEM.question);
    expect(listItem).toHaveTextContent("ネットワーク");
    // Naive-UTC on the wire -> JST display is +9h (#418).
    expect(listItem).toHaveTextContent("2026-08-23 18:30");

    // The detail pane (AnswerScreen) renders for the first item with no click.
    await waitFor(() => expect(getHandoffMock).toHaveBeenCalledWith("sess-42"));
    expect(await screen.findByRole("heading", { name: "あなたに届いた質問" })).toBeInTheDocument();
    expect(screen.getByText("あなたが選ばれた理由")).toBeInTheDocument();
    expect(screen.getByText("依頼内容（下書き）")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "引き受ける" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "今は難しい" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "自分より適任がいる" })).toBeInTheDocument();
  });

  it("switches the detail pane when a different item is clicked", async () => {
    useCurrentUserMock.mockReturnValue(asUser("E001", "高梨 健太"));
    getInboxMock.mockResolvedValue([ITEM, ITEM2]);
    getHandoffMock.mockImplementation((sessionId: string) =>
      Promise.resolve(handoffFor(sessionId === ITEM.session_id ? ITEM : ITEM2)),
    );
    render(<InboxScreen />);

    await waitFor(() => expect(getHandoffMock).toHaveBeenCalledWith(ITEM.session_id));
    // The question appears both in the list preview and the detail heading;
    // scope to the heading (level 2, per AnswerScreen) to pick the detail pane.
    expect(
      await screen.findByRole("heading", { name: ITEM.question, level: 2 }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /森田 恵 さんからの質問/ }));

    await waitFor(() => expect(getHandoffMock).toHaveBeenCalledWith(ITEM2.session_id));
    expect(
      await screen.findByRole("heading", { name: ITEM2.question, level: 2 }),
    ).toBeInTheDocument();
  });

  it("keeps the accepted item's confirmation visible and drops it from the list", async () => {
    useCurrentUserMock.mockReturnValue(asUser("E001", "高梨 健太"));
    getInboxMock.mockResolvedValueOnce([ITEM]).mockResolvedValueOnce([]);
    getHandoffMock.mockResolvedValue(handoffFor(ITEM));
    render(<InboxScreen />);

    await screen.findByRole("button", { name: "引き受ける" });
    fireEvent.click(screen.getByRole("button", { name: "引き受ける" }));

    expect(await screen.findByRole("heading", { name: /お引き受け/ })).toBeInTheDocument();
    await waitFor(() => expect(getInboxMock).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(screen.getByText("いまは届いている質問はありません。")).toBeInTheDocument(),
    );
    // The confirmation stays up even though the item is gone from the list.
    expect(screen.getByRole("heading", { name: /お引き受け/ })).toBeInTheDocument();
  });

  it("drops the item immediately (no lingering confirmation) when 今は難しい is clicked", async () => {
    useCurrentUserMock.mockReturnValue(asUser("E001", "高梨 健太"));
    getInboxMock.mockResolvedValueOnce([ITEM, ITEM2]).mockResolvedValueOnce([ITEM2]);
    getHandoffMock.mockImplementation((sessionId: string) =>
      Promise.resolve(handoffFor(sessionId === ITEM.session_id ? ITEM : ITEM2)),
    );
    render(<InboxScreen />);

    await screen.findByRole("button", { name: "今は難しい" });
    fireEvent.click(screen.getByRole("button", { name: "今は難しい" }));

    // No "承知しました" confirmation lingers for the declined item; the pane
    // moves straight to the next pending item's detail.
    await waitFor(() => expect(getInboxMock).toHaveBeenCalledTimes(2));
    expect(
      await screen.findByRole("heading", { name: ITEM2.question, level: 2 }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "承知しました" })).not.toBeInTheDocument();
  });

  it("shows an empty state when there are no pending handoffs", async () => {
    useCurrentUserMock.mockReturnValue(asUser("E002", "佐藤 花子"));
    getInboxMock.mockResolvedValue([]);
    render(<InboxScreen />);

    await waitFor(() =>
      expect(screen.getByText("いまは届いている質問はありません。")).toBeInTheDocument(),
    );
  });

  it("shows an error state when the fetch fails", async () => {
    useCurrentUserMock.mockReturnValue(asUser("E001", "高梨 健太"));
    getInboxMock.mockRejectedValue(new Error("network"));
    render(<InboxScreen />);

    expect(await screen.findByRole("alert")).toHaveTextContent("受信箱の取得に失敗しました");
  });

  it("falls back to 匿名 when the asker has no name", async () => {
    useCurrentUserMock.mockReturnValue(asUser("E001", "高梨 健太"));
    getInboxMock.mockResolvedValue([{ ...ITEM, asker: { id: "E010" }, topics: [] }]);
    getHandoffMock.mockResolvedValue(handoffFor(ITEM));
    render(<InboxScreen />);

    await waitFor(() => expect(screen.getByText("匿名 さんからの質問")).toBeInTheDocument());
  });
  it("labels each item with the asker's chosen consultation method (#245)", async () => {
    useCurrentUserMock.mockReturnValue(asUser("E001"));
    getInboxMock.mockResolvedValue([ITEM, ITEM2]);
    getHandoffMock.mockImplementation((sessionId: string) =>
      Promise.resolve(handoffFor(sessionId === ITEM.session_id ? ITEM : ITEM2)),
    );
    render(<InboxScreen />);

    // The responder must be able to tell BEFORE accepting: "直接相談" never
    // opens a chat thread, so it changes what accepting commits them to.
    // Both badges live on the list buttons themselves, so the responder sees
    // them without opening anything.
    const chat = await screen.findByRole("button", { name: /藤田 悠斗 さんからの質問/ });
    expect(within(chat).getByText("チャットで相談")).toBeInTheDocument();
    const direct = screen.getByRole("button", { name: /森田 恵 さんからの質問/ });
    expect(within(direct).getByText("直接相談")).toBeInTheDocument();
  });
});

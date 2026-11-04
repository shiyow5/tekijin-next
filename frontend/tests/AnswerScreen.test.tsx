import { AnswerScreen } from "@/components/AnswerScreen";
import type { HandoffResponse } from "@/lib/api-types";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getHandoffMock = vi.fn();
const postAnswerMock = vi.fn();
const advanceSessionMock = vi.fn();

vi.mock("@/lib/api-client", () => ({
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

const HANDOFF: HandoffResponse = {
  session_id: "s1",
  question: "UTM移行時の注意点について",
  asker: { id: "E010", name: "藤田 悠斗", dept: "第3営業部" },
  topics: ["ネットワーク・VPN"],
  products: ["UTM"],
  situation: "移行",
  missing: ["予算感"],
  responder: {
    person_id: "E001",
    name: "高梨 健太",
    dept: "技術部",
    score: 0.9,
    confidence: "高",
    reasons: [
      { type: "recency", detail: "直近1か月以内の関連実績あり" },
      { type: "answers", detail: "類似の質問に過去5件回答（うち有用と評価3件）" },
    ],
  },
  draft: "高梨さんへの依頼文です。",
  reuse_count: 7,
  helpful_answer_count: 5,
};

const THREE_BUTTONS = ["引き受ける", "今は難しい", "自分より適任がいる"];

describe("AnswerScreen", () => {
  beforeEach(() => {
    getHandoffMock.mockReset();
    postAnswerMock.mockReset();
    advanceSessionMock.mockReset();
    postAnswerMock.mockResolvedValue({ session_id: "s1", status: "resumed" });
    advanceSessionMock.mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("shows a loading placeholder before the handoff arrives", () => {
    getHandoffMock.mockReturnValue(new Promise(() => {})); // never resolves
    render(<AnswerScreen sessionId="s1" />);
    expect(screen.getByText(/読み込み中|準備中/)).toBeInTheDocument();
  });

  it("renders the question, asker, selection reasons and reuse count", async () => {
    getHandoffMock.mockResolvedValue(HANDOFF);
    render(<AnswerScreen sessionId="s1" showBackLink />);

    expect(await screen.findByText("UTM移行時の注意点について")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "受信箱へ戻る" })).toHaveAttribute("href", "/inbox");
    expect(screen.getByText(/藤田 悠斗/)).toBeInTheDocument();
    expect(screen.getByText(/第3営業部/)).toBeInTheDocument();
    // selection reasons (verbatim details from the backend).
    expect(screen.getByText(/直近1か月以内の関連実績あり/)).toBeInTheDocument();
    expect(screen.getByText(/類似の質問に過去5件回答/)).toBeInTheDocument();
    // draft (the filled-in 依頼文 the responder receives).
    expect(screen.getByText(/高梨さんへの依頼文です/)).toBeInTheDocument();
    // reuse count (the 見返り at the bottom).
    expect(screen.getByText(/7/)).toBeInTheDocument();
  });

  it("renders the three action buttons at an equal size (F-09: 断るを二級にしない)", async () => {
    getHandoffMock.mockResolvedValue(HANDOFF);
    render(<AnswerScreen sessionId="s1" />);
    await screen.findByText("UTM移行時の注意点について");

    const buttons = THREE_BUTTONS.map((name) => screen.getByRole("button", { name }));
    // Every button opts into the same equal-size contract, so none is demoted.
    for (const b of buttons) {
      expect(b).toHaveClass("min-h-[56px]");
      expect(b).toHaveClass("flex-1");
    }
    // The two decline options share identical classes (same visual weight).
    const [, defer, refer] = buttons;
    expect(refer.className).toBe(defer.className);
  });

  it("sends accepted and shows a confirmation when 引き受ける is clicked", async () => {
    getHandoffMock.mockResolvedValue(HANDOFF);
    render(<AnswerScreen sessionId="s1" />);
    await screen.findByText("UTM移行時の注意点について");

    fireEvent.click(screen.getByRole("button", { name: "引き受ける" }));

    await waitFor(() =>
      expect(postAnswerMock).toHaveBeenCalledWith({ session_id: "s1", outcome: "accepted" }),
    );
    expect(await screen.findByRole("heading", { name: /お引き受け/ })).toBeInTheDocument();
    // #176: the app captures the accept, not the answer text, so the copy must
    // connect the asker to this person — never promise a delivered answer.
    expect(screen.getByText(/お繋ぎします/)).toBeInTheDocument();
    expect(screen.queryByText(/回答をお届け/)).not.toBeInTheDocument();
  });

  it("captures a typed answer body on accept (#274)", async () => {
    getHandoffMock.mockResolvedValue(HANDOFF);
    render(<AnswerScreen sessionId="s1" />);
    await screen.findByText("UTM移行時の注意点について");

    fireEvent.change(screen.getByLabelText("回答内容（任意）"), {
      target: { value: "移行前にポリシーを棚卸ししてください。" },
    });
    fireEvent.click(screen.getByRole("button", { name: "引き受ける" }));

    await waitFor(() =>
      expect(postAnswerMock).toHaveBeenCalledWith({
        session_id: "s1",
        outcome: "accepted",
        answer_body: "移行前にポリシーを棚卸ししてください。",
      }),
    );
  });

  it("accepts without an answer body when the field is left blank (#274)", async () => {
    getHandoffMock.mockResolvedValue(HANDOFF);
    render(<AnswerScreen sessionId="s1" />);
    await screen.findByText("UTM移行時の注意点について");

    // Field left empty -> no answer_body key in the payload.
    fireEvent.click(screen.getByRole("button", { name: "引き受ける" }));

    await waitFor(() =>
      expect(postAnswerMock).toHaveBeenCalledWith({ session_id: "s1", outcome: "accepted" }),
    );
  });

  it("links back to the inbox after answering in standalone view (#126: label matches destination)", async () => {
    getHandoffMock.mockResolvedValue(HANDOFF);
    render(<AnswerScreen sessionId="s1" showBackLink />);
    await screen.findByText("UTM移行時の注意点について");
    fireEvent.click(screen.getByRole("button", { name: "引き受ける" }));

    const back = await screen.findByRole("link", { name: "受信箱へ戻る" });
    expect(back).toHaveAttribute("href", "/inbox");
    expect(screen.getAllByRole("link", { name: "受信箱へ戻る" })).toHaveLength(1);
  });

  it("hides the inbox back-link after answering when embedded (#513: showBackLink defaults false)", async () => {
    getHandoffMock.mockResolvedValue(HANDOFF);
    render(<AnswerScreen sessionId="s1" />);
    await screen.findByText("UTM移行時の注意点について");
    fireEvent.click(screen.getByRole("button", { name: "引き受ける" }));

    await screen.findByRole("heading", { name: /お引き受け/ });
    expect(screen.queryByRole("link", { name: "受信箱へ戻る" })).toBeNull();
  });

  it("links to the chat thread after accepting when a recommendation_id is present (#224)", async () => {
    getHandoffMock.mockResolvedValue({
      ...HANDOFF,
      recommendation_id: 42,
    } satisfies HandoffResponse);
    render(<AnswerScreen sessionId="s1" />);
    await screen.findByText("UTM移行時の注意点について");
    fireEvent.click(screen.getByRole("button", { name: "引き受ける" }));

    const chatLink = await screen.findByRole("link", { name: "チャットを開く" });
    expect(chatLink).toHaveAttribute("href", "/chat?thread=42");
  });

  it("does not show a chat link after declining (#224: only 引き受ける opens a thread)", async () => {
    getHandoffMock.mockResolvedValue({
      ...HANDOFF,
      recommendation_id: 42,
    } satisfies HandoffResponse);
    render(<AnswerScreen sessionId="s1" />);
    await screen.findByText("UTM移行時の注意点について");
    fireEvent.click(screen.getByRole("button", { name: "今は難しい" }));

    await screen.findByRole("heading", { name: "承知しました" });
    expect(screen.queryByRole("link", { name: "チャットを開く" })).toBeNull();
  });

  it("sends declined when 今は難しい is clicked", async () => {
    getHandoffMock.mockResolvedValue(HANDOFF);
    render(<AnswerScreen sessionId="s1" />);
    await screen.findByText("UTM移行時の注意点について");

    fireEvent.click(screen.getByRole("button", { name: "今は難しい" }));

    await waitFor(() =>
      expect(postAnswerMock).toHaveBeenCalledWith({ session_id: "s1", outcome: "declined" }),
    );
  });

  it("sends declined when 自分より適任がいる is clicked (interim mapping)", async () => {
    getHandoffMock.mockResolvedValue(HANDOFF);
    render(<AnswerScreen sessionId="s1" />);
    await screen.findByText("UTM移行時の注意点について");

    fireEvent.click(screen.getByRole("button", { name: "自分より適任がいる" }));

    await waitFor(() =>
      expect(postAnswerMock).toHaveBeenCalledWith({ session_id: "s1", outcome: "declined" }),
    );
    // #176: refer == defer's auto reroute; the copy must not promise a named referral.
    expect(await screen.findByText(/別の候補者を自動でお探しします/)).toBeInTheDocument();
    expect(screen.queryByText(/別の候補者へお繋ぎします/)).not.toBeInTheDocument();
  });

  it("shows a 'gone' message when the session is no longer awaiting a response", async () => {
    const { ApiError } = await import("@/lib/api-client");
    getHandoffMock.mockRejectedValue(new ApiError(404, "no responder handoff"));
    render(<AnswerScreen sessionId="s1" />);

    expect(await screen.findByRole("alert")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "引き受ける" })).toBeNull();
  });

  it("shows a load-failure message (not 'gone') for a generic load error", async () => {
    getHandoffMock.mockRejectedValue(new Error("network"));
    render(<AnswerScreen sessionId="s1" />);

    expect(await screen.findByRole("alert")).toHaveTextContent(/読み込みに失敗/);
  });

  it("links to the chat list on a 'gone' load (#224: recovery after reload)", async () => {
    const { ApiError } = await import("@/lib/api-client");
    getHandoffMock.mockRejectedValue(new ApiError(404, "no responder handoff"));
    render(<AnswerScreen sessionId="s1" />);

    const chatLink = await screen.findByRole("link", { name: "チャット一覧を見る" });
    expect(chatLink).toHaveAttribute("href", "/chat");
  });

  it("does not link to the chat list on a generic (non-'gone') load error", async () => {
    getHandoffMock.mockRejectedValue(new Error("network"));
    render(<AnswerScreen sessionId="s1" />);

    await screen.findByRole("alert");
    expect(screen.queryByRole("link", { name: "チャット一覧を見る" })).toBeNull();
  });

  it("hides the inbox back-link on a load error when embedded (#513: showBackLink defaults false)", async () => {
    getHandoffMock.mockRejectedValue(new Error("network"));
    render(<AnswerScreen sessionId="s1" />);

    await screen.findByRole("alert");
    expect(screen.queryByRole("link", { name: "受信箱へ戻る" })).toBeNull();
  });

  it("still shows the inbox back-link on a load error in standalone view (#513: showBackLink)", async () => {
    getHandoffMock.mockRejectedValue(new Error("network"));
    render(<AnswerScreen sessionId="s1" showBackLink />);

    const back = await screen.findByRole("link", { name: "受信箱へ戻る" });
    expect(back).toHaveAttribute("href", "/inbox");
  });

  it("handles a handoff with no responder, empty slots and no reasons", async () => {
    getHandoffMock.mockResolvedValue({
      ...HANDOFF,
      asker: { id: "E010", name: null, dept: null },
      topics: [],
      products: [],
      situation: null,
      missing: [],
      responder: null,
      reuse_count: 0,
      helpful_answer_count: 0,
    } satisfies HandoffResponse);
    render(<AnswerScreen sessionId="s1" />);

    expect(await screen.findByText("UTM移行時の注意点について")).toBeInTheDocument();
    expect(screen.getByText("質問者さん")).toBeInTheDocument(); // name fallback
    expect(screen.getByText("根拠を確認中…")).toBeInTheDocument(); // empty reasons
    // still fully actionable even without candidate metadata.
    expect(screen.getByRole("button", { name: "引き受ける" })).toBeEnabled();
  });

  it("surfaces a retryable error when the outcome submit fails", async () => {
    getHandoffMock.mockResolvedValue(HANDOFF);
    postAnswerMock.mockRejectedValueOnce(new Error("network"));
    render(<AnswerScreen sessionId="s1" />);
    await screen.findByText("UTM移行時の注意点について");

    fireEvent.click(screen.getByRole("button", { name: "引き受ける" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/失敗|お試し/);
    // still interactive (retryable), not dead-ended.
    expect(screen.getByRole("button", { name: "引き受ける" })).toBeEnabled();
  });
});

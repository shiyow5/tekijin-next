import {
  ApiError,
  advanceSession,
  correctInterpretation,
  deleteQuestion,
  excludeHandoffCandidate,
  getDashboard,
  getDocument,
  getEmployees,
  getHandoff,
  getInbox,
  getRecentQuestions,
  getRetrospectiveContext,
  getSlackAuthorizeUrl,
  getSlackLoginUrl,
  getSlackStatus,
  postAnswer,
  postAsk,
  postSlackUnlink,
  redeemDiscordLogin,
  regenerateHandoffDraft,
  requestDocumentFallback,
  resolveQuestion,
  selectHandoffCandidate,
  structureQuestion,
} from "@/lib/api-client";
import type {
  AskRequest,
  ConsultRetrospectiveContext,
  HandoffResponse,
  ResumeRequest,
} from "@/lib/api-types";
import { DEFAULT_API_BASE_URL } from "@/lib/config";
import { describe, expect, it, vi } from "vitest";

const REQUEST: AskRequest = {
  asker_id: 1,
  question: "UTMの移行時の注意点",
  session_id: "abc-123",
};

function jsonResponse(body: unknown, init: { ok?: boolean; status?: number } = {}): Response {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    json: async () => body,
  } as Response;
}

describe("postAsk", () => {
  it("POSTs to {base}/ask with JSON headers and body", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ session_id: "abc-123", status: "accepted" }));

    const result = await postAsk(REQUEST, { fetchImpl });

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/ask`);
    expect(init?.method).toBe("POST");
    expect(init?.headers).toMatchObject({ "Content-Type": "application/json" });
    expect(JSON.parse(init?.body as string)).toEqual(REQUEST);
    expect(result).toEqual({ session_id: "abc-123", status: "accepted" });
  });

  it("uses an explicit baseUrl override", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ session_id: "abc-123", status: "accepted" }));

    await postAsk(REQUEST, { fetchImpl, baseUrl: "https://api.example.com" });

    expect(fetchImpl.mock.calls[0][0]).toBe("https://api.example.com/ask");
  });

  it("normalizes a trailing slash on the baseUrl override", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ session_id: "abc-123", status: "accepted" }));

    await postAsk(REQUEST, { fetchImpl, baseUrl: "https://api.example.com/" });

    expect(fetchImpl.mock.calls[0][0]).toBe("https://api.example.com/ask");
  });

  it("throws ApiError using the `error` field when `detail` is absent", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ error: "internal error" }, { ok: false, status: 500 }));

    await expect(postAsk(REQUEST, { fetchImpl })).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
      message: "internal error",
    });
  });

  it("throws ApiError with the status and detail on a non-2xx response", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        jsonResponse({ detail: "question must not be blank" }, { ok: false, status: 422 }),
      );

    await expect(postAsk(REQUEST, { fetchImpl })).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      message: "question must not be blank",
    });
  });

  it("throws ApiError with a generic message when the error body is not JSON", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => {
        throw new Error("not json");
      },
    } as unknown as Response);

    await expect(postAsk(REQUEST, { fetchImpl })).rejects.toBeInstanceOf(ApiError);
    await expect(postAsk(REQUEST, { fetchImpl })).rejects.toThrow("status 500");
  });
});

describe("postAnswer", () => {
  const RESUME: ResumeRequest = { session_id: "abc-123", reply: "UTMです" };

  it("POSTs to {base}/answer with JSON headers and the resume body", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ session_id: "abc-123", status: "accepted" }));

    const result = await postAnswer(RESUME, { fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/answer`);
    expect(init?.method).toBe("POST");
    expect(init?.headers).toMatchObject({ "Content-Type": "application/json" });
    expect(JSON.parse(init?.body as string)).toEqual(RESUME);
    expect(result).toEqual({ session_id: "abc-123", status: "accepted" });
  });

  it("throws ApiError on a non-2xx response", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ detail: "bad" }, { ok: false, status: 409 }));

    await expect(postAnswer(RESUME, { fetchImpl })).rejects.toMatchObject({
      name: "ApiError",
      status: 409,
    });
  });
});

describe("requestDocumentFallback", () => {
  it("POSTs the same session to the document-fallback endpoint", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ session_id: "abc-123", status: "handoff_queued" }));

    await requestDocumentFallback({ session_id: "abc-123" }, { fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/handoff/document-fallback`);
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({ session_id: "abc-123" });
  });
});

describe("getHandoff", () => {
  const HANDOFF: HandoffResponse = {
    session_id: "abc-123",
    question: "UTM移行時の注意点",
    asker: { id: "E010", name: "藤田 悠斗", dept: "第3営業部" },
    topics: ["ネットワーク・VPN"],
    products: ["UTM"],
    situation: "移行",
    missing: [],
    responder: {
      person_id: "E001",
      name: "高梨 健太",
      dept: "技術部",
      score: 0.9,
      confidence: "高",
      reasons: [{ type: "cert", detail: "情報処理安全確保支援士" }],
    },
    draft: "高梨さんへの依頼文",
    reuse_count: 7,
    helpful_answer_count: 5,
  };

  it("GETs {base}/handoff/{id} and returns the payload", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(HANDOFF));

    const result = await getHandoff("abc-123", { fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/handoff/abc-123`);
    expect(init?.method).toBe("GET");
    expect(result).toEqual(HANDOFF);
  });

  it("url-encodes the session id path segment", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(HANDOFF));

    await getHandoff("a b/c", { fetchImpl });

    expect(fetchImpl.mock.calls[0][0]).toBe(`${DEFAULT_API_BASE_URL}/handoff/a%20b%2Fc`);
  });

  it("throws ApiError on a non-2xx response (e.g. 404 no handoff)", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        jsonResponse({ detail: "no responder handoff" }, { ok: false, status: 404 }),
      );

    await expect(getHandoff("gone", { fetchImpl })).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
    });
  });
});

describe("getRetrospectiveContext", () => {
  const CONTEXT: ConsultRetrospectiveContext = {
    session_id: "abc-123",
    question_id: "q_0001",
    question: "拠点間VPNが不安定です",
    consult_method: "direct",
    responder: { person_id: "E001", name: "高梨 健太" },
    already_recorded: false,
  };

  it("GETs {base}/consult-retrospective/{id} and returns the payload", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(CONTEXT));

    const result = await getRetrospectiveContext("abc-123", { fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/consult-retrospective/abc-123`);
    expect(init?.method).toBe("GET");
    expect(result).toEqual(CONTEXT);
  });

  it("url-encodes the session id path segment", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(CONTEXT));

    await getRetrospectiveContext("a b/c", { fetchImpl });

    expect(fetchImpl.mock.calls[0][0]).toBe(
      `${DEFAULT_API_BASE_URL}/consult-retrospective/a%20b%2Fc`,
    );
  });
});

describe("advanceSession", () => {
  function sseResponse(): Response {
    return {
      ok: true,
      status: 200,
      text: async () => "event: done\ndata: {}\n\n",
    } as Response;
  }

  it("GETs {base}/events/{id} and drains the stream to completion", async () => {
    const textSpy = vi.fn(async () => "event: done\ndata: {}\n\n");
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue({ ok: true, status: 200, text: textSpy } as unknown as Response);

    await advanceSession("abc-123", { fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/events/abc-123`);
    expect(init?.method).toBe("GET");
    expect(textSpy).toHaveBeenCalledTimes(1); // read to completion
  });

  it("url-encodes the session id path segment", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(sseResponse());
    await advanceSession("a b/c", { fetchImpl });
    expect(fetchImpl.mock.calls[0][0]).toBe(`${DEFAULT_API_BASE_URL}/events/a%20b%2Fc`);
  });

  it("swallows a transport error (best-effort: the outcome is already recorded)", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockRejectedValue(new Error("network"));
    await expect(advanceSession("x", { fetchImpl })).resolves.toBeUndefined();
  });
});

describe("getDashboard", () => {
  it("GETs {base}/dashboard and returns the payload", async () => {
    const body = { total_questions: 150, self_resolution_rate: 0.4, latest_eval: null };
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(body));

    const result = await getDashboard({ fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/dashboard`);
    expect(init?.method).toBe("GET");
    expect(result).toMatchObject(body);
  });

  it("throws ApiError on a non-2xx response", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ detail: "boom" }, { ok: false, status: 500 }));
    await expect(getDashboard({ fetchImpl })).rejects.toMatchObject({
      name: "ApiError",
      status: 500,
    });
  });
});

describe("getEmployees", () => {
  it("GETs /employees and unwraps the employees array", async () => {
    const employees = [
      { id: "E001", name: "山田 太郎", dept: "営業部" },
      { id: "E002", name: "佐藤 花子", dept: "技術部" },
    ];
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ employees }));

    const result = await getEmployees({ fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/employees`);
    expect(init?.method).toBe("GET");
    expect(result).toEqual(employees);
  });

  it("throws ApiError on a non-2xx response", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ detail: "boom" }, { ok: false, status: 503 }));
    await expect(getEmployees({ fetchImpl })).rejects.toBeInstanceOf(ApiError);
  });
});

describe("getInbox", () => {
  const ITEMS = [
    {
      session_id: "sess-1",
      question_id: "api_q1",
      question: "UTM 移行の注意点",
      topics: ["ネットワーク"],
      asker: { id: "E010", name: "藤田 悠斗", dept: "第3営業部" },
      created_at: "2026-08-23T09:30:00",
    },
  ];

  it("GETs /inbox with the responder_id query and unwraps items", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ items: ITEMS }));

    const result = await getInbox("E001", { fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/inbox?responder_id=E001`);
    expect(init?.method).toBe("GET");
    expect(result).toEqual(ITEMS);
  });

  it("url-encodes the responder_id", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ items: [] }));
    await getInbox("E 1/2", { fetchImpl });
    expect(fetchImpl.mock.calls[0][0]).toBe(`${DEFAULT_API_BASE_URL}/inbox?responder_id=E%201%2F2`);
  });

  it("throws ApiError on a non-2xx response", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ detail: "bad" }, { ok: false, status: 422 }));
    await expect(getInbox("nope", { fetchImpl })).rejects.toBeInstanceOf(ApiError);
  });
});

describe("getRecentQuestions", () => {
  const ITEMS = [
    {
      question_id: "q1",
      title: "UTM移行",
      resolved: true,
      responder_name: "高梨 健太",
      created_at: "2026-08-20T10:00:00",
    },
  ];

  it("GETs /questions with the asker_id query and unwraps items", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ items: ITEMS }));

    const result = await getRecentQuestions("E001", { fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/questions?asker_id=E001`);
    expect(init?.method).toBe("GET");
    expect(result).toEqual(ITEMS);
  });

  it("appends a limit query when given (history screen, #208)", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ items: ITEMS }));

    await getRecentQuestions("E001", { fetchImpl, limit: 200 });

    expect(fetchImpl.mock.calls[0][0]).toBe(
      `${DEFAULT_API_BASE_URL}/questions?asker_id=E001&limit=200`,
    );
  });

  it("throws ApiError on a non-2xx response", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ detail: "bad" }, { ok: false, status: 422 }));
    await expect(getRecentQuestions("nope", { fetchImpl })).rejects.toBeInstanceOf(ApiError);
  });
});

describe("getDocument", () => {
  const DOC = {
    id: "doc_001",
    title: "社内IT手順書",
    body: "PCセットアップ手順…",
    source: "社内Wiki",
    updated_at: "2026-08-01T09:00:00",
  };

  it("GETs /documents/{id} (id encoded) and returns the document", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(DOC));

    const result = await getDocument("doc_001", { fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/documents/doc_001`);
    expect(init?.method).toBe("GET");
    expect(result).toEqual(DOC);
  });

  it("throws ApiError with the 404 status for an unknown document", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        jsonResponse({ detail: "document not found" }, { ok: false, status: 404 }),
      );
    await expect(getDocument("nope", { fetchImpl })).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
    });
  });
});

describe("selectHandoffCandidate", () => {
  it("POSTs /handoff/select with session_id and person_id", async () => {
    const response = {
      session_id: "s1",
      responder: {
        person_id: "E002",
        name: "鈴木",
        dept: null,
        score: 0.5,
        confidence: "中",
        reasons: [],
      },
      draft: "鈴木さん向けの下書き",
      recommendation_id: 42,
    };
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(response));
    const result = await selectHandoffCandidate(
      { session_id: "s1", person_id: "E002" },
      { fetchImpl },
    );
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/handoff/select`);
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({ session_id: "s1", person_id: "E002" });
    expect(result).toEqual(response);
  });

  it("throws ApiError with the 422 status for an unknown person_id", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ detail: "unknown" }, { ok: false, status: 422 }));
    await expect(
      selectHandoffCandidate({ session_id: "s1", person_id: "E099" }, { fetchImpl }),
    ).rejects.toMatchObject({ name: "ApiError", status: 422 });
  });
});

describe("excludeHandoffCandidate", () => {
  it("POSTs /handoff/exclude with session_id and person_id and returns the ack (#260)", async () => {
    const response = { session_id: "s1", status: "reroute_queued" };
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(response));
    const result = await excludeHandoffCandidate(
      { session_id: "s1", person_id: "E001" },
      { fetchImpl },
    );
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/handoff/exclude`);
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({ session_id: "s1", person_id: "E001" });
    expect(result).toEqual(response);
  });

  it("throws ApiError with the 422 status when the person is not the send target", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ detail: "not target" }, { ok: false, status: 422 }));
    await expect(
      excludeHandoffCandidate({ session_id: "s1", person_id: "E003" }, { fetchImpl }),
    ).rejects.toMatchObject({ name: "ApiError", status: 422 });
  });
});

describe("correctInterpretation", () => {
  it("POSTs /handoff/correct with session_id and supplement and returns the ack (#260)", async () => {
    const response = { session_id: "s1", status: "reinterpret_queued" };
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(response));
    const result = await correctInterpretation(
      { session_id: "s1", supplement: "対象は情報システム部です" },
      { fetchImpl },
    );
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/handoff/correct`);
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      session_id: "s1",
      supplement: "対象は情報システム部です",
    });
    expect(result).toEqual(response);
  });

  it("throws ApiError with the 409 status when a clarification is owed", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ detail: "awaiting" }, { ok: false, status: 409 }));
    await expect(
      correctInterpretation({ session_id: "s1", supplement: "補足" }, { fetchImpl }),
    ).rejects.toMatchObject({ name: "ApiError", status: 409 });
  });
});

describe("resolveQuestion", () => {
  it("POSTs /questions/{id}/resolve and returns the ack (#159)", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ question_id: "q1", resolved: true }));
    const result = await resolveQuestion("q1", { fetchImpl });
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/questions/q1/resolve`);
    expect(init?.method).toBe("POST");
    expect(result).toEqual({ question_id: "q1", resolved: true });
  });

  it("url-encodes the id and throws ApiError with the 403 status for a non-owner", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ detail: "forbidden" }, { ok: false, status: 403 }));
    await expect(resolveQuestion("a b/c", { fetchImpl })).rejects.toMatchObject({
      name: "ApiError",
      status: 403,
    });
    expect(fetchImpl.mock.calls[0][0]).toBe(`${DEFAULT_API_BASE_URL}/questions/a%20b%2Fc/resolve`);
  });
});

describe("regenerateHandoffDraft", () => {
  it("POSTs /handoff/redraft with session_id and returns the ack (#260)", async () => {
    const response = { session_id: "s1", status: "redraft_queued" };
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(response));
    const result = await regenerateHandoffDraft({ session_id: "s1" }, { fetchImpl });
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/handoff/redraft`);
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({ session_id: "s1" });
    expect(result).toEqual(response);
  });

  it("throws ApiError with the 409 status when a clarification is owed", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ detail: "awaiting" }, { ok: false, status: 409 }));
    await expect(regenerateHandoffDraft({ session_id: "s1" }, { fetchImpl })).rejects.toMatchObject(
      { name: "ApiError", status: 409 },
    );
  });
});

describe("structureQuestion", () => {
  it("POSTs /handoff/structure with session_id and returns the four fields (#475)", async () => {
    const response = {
      session_id: "s1",
      summary: "起きていること",
      environment: "M2 Mac",
      tried: "再ビルド",
      blocker: "原因不明",
    };
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(response));
    const result = await structureQuestion({ session_id: "s1" }, { fetchImpl });
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/handoff/structure`);
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({ session_id: "s1" });
    expect(result).toEqual(response);
  });

  it("throws ApiError with the 404 status when the session has no question", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ detail: "no question" }, { ok: false, status: 404 }));
    await expect(structureQuestion({ session_id: "s1" }, { fetchImpl })).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
    });
  });
});

describe("deleteQuestion", () => {
  it("DELETEs {base}/questions/{id} with auth headers and returns the ack (#207)", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ question_id: "api_del1", deleted: true }));

    const result = await deleteQuestion("api_del1", { fetchImpl });

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/questions/api_del1`);
    expect(init?.method).toBe("DELETE");
    expect(result).toEqual({ question_id: "api_del1", deleted: true });
  });

  it("URL-encodes the question id", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ question_id: "a/b", deleted: true }));

    await deleteQuestion("a/b", { fetchImpl });

    expect(fetchImpl.mock.calls[0][0]).toBe(`${DEFAULT_API_BASE_URL}/questions/a%2Fb`);
  });

  it("throws ApiError on a non-2xx response (403/404)", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        jsonResponse({ detail: "question not found" }, { ok: false, status: 404 }),
      );

    await expect(deleteQuestion("nope", { fetchImpl })).rejects.toMatchObject({
      status: 404,
    });
  });
});

describe("getSlackStatus", () => {
  it("GETs {base}/slack/status with auth headers", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ linked: true }));

    const result = await getSlackStatus({ fetchImpl });

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/slack/status`);
    expect(init?.method).toBe("GET");
    expect(result).toEqual({ linked: true });
  });
});

describe("getSlackAuthorizeUrl", () => {
  it("GETs {base}/slack/authorize-url and returns the URL", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(jsonResponse({ url: "https://slack.com/oauth/v2/authorize?x=1" }));

    const result = await getSlackAuthorizeUrl({ fetchImpl });

    expect(fetchImpl.mock.calls[0][0]).toBe(`${DEFAULT_API_BASE_URL}/slack/authorize-url`);
    expect(result).toEqual({ url: "https://slack.com/oauth/v2/authorize?x=1" });
  });

  it("throws ApiError with status 503 when Slack is not configured", async () => {
    const fetchImpl = vi
      .fn<typeof fetch>()
      .mockResolvedValue(
        jsonResponse({ detail: "Slack連携は現在利用できません。" }, { ok: false, status: 503 }),
      );

    await expect(getSlackAuthorizeUrl({ fetchImpl })).rejects.toMatchObject({
      status: 503,
    });
    await expect(getSlackAuthorizeUrl({ fetchImpl })).rejects.toBeInstanceOf(ApiError);
  });
});

describe("postSlackUnlink", () => {
  it("POSTs {base}/slack/unlink with auth headers and returns the ack", async () => {
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse({ ok: true }));

    const result = await postSlackUnlink({ fetchImpl });

    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/slack/unlink`);
    expect(init?.method).toBe("POST");
    expect(result).toEqual({ ok: true });
  });
});

describe("redeemDiscordLogin", () => {
  it("POSTs the one-time token to {base}/auth/discord/redeem and returns the login", async () => {
    const login = { access_token: "jwt", token_type: "bearer" };
    const fetchImpl = vi.fn<typeof fetch>().mockResolvedValue(jsonResponse(login));

    const result = await redeemDiscordLogin("once-1", { fetchImpl });

    const [url, init] = fetchImpl.mock.calls[0];
    expect(url).toBe(`${DEFAULT_API_BASE_URL}/auth/discord/redeem`);
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ token: "once-1" });
    expect(result).toEqual(login);
  });
});

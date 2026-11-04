import type { Page, Route } from "@playwright/test";

/**
 * Network mocking helpers + fixtures for the E2E suite.
 *
 * The frontend talks to the backend at `NEXT_PUBLIC_API_BASE_URL` (pinned to
 * this value by playwright.config.ts). We intercept those absolute URLs with
 * `page.route(...)` so the tests never need a live backend. Patterns are anchored
 * to the API host so they cannot accidentally match the Next.js page routes
 * (e.g. the `/dashboard` PAGE on :3100 vs the `/dashboard` API call on :8000).
 */
export const API_BASE = "http://localhost:8000";

interface SseFrame {
  event: string;
  data: unknown;
}

/**
 * Build a Server-Sent-Events body from named event frames.
 *
 * A leading `retry` of one hour keeps the browser's native EventSource from
 * reconnecting when a mocked stream ends WITHOUT a terminal event — the asker's
 * person-route view is reached by a non-terminal stream (understood → route →
 * recommend → draft, no `done`). Terminal streams (`done`/`message`/`error`)
 * make the client close the connection itself, so `retry` is harmless there.
 */
export function sseBody(frames: SseFrame[], options: { retryMs?: number } = {}): string {
  const retry = options.retryMs ?? 3_600_000;
  const parts = [`retry: ${retry}\n\n`];
  for (const frame of frames) {
    parts.push(`event: ${frame.event}\ndata: ${JSON.stringify(frame.data)}\n\n`);
  }
  return parts.join("");
}

export async function fulfillSse(route: Route, body: string): Promise<void> {
  await route.fulfill({
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache",
    },
    body,
  });
}

export async function fulfillJson(route: Route, body: unknown, status = 200): Promise<void> {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

// --------------------------------------------------------------------------- //
// Fixtures
// --------------------------------------------------------------------------- //

export const RECOMMENDATION = {
  person_id: "E001",
  name: "高梨 健太",
  dept: "技術部",
  score: 0.92,
  confidence: "高",
  reasons: [{ type: "answers", detail: "類似の質問に過去3件回答" }],
};

/** The draft the person-route stream feeds into the editor (asserted verbatim). */
export const PERSON_ROUTE_DRAFT = "高梨さんへ。UTM の移行時に注意すべき点をご相談させてください。";

/** Non-terminal person-route stream → reaches the PersonRouteView result. */
export const PERSON_ROUTE_FRAMES: SseFrame[] = [
  {
    event: "understood",
    data: {
      topics: ["ネットワーク"],
      products: ["UTM"],
      situation: "移行作業",
      question_type: "how",
      confidence: 0.9,
    },
  },
  {
    event: "route",
    data: { route: "person", reason: "同様の質問に直近で答えた方がいます。", confidence: 0.82 },
  },
  { event: "recommend", data: { recommendations: [RECOMMENDATION] } },
  { event: "draft", data: { draft: PERSON_ROUTE_DRAFT } },
];

/**
 * A FINISHED person hand-off, as /events replays it on a reconnect (#512): the
 * whole progress, then the stored terminal. This is what a session reached from
 * the history list actually receives — the shape #520's read-only candidate view
 * depends on.
 */
export const COMPLETED_PERSON_ROUTE_FRAMES: SseFrame[] = [
  ...PERSON_ROUTE_FRAMES,
  { event: "done", data: { status: "sent" } },
];

/** Non-terminal prior_answer stream (single candidate) → reaches the same
 * PersonRouteView result as the main line, with no intermediate screen (#310). */
export const PRIOR_ANSWER_FRAMES: SseFrame[] = [
  {
    event: "understood",
    data: {
      topics: ["ネットワーク"],
      products: ["UTM"],
      situation: "移行作業",
      question_type: "how",
      confidence: 0.88,
    },
  },
  {
    event: "route",
    data: {
      route: "prior_answer",
      reason: "同様の質問に過去に回答した方がいます。",
      confidence: 0.8,
    },
  },
  { event: "recommend", data: { recommendations: [RECOMMENDATION] } },
  { event: "draft", data: { draft: PERSON_ROUTE_DRAFT } },
];

/** Clarification stream → ProcessingScreen shows the FollowupForm (逆質問). */
export const FOLLOWUP_FRAMES: SseFrame[] = [
  {
    event: "understood",
    data: {
      topics: ["ネットワーク"],
      products: [],
      situation: null,
      question_type: "how",
      confidence: 0.5,
    },
  },
  {
    event: "followup",
    data: { question: "現在お使いの機器を教えてください。", missing: ["現行製品"] },
  },
];

/** Terminal no-result stream → ProcessingScreen "回答をお届けします" + message. */
export const MESSAGE_FRAMES: SseFrame[] = [
  {
    event: "understood",
    data: {
      topics: ["総務"],
      products: [],
      situation: null,
      question_type: "what",
      confidence: 0.6,
    },
  },
  {
    event: "route",
    data: { route: "document", reason: "該当者が見つかりませんでした。", confidence: 0.3 },
  },
  {
    event: "message",
    data: { status: "no_result", message: "該当する回答が見つかりませんでした。" },
  },
];

/** The admin principal for GET /auth/me (auth is required app-wide, #241). */
export const ADMIN_PRINCIPAL = { id: null, name: "管理者", dept: null, is_admin: true };

/**
 * Authenticate the E2E session (#241). Seeds a token in localStorage BEFORE the
 * app scripts run (so AuthProvider restores a session) and stubs GET /auth/me +
 * POST /auth/logout. Defaults to ADMIN so the demo switcher and dashboard — which
 * the existing specs expect — are available. Every spec's setup calls this
 * (right after mockEmployees); calling it more than once per page is harmless.
 */
export async function mockAuth(page: Page, principal: unknown = ADMIN_PRINCIPAL): Promise<void> {
  await page.addInitScript(() => {
    window.localStorage.setItem("tekijin.authToken", "e2e-token");
  });
  await page.route(`${API_BASE}/auth/me`, (route) => fulfillJson(route, principal));
  await page.route(`${API_BASE}/auth/logout`, (route) =>
    fulfillJson(route, { session_id: "", status: "logged_out" }),
  );
}

/** Directory for the current-user switcher (GET /employees). */
export const EMPLOYEES = [
  { id: "E001", name: "山田 太郎", dept: "営業部" },
  { id: "E002", name: "佐藤 花子", dept: "技術部" },
];

/**
 * Mock GET /employees. The header's current-user switcher fetches this on every
 * page, and the asker's submit is gated on having a current user — so any flow
 * that renders the app chrome needs it stubbed.
 */
export async function mockEmployees(page: Page, employees: unknown[] = EMPLOYEES): Promise<void> {
  await page.route(`${API_BASE}/employees`, (route) => fulfillJson(route, { employees }));
}

/** One pending handoff for the inbox (GET /inbox). */
export const INBOX_ITEM = {
  session_id: "11111111-1111-4111-8111-111111111111",
  question_id: "api_q1",
  question: "UTM の移行時に気をつけることは？",
  topics: ["ネットワーク"],
  asker: { id: "E010", name: "藤田 悠斗", dept: "第3営業部" },
  created_at: "2026-08-23T09:30:00",
};

/**
 * Mock GET /inbox (any responder_id query). The URL carries a query string, so
 * match by prefix with a predicate rather than a glob.
 */
export async function mockInbox(page: Page, items: unknown[] = [INBOX_ITEM]): Promise<void> {
  await page.route(
    (url) => url.href.startsWith(`${API_BASE}/inbox`),
    (route) => fulfillJson(route, { items }),
  );
}

/** Sample asker history for the "最近のあなたの質問" panel. */
export const RECENT_QUESTIONS = [
  {
    question_id: "api_rq1",
    title: "UTMの移行時の注意点",
    resolved: true,
    resolution: "person",
    responder_name: "高梨 健太",
    session_id: "sess-rq1",
    created_at: "2026-08-20T10:00:00",
  },
  {
    question_id: "api_rq2",
    title: "社内Wi-Fiの申請方法",
    resolved: false,
    resolution: "pending",
    responder_name: null,
    session_id: "sess-rq2",
    created_at: "2026-08-21T10:00:00",
  },
  {
    question_id: "api_rq3",
    title: "社内PCのセットアップ手順",
    resolved: true,
    resolution: "document",
    responder_name: null,
    session_id: null,
    created_at: "2026-08-22T10:00:00",
  },
];

/**
 * Mock GET /questions (any asker_id query) — the question screen's recent-history
 * panel. Defaults to empty so the asker flow is unaffected by history.
 */
export async function mockRecentQuestions(page: Page, items: unknown[] = []): Promise<void> {
  await page.route(
    (url) => url.href.startsWith(`${API_BASE}/questions`),
    (route) => fulfillJson(route, { items }),
  );
}

export const HANDOFF = {
  question: "UTM の移行時に気をつけることは？",
  asker: { id: 1, name: "山田 太郎", dept: "営業部" },
  topics: ["ネットワーク"],
  products: ["UTM"],
  situation: "移行作業中",
  missing: [],
  responder: RECOMMENDATION,
  draft: "山田さんから UTM 移行の注意点について質問が届いています。",
  reuse_count: 3,
  helpful_answer_count: 5,
};

export const DASHBOARD = {
  total_employees: 42,
  total_questions: 120,
  total_answers: 98,
  recommendation_count: 60,
  recommendation_outcomes: { accepted: 40, declined: 10, pending: 10 },
  acceptance_rate: 0.8,
  self_resolution_rate: 0.25,
  avg_resolution_hours: 4.5,
  top_responder_share: 0.3,
  processing_latency: { p50_ms: 900, p95_ms: 3100, sample_size: 30 },
  latest_eval: {
    top1_accuracy: 0.66,
    recall_at_3: 0.59,
    mrr: 0.7,
    route_accuracy: 0.7,
    created_at: "2026-08-22T00:00:00",
  },
  answers_per_responder: [
    { employee_id: 1, name: "高梨 健太", answer_count: 30 },
    { employee_id: 2, name: "鈴木 花子", answer_count: 20 },
  ],
  topic_distribution: [
    { topic: "ネットワーク", count: 50 },
    { topic: "セキュリティ", count: 30 },
  ],
  feedback_by_stage: { c1: 1, c6: 4, c7: 2, total: 7 },
  // #294: 蓄積メトリクス。Runtime-only counts, so a demo fixture states them
  // explicitly rather than letting the screen fall back to an empty section.
  knowledge_accumulation: {
    this_month: 3,
    last_month: 1,
    captured_answers: 2,
    consult_retrospectives: 1,
    accepted_handoffs: 4,
    capture_rate: 0.5,
    monthly: [
      { month: "2026-04", count: 0 },
      { month: "2026-05", count: 0 },
      { month: "2026-06", count: 1 },
      { month: "2026-07", count: 0 },
      { month: "2026-08", count: 1 },
      { month: "2026-09", count: 3 },
    ],
  },
};

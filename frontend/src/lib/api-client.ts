/**
 * Fetch-based client for the TEKIJIN API boundary.
 *
 * `postAsk` (POST /ask) starts a question; `postAnswer` (POST /answer) resumes a
 * paused run with a clarification reply or a responder outcome. The base URL
 * comes from config (`NEXT_PUBLIC_API_BASE_URL`) — never hardcoded at call
 * sites. Non-2xx responses throw a typed {@link ApiError}.
 */

import type {
  AckResponse,
  AskRequest,
  ChatMessage,
  ChatThreadDetail,
  ChatThreadListResponse,
  ChatThreadSummary,
  ConsultRetrospectiveAck,
  ConsultRetrospectiveContext,
  ConsultRetrospectiveRequest,
  DashboardResponse,
  DeleteQuestionResponse,
  DocumentDetail,
  DocumentFallbackRequest,
  EmployeeListResponse,
  EmployeeSummary,
  HandoffCorrectRequest,
  HandoffDraftRequest,
  HandoffExcludeRequest,
  HandoffRedraftRequest,
  HandoffResponse,
  HandoffSelectRequest,
  HandoffSelectResponse,
  InboxItem,
  InboxResponse,
  KnowledgeItem,
  KnowledgeListResponse,
  LoginRequest,
  LoginResponse,
  Notification,
  NotificationAckRequest,
  NotificationAckResponse,
  NotificationsResponse,
  Principal,
  QuestionStructureRequest,
  QuestionStructureResponse,
  RecentQuestionItem,
  RecentQuestionsResponse,
  ResolveQuestionResponse,
  ResumeRequest,
  SendMessageRequest,
  SlackAuthorizeUrlResponse,
  SlackStatusResponse,
  SlackUnlinkResponse,
  TopicVocabularyResponse,
} from "@/lib/api-types";
import { getAuthToken } from "@/lib/auth-token";
import { getApiBaseUrl } from "@/lib/config";

/** Authorization header for the current token, or an empty object when logged out. */
function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** Error thrown for a non-2xx API response, carrying the HTTP status. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export interface RequestOptions {
  /** Override the resolved base URL (mainly for tests / SSR wiring). */
  baseUrl?: string;
  /** Override the `fetch` implementation (mainly for tests). */
  fetchImpl?: typeof fetch;
  /** Optional abort signal. */
  signal?: AbortSignal;
}

async function extractErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown; error?: unknown };
    const detail = body.detail ?? body.error;
    if (typeof detail === "string" && detail.trim() !== "") {
      return detail;
    }
  } catch {
    // Non-JSON or empty body — fall through to the status-based message.
  }
  return `Request failed with status ${response.status}`;
}

/** POST `body` as JSON to `{base}{path}`, throwing {@link ApiError} on non-2xx. */
async function postJson<T>(path: string, body: unknown, options: RequestOptions = {}): Promise<T> {
  // Trim a trailing slash on an explicit override so `${base}${path}` never
  // doubles up; the config default is already normalized.
  const baseUrl = (options.baseUrl ?? getApiBaseUrl()).replace(/\/+$/, "");
  const doFetch = options.fetchImpl ?? fetch;

  const response = await doFetch(`${baseUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(body),
    signal: options.signal,
  });

  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }

  return (await response.json()) as T;
}

/** GET `{base}{path}` as JSON, throwing {@link ApiError} on non-2xx. */
async function getJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const baseUrl = (options.baseUrl ?? getApiBaseUrl()).replace(/\/+$/, "");
  const doFetch = options.fetchImpl ?? fetch;

  const response = await doFetch(`${baseUrl}${path}`, {
    method: "GET",
    headers: { ...authHeaders() },
    signal: options.signal,
  });

  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }

  return (await response.json()) as T;
}

/** DELETE `{base}{path}` as JSON, throwing {@link ApiError} on non-2xx. */
async function deleteJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const baseUrl = (options.baseUrl ?? getApiBaseUrl()).replace(/\/+$/, "");
  const doFetch = options.fetchImpl ?? fetch;

  const response = await doFetch(`${baseUrl}${path}`, {
    method: "DELETE",
    headers: { ...authHeaders() },
    signal: options.signal,
  });

  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }

  return (await response.json()) as T;
}

/**
 * POST /auth/login — exchange email+password for a bearer token + the principal.
 * Throws {@link ApiError} with 401 (bad credentials) or 429 (rate limited).
 */
export function postLogin(
  request: LoginRequest,
  options: RequestOptions = {},
): Promise<LoginResponse> {
  return postJson<LoginResponse>("/auth/login", request, options);
}

/**
 * GET /auth/me — the principal for the current token (session restore). Throws
 * {@link ApiError} with 401 when the token is missing/expired.
 */
export function getMe(options: RequestOptions = {}): Promise<Principal> {
  return getJson<Principal>("/auth/me", options);
}

/**
 * POST /auth/logout — stateless server ack; the caller drops the local token.
 * Best-effort: a transport error is swallowed (logout still clears the token).
 */
export async function postLogout(options: RequestOptions = {}): Promise<void> {
  try {
    await postJson<AckResponse>("/auth/logout", {}, options);
  } catch {
    // The token is cleared client-side regardless; nothing to surface.
  }
}

/**
 * POST /ask — start a question for a session. Returns the acknowledgement; the
 * actual stream flows over GET /events (subscribed via `useEventStream`).
 */
export function postAsk(request: AskRequest, options: RequestOptions = {}): Promise<AckResponse> {
  return postJson<AckResponse>("/ask", request, options);
}

/**
 * POST /answer — resume a paused run. Carries either a clarification `reply`
 * (answering a `followup` interrupt) or a responder `outcome`. The stream
 * resumes over the still-open GET /events connection.
 */
export function postAnswer(
  request: ResumeRequest,
  options: RequestOptions = {},
): Promise<AckResponse> {
  return postJson<AckResponse>("/answer", request, options);
}

/** Convert a completed document result into the existing person hand-off flow. */
export function requestDocumentFallback(
  request: DocumentFallbackRequest,
  options: RequestOptions = {},
): Promise<AckResponse> {
  return postJson<AckResponse>("/handoff/document-fallback", request, options);
}

/**
 * POST /handoff/draft — persist the asker's edited hand-off draft so the
 * responder sees the edited text (#174). Draft-only. Throws {@link ApiError}
 * with 404 (no hand-off pending / already answered) or 409 (awaiting a
 * clarification instead).
 */
export function updateHandoffDraft(
  request: HandoffDraftRequest,
  options: RequestOptions = {},
): Promise<AckResponse> {
  return postJson<AckResponse>("/handoff/draft", request, options);
}

/**
 * POST /handoff/select — the asker picks a different (of the currently shown)
 * candidate as the hand-off target; the draft is regenerated for them
 * (#200/#A1/#204). Throws {@link ApiError} with 404 (no hand-off pending /
 * already answered), 409 (awaiting a clarification instead), or 422
 * (`person_id` is not among the currently shown recommendations).
 */
export function selectHandoffCandidate(
  request: HandoffSelectRequest,
  options: RequestOptions = {},
): Promise<HandoffSelectResponse> {
  return postJson<HandoffSelectResponse>("/handoff/select", request, options);
}

/**
 * POST /handoff/exclude — the asker excludes the current send target
 * ("この人には聞かない"), rerouting to a freshly-scored next candidate (#260). The
 * new candidate + draft arrive over the open `/events` stream, so this only acks.
 * Throws {@link ApiError} with 404 (no hand-off pending / already answered), 409
 * (awaiting a clarification instead), or 422 (`person_id` is not the current
 * hand-off target).
 */
export function excludeHandoffCandidate(
  request: HandoffExcludeRequest,
  options: RequestOptions = {},
): Promise<AckResponse> {
  return postJson<AckResponse>("/handoff/exclude", request, options);
}

/**
 * POST /handoff/correct — the asker corrects the AI's interpretation ("解釈の訂正",
 * #260). The `supplement` is folded into the question and the whole pipeline
 * re-runs from C1; the re-run streams over `/events`, so this only acks. Throws
 * {@link ApiError} with 404 (no hand-off pending / already answered), 409
 * (awaiting a clarification instead), or 422 (blank supplement / nothing to
 * correct).
 */
export function correctInterpretation(
  request: HandoffCorrectRequest,
  options: RequestOptions = {},
): Promise<AckResponse> {
  return postJson<AckResponse>("/handoff/correct", request, options);
}

/**
 * POST /handoff/redraft — the asker asks the AI to regenerate the hand-off draft
 * for the current send target ("下書きの作り直し", #260), discarding any saved edit.
 * The new draft arrives over the open `/events` stream, so this only acks. Throws
 * {@link ApiError} with 404 (no hand-off pending / already answered) or 409
 * (awaiting a clarification instead).
 */
export function regenerateHandoffDraft(
  request: HandoffRedraftRequest,
  options: RequestOptions = {},
): Promise<AckResponse> {
  return postJson<AckResponse>("/handoff/redraft", request, options);
}

/**
 * POST /handoff/structure — the asker asks the AI to tidy their raw question into
 * the four hand-off fields (起きていること / 環境 / 試したこと / 詰まっている点, #475).
 * On-demand and read-only: it reshapes the already-stored question OUTSIDE the
 * graph (never on the C1 critical path) and returns the fields synchronously.
 * Throws {@link ApiError} with 404 when the session has no question yet.
 */
export function structureQuestion(
  request: QuestionStructureRequest,
  options: RequestOptions = {},
): Promise<QuestionStructureResponse> {
  return postJson<QuestionStructureResponse>("/handoff/structure", request, options);
}

/**
 * GET /handoff/{session_id} — the responder-facing view of a session paused at
 * the `send` interrupt (product-spec 画面4). Read-only: it does not advance the
 * graph. Throws {@link ApiError} with status 404 (no handoff pending) or 409
 * (the session is awaiting a clarification instead).
 */
export function getHandoff(
  sessionId: string,
  options: RequestOptions = {},
): Promise<HandoffResponse> {
  return getJson<HandoffResponse>(`/handoff/${encodeURIComponent(sessionId)}`, options);
}

/**
 * GET /dashboard — aggregate usage metrics for the admin dashboard (画面5).
 * Aggregate-only: no individual question content is returned.
 */
export function getDashboard(options: RequestOptions = {}): Promise<DashboardResponse> {
  return getJson<DashboardResponse>("/dashboard", options);
}

/**
 * GET /employees — the employee directory for the ADMIN's demo user switcher
 * (admin-only, #241). Returns the unwrapped array (ids in the external "E###"
 * form). Regular users never call this.
 */
export async function getEmployees(options: RequestOptions = {}): Promise<EmployeeSummary[]> {
  const body = await getJson<EmployeeListResponse>("/employees", options);
  return body.employees;
}

/**
 * GET /inbox — the questions currently awaiting `responderId` (the responder
 * inbox, #123). `responderId` is the external "E###" form. Returns the unwrapped
 * items array, newest first; each carries a `session_id` for `/answer/{id}`.
 */
export async function getInbox(
  responderId: string,
  options: RequestOptions = {},
): Promise<InboxItem[]> {
  const query = `?responder_id=${encodeURIComponent(responderId)}`;
  const body = await getJson<InboxResponse>(`/inbox${query}`, options);
  return body.items;
}

/**
 * GET /questions — the asker's own recent questions with resolution state
 * (画面1 の "最近のあなたの質問"). `askerId` is the external "E###" form.
 * `limit` caps how many newest-first questions come back (default 5 for the panel;
 * the history screen #208 passes a larger value). Returns the unwrapped items array.
 */
export async function getRecentQuestions(
  askerId: string,
  options: RequestOptions & { limit?: number } = {},
): Promise<RecentQuestionItem[]> {
  const { limit, ...requestOptions } = options;
  const limitQuery = limit === undefined ? "" : `&limit=${encodeURIComponent(limit)}`;
  const query = `?asker_id=${encodeURIComponent(askerId)}${limitQuery}`;
  const body = await getJson<RecentQuestionsResponse>(`/questions${query}`, requestOptions);
  return body.items;
}

/**
 * GET /knowledge — the company-wide list of resolved-by-a-person questions
 * (#293, #301), with optional search/filter and a side-panel summary. Unlike
 * {@link getRecentQuestions}, this is NOT scoped to one asker.
 *
 * The period filter is a start bound only (`since`). The endpoint's matching
 * `until` was removed in #394: nothing sent it, nothing tested it, and it cut
 * the end day off (see `knowledge_library.list_knowledge`).
 */
export async function getKnowledgeList(
  options: RequestOptions & {
    q?: string;
    department?: string;
    topic?: string;
    since?: string;
    offset?: number;
    limit?: number;
  } = {},
): Promise<KnowledgeListResponse> {
  const { q, department, topic, since, offset, limit, ...requestOptions } = options;
  const params = new URLSearchParams();
  if (q) params.set("q", q);
  if (department) params.set("department", department);
  if (topic) params.set("topic", topic);
  if (since) params.set("since", since);
  if (offset !== undefined) params.set("offset", String(offset));
  if (limit !== undefined) params.set("limit", String(limit));
  const query = params.size > 0 ? `?${params.toString()}` : "";
  return getJson<KnowledgeListResponse>(`/knowledge${query}`, requestOptions);
}

/**
 * GET /knowledge/{sourceId} — full detail of one past-Q&A knowledge item
 * (`kind="qa"`), keyed by the same `source_id` a self-answer's citation
 * carries (`Answer.id`). The `"document"` counterpart already has its own
 * viewer at `GET /documents/{doc_id}` (#143).
 */
export async function getKnowledgeDetail(
  sourceId: string,
  options: RequestOptions = {},
): Promise<KnowledgeItem> {
  return getJson<KnowledgeItem>(`/knowledge/${encodeURIComponent(sourceId)}`, options);
}

/**
 * DELETE /questions/{questionId} — remove one of the asker's own past questions
 * and its history (#207). Only the owning asker (or an admin) may delete; the API
 * answers 403 otherwise and 404 for a missing question. Returns the acknowledgement.
 */
export async function deleteQuestion(
  questionId: string,
  options: RequestOptions = {},
): Promise<DeleteQuestionResponse> {
  return deleteJson<DeleteQuestionResponse>(
    `/questions/${encodeURIComponent(questionId)}`,
    options,
  );
}

/**
 * POST /questions/{questionId}/resolve — mark one of the asker's own questions
 * self-resolved ("人を介さず解決した", #159). Only the owning asker (or an admin) may;
 * 403 otherwise, 404 for a missing question. Idempotent. Returns the ack.
 */
export async function resolveQuestion(
  questionId: string,
  options: RequestOptions = {},
): Promise<ResolveQuestionResponse> {
  return postJson<ResolveQuestionResponse>(
    `/questions/${encodeURIComponent(questionId)}/resolve`,
    {},
    options,
  );
}

/**
 * GET /topics — the closed topic vocabulary the C6 scorer joins on (#247).
 * Served rather than duplicated here: a hard-coded copy would drift from
 * `scorer/topics.py`, and a topic the scorer does not know matches no evidence.
 */
export async function getTopics(options: RequestOptions = {}): Promise<string[]> {
  const body = await getJson<TopicVocabularyResponse>("/topics", options);
  return body.topics;
}

/**
 * GET /consult-retrospective/{session_id} — the durable context for the write-up
 * form (#247): the question, how it was handed off, and who accepted it.
 *
 * Deliberately not {@link getHandoff}: that endpoint 404s once the responder
 * records an outcome, so a form built on it could only be reached BEFORE the
 * consultation it documents had happened.
 */
export function getRetrospectiveContext(
  sessionId: string,
  options: RequestOptions = {},
): Promise<ConsultRetrospectiveContext> {
  return getJson<ConsultRetrospectiveContext>(
    `/consult-retrospective/${encodeURIComponent(sessionId)}`,
    options,
  );
}

/**
 * POST /consult-retrospective — record the asker's write-up of a face-to-face
 * 直接相談 (#247). Only the question's own asker may (403 otherwise, 404 for an
 * unknown question, 422 for a topic outside the vocabulary, for a consultation
 * nobody accepted, or for a responder other than the one who accepted it, 409 if
 * this question has already been written up, 503 if the feature is switched off).
 */
export function postConsultRetrospective(
  request: ConsultRetrospectiveRequest,
  options: RequestOptions = {},
): Promise<ConsultRetrospectiveAck> {
  return postJson<ConsultRetrospectiveAck>("/consult-retrospective", request, options);
}

/**
 * GET /notifications — notifications the caller hasn't seen yet, newest
 * first (#E7 declined, #509 accepted/request_received). Exactly one of
 * `askerId` (declined + accepted, on the caller's own questions) or
 * `employeeId` (incoming requests still awaiting the caller's decision)
 * must be given — both are the external "E###" form. Returns the unwrapped
 * items array.
 */
export async function getNotifications(
  params: { askerId: string } | { employeeId: string },
  options: RequestOptions = {},
): Promise<Notification[]> {
  const query =
    "askerId" in params
      ? `?asker_id=${encodeURIComponent(params.askerId)}`
      : `?employee_id=${encodeURIComponent(params.employeeId)}`;
  const body = await getJson<NotificationsResponse>(`/notifications${query}`, options);
  return body.items;
}

/** POST /notifications/ack — mark notifications as seen (#E7 declined, #509 accepted/request_received). */
export function ackNotifications(
  request: NotificationAckRequest,
  options: RequestOptions = {},
): Promise<NotificationAckResponse> {
  return postJson<NotificationAckResponse>("/notifications/ack", request, options);
}

/**
 * GET /slack/status — whether the acting employee has a linked Slack account.
 */
export function getSlackStatus(options: RequestOptions = {}): Promise<SlackStatusResponse> {
  return getJson<SlackStatusResponse>("/slack/status", options);
}

/**
 * GET /slack/authorize-url — the "Sign in with Slack" URL to navigate the
 * browser to. Throws {@link ApiError} with status 503 while no Slack App is
 * registered yet (see `Settings.slack_configured` on the backend).
 */
export function getSlackAuthorizeUrl(
  options: RequestOptions = {},
): Promise<SlackAuthorizeUrlResponse> {
  return getJson<SlackAuthorizeUrlResponse>("/slack/authorize-url", options);
}

/**
 * POST /slack/login-url — the "Sign in with Slack" start URL for someone with NO
 * session yet (#406). Unauthenticated by design. Rejects with a 503 `ApiError`
 * when Slack login is switched off, which callers treat as "hide the button".
 */
export async function getSlackLoginUrl(
  options?: RequestOptions,
): Promise<SlackAuthorizeUrlResponse> {
  // POST so a third-party page cannot trigger it with `<img src>` (#494).
  return postJson<SlackAuthorizeUrlResponse>("/slack/login-url", undefined, options);
}

/**
 * POST /auth/discord/redeem — trade the one-time token from `/tekijin join` (#66)
 * for a bearer token. Unauthenticated by design; rejects with 401 when the token
 * is unknown, expired, or already used.
 */
export function redeemDiscordLogin(
  token: string,
  options?: RequestOptions,
): Promise<LoginResponse> {
  return postJson<LoginResponse>("/auth/discord/redeem", { token }, options);
}

/**
 * POST /slack/link/complete — redeem the pending token the OAuth callback left
 * in the URL fragment, attaching that Slack account to the CALLER (#494).
 *
 * The callback cannot do this itself: it has no session, so it does not know who
 * is linking. Rejects with 409 when the Slack account already belongs to someone.
 */
export async function completeSlackLink(
  pendingToken: string,
  options?: RequestOptions,
): Promise<SlackStatusResponse> {
  return postJson<SlackStatusResponse>(
    "/slack/link/complete",
    { pending_token: pendingToken },
    options,
  );
}

/** POST /slack/unlink — remove the acting employee's linked Slack account. */
export function postSlackUnlink(options: RequestOptions = {}): Promise<SlackUnlinkResponse> {
  return postJson<SlackUnlinkResponse>("/slack/unlink", undefined, options);
}

/**
 * GET /messages/threads — accepted chat threads where `employeeId` is a party
 * (asker or the accepted responder), newest activity first (#224).
 */
export async function getChatThreads(
  employeeId: string,
  options: RequestOptions = {},
): Promise<ChatThreadSummary[]> {
  const query = `?employee_id=${encodeURIComponent(employeeId)}`;
  const body = await getJson<ChatThreadListResponse>(`/messages/threads${query}`, options);
  return body.items;
}

/**
 * GET /messages/threads/{threadId} — one thread's full history, oldest first
 * (#224). Throws {@link ApiError} with status 404 if the thread isn't accepted
 * or `employeeId` isn't a party.
 */
export function getChatThread(
  threadId: number,
  employeeId: string,
  options: RequestOptions = {},
): Promise<ChatThreadDetail> {
  const query = `?employee_id=${encodeURIComponent(employeeId)}`;
  return getJson<ChatThreadDetail>(`/messages/threads/${threadId}${query}`, options);
}

/**
 * POST /messages — send one chat message on an accepted thread (#224). Throws
 * {@link ApiError} with status 404 (unaccepted thread / non-party sender) or
 * 422 (blank body).
 */
export function postMessage(
  request: SendMessageRequest,
  options: RequestOptions = {},
): Promise<ChatMessage> {
  return postJson<ChatMessage>("/messages", request, options);
}

/** Fetch one internal document's full content for the viewer (GET /documents/{id}, #143). */
export async function getDocument(
  docId: string,
  options: RequestOptions = {},
): Promise<DocumentDetail> {
  return getJson<DocumentDetail>(`/documents/${encodeURIComponent(docId)}`, options);
}

/**
 * Drive a paused run forward by consuming one GET /events pass to completion.
 *
 * POST /answer only *queues* the resume; the backend advances the graph when an
 * /events reader enters its dispatch loop. The responder's answer screen has no
 * long-lived stream of its own, so after submitting an outcome it calls this to
 * guarantee the queued resume is consumed — accept reaches C8 (done), decline
 * reroutes to the next candidate — instead of depending on the asker keeping a
 * tab open. The streamed content is irrelevant here; we only read it to
 * completion (the stream ends when the run next interrupts or terminates). This
 * is best-effort: a non-2xx (e.g. the run already finished → 404) or a transport
 * error is swallowed, since the outcome is already recorded server-side.
 */
export async function advanceSession(
  sessionId: string,
  options: RequestOptions = {},
): Promise<void> {
  const baseUrl = (options.baseUrl ?? getApiBaseUrl()).replace(/\/+$/, "");
  const doFetch = options.fetchImpl ?? fetch;
  try {
    const response = await doFetch(`${baseUrl}/events/${encodeURIComponent(sessionId)}`, {
      method: "GET",
      headers: { ...authHeaders() },
      signal: options.signal,
    });
    await response.text();
  } catch {
    // Best-effort: the outcome is already persisted; the asker's own reconnecting
    // stream is the backup consumer. Never surface this to the responder.
  }
}

/**
 * TypeScript mirror of the backend API contract.
 *
 * Source of truth: `backend/src/tekijin/api/schemas.py` (Pydantic v2 models) and
 * `backend/src/tekijin/api/events.py` (SSE event mapping). Kept in sync by hand;
 * the shapes below match those models field-for-field.
 *
 * Only `postAsk` (POST /ask) is wired in #35. The SSE data types are defined
 * ahead of time so the /events subscription hook (#36) can reuse them.
 */

// --------------------------------------------------------------------------- //
// requests / responses
// --------------------------------------------------------------------------- //

/**
 * Employee id as accepted by the API boundary. The DB stores it as an `int`, but
 * the external contract also accepts the spec's `"E###"` string form
 * (schemas.py `_coerce_asker_id`). The frontend normally sends the numeric form.
 */
export type EmployeeId = number | string;

/** POST /ask body — start (or restart) a question for a session. */
export interface AskRequest {
  asker_id: EmployeeId;
  question: string;
  session_id: string;
}

/** POST /handoff/document-fallback — continue a document result with its candidate. */
export interface DocumentFallbackRequest {
  session_id: string;
}

/**
 * POST /answer body — resume a paused run with exactly one of `outcome`
 * (responder accept/decline) or `reply` (clarification answer). Defined for
 * later screens; #35 does not call /answer.
 */
export type Outcome = "accepted" | "declined";

/**
 * The asker's chosen consultation method. "chat" is the implicit default — an
 * unset value from the backend is always coalesced to it.
 */
export type ConsultMethod = "direct" | "chat";

/**
 * Exactly one of `outcome` or `reply` — encoded as a discriminated union so a
 * caller (the #38 answer screen) cannot construct a payload with both or neither,
 * matching the backend's `_exactly_one` validator. The `never` on the unused arm
 * makes the wrong field a compile error rather than a runtime 422.
 */
export type ResumeRequest =
  | {
      session_id: string;
      outcome: Outcome;
      recommendation_id?: number | null;
      // The responder's answer text, captured on accept (#274). Only meaningful
      // with outcome "accepted" — the backend 422s an answer_body on a decline.
      answer_body?: string;
      reply?: never;
    }
  | { session_id: string; reply: string; outcome?: never; recommendation_id?: never };

/** Acknowledgement returned by /ask and /answer (the stream flows over /events). */
export interface AckResponse {
  session_id: string;
  status: string;
}

/** DELETE /questions/{id} — acknowledgement that a past question was removed (#207). */
export interface DeleteQuestionResponse {
  question_id: string;
  deleted: boolean;
}

/** POST /questions/{id}/resolve — acknowledgement of a self-resolution (#159). */
export interface ResolveQuestionResponse {
  question_id: string;
  resolved: boolean;
}

/**
 * POST /handoff/draft body — persist the asker's edited hand-off draft (画面3) so
 * the responder (画面4) reads the edited text. Draft-only; it never changes the
 * recommendation or the accept/decline outcome (#174).
 */
export interface HandoffDraftRequest {
  session_id: string;
  draft: string;
  /** Defaults to "chat" server-side when omitted. */
  consult_method?: ConsultMethod;
}

/**
 * POST /handoff/select body — the asker picks a different (of the currently
 * shown) candidate as the hand-off target; the draft is regenerated for them
 * (#200/#A1/#204).
 */
export interface HandoffSelectRequest {
  session_id: string;
  person_id: EmployeeId;
}

/**
 * POST /handoff/exclude body — the asker excludes the current send target
 * ("この人には聞かない"), rerouting to a freshly-scored next candidate (#260).
 * `person_id` must be the current hand-off target. The new candidate + draft
 * arrive over the open `/events` stream, so the response only acks.
 */
export interface HandoffExcludeRequest {
  session_id: string;
  person_id: EmployeeId;
}

/**
 * POST /handoff/correct body — the asker corrects the AI's interpretation
 * ("解釈の訂正", #260). The `supplement` is folded into the question and the whole
 * pipeline re-runs from C1; the re-run streams over `/events`, so the response
 * only acks.
 */
export interface HandoffCorrectRequest {
  session_id: string;
  supplement: string;
}

/**
 * POST /handoff/redraft body — the asker asks the AI to regenerate the hand-off
 * draft for the current send target ("下書きの作り直し", #260), discarding any saved
 * edit. The new draft arrives over the open `/events` stream, so the response
 * only acks.
 */
export interface HandoffRedraftRequest {
  session_id: string;
}

/**
 * POST /handoff/structure body — the asker asks the AI to tidy their raw question
 * into the four hand-off fields (#475 Screen 01). On-demand and read-only: it
 * reshapes the already-stored question, so only the session is named.
 */
export interface QuestionStructureRequest {
  session_id: string;
}

/**
 * POST /handoff/structure response (schemas.py `QuestionStructureResponse`) — the
 * re-drafted question: 起きていること / 環境 / 試したこと / 詰まっている点 (#475).
 * Any field may be empty — the model leaves a field blank rather than inventing
 * what the question never stated, and the asker fills or edits it before sending.
 */
export interface QuestionStructureResponse {
  session_id: string;
  summary: string;
  environment: string;
  tried: string;
  blocker: string;
}

/** POST /handoff/select response (schemas.py `HandoffSelectResponse`). */
export interface HandoffSelectResponse {
  session_id: string;
  responder: Recommendation;
  draft: string;
  recommendation_id: number;
}

/**
 * One employee for the current-user switcher (GET /employees). `id` is the
 * external "E###" form — the same shape accepted back as `asker_id` and used as
 * the responder id for the inbox, so a selection round-trips without conversion.
 */
export interface EmployeeSummary {
  id: string;
  name: string;
  dept?: string | null;
}

/** GET /employees payload (schemas.py `EmployeeListResponse`). */
export interface EmployeeListResponse {
  employees: EmployeeSummary[];
}

/**
 * One pending handoff awaiting the responder (GET /inbox, schemas.py `InboxItem`).
 * `session_id` deep-links to `/answer/{session_id}`.
 */
export interface InboxItem {
  session_id: string;
  question_id: string;
  question: string;
  topics: string[];
  asker: HandoffAsker;
  /** How the asker chose to be helped (#245); absent on the question = "chat". */
  consult_method: ConsultMethod;
  created_at?: string | null;
}

/** GET /inbox payload (schemas.py `InboxResponse`). */
export interface InboxResponse {
  items: InboxItem[];
}

/**
 * One of the asker's own recent questions (GET /questions, schemas.py
 * `RecentQuestionItem`). `responder_name` is the accepting/answering person, or
 * null while unresolved.
 */
/**
 * How a question was resolved (or that it is still pending): "person" a responder
 * took it, "document" the document route self-resolved it (no human), "pending"
 * still awaiting a hand-off.
 */
export type QuestionResolution = "person" | "self" | "document" | "pending";

export interface RecentQuestionItem {
  question_id: string;
  title: string;
  resolved: boolean;
  resolution: QuestionResolution;
  responder_name?: string | null;
  /** Deep-link target for re-viewing the result (/session/{session_id}); null for seeded history. */
  session_id?: string | null;
  created_at?: string | null;
}

/** GET /questions payload (schemas.py `RecentQuestionsResponse`). */
export interface RecentQuestionsResponse {
  items: RecentQuestionItem[];
}

/**
 * One accumulated piece of knowledge: an answered question OR an internal
 * document (GET /knowledge, schemas.py `KnowledgeItem`) — the same two kinds
 * a self-answer's citation carries. `source_id` matches a chat citation's
 * `SourceCitation.source_id` for the same `kind` (`Answer.id` for `"qa"`,
 * `Document.id` for `"document"`). Unlike `RecentQuestionItem`, this is NOT
 * scoped to the acting user. The `"document"`-only fields (`question_id`,
 * `session_id`, responder, topics) are absent/empty for that kind.
 */
export interface KnowledgeItem {
  source_id: string;
  kind: "qa" | "document" | "knowledge";
  title: string;
  summary: string;
  topics: string[];
  responder_name?: string | null;
  responder_department?: string | null;
  /** The item's own timestamp: the ANSWER's for "qa", `updated_at` for "document". */
  resolved_at?: string | null;
  question_id?: string | null;
  /** Deep-link target for viewing the result (/session/{session_id}); null for seeded history. */
  session_id?: string | null;
}

/**
 * Side-panel aggregate stats on GET /knowledge (schemas.py `KnowledgeSummary`).
 * Per-responder aggregates are deliberately absent — that view belongs to
 * `/dashboard`, not a knowledge browser (PR #340 review).
 */
export interface KnowledgeSummary {
  total_items: number;
  self_resolution_rate: number;
}

/**
 * GET /knowledge payload (schemas.py `KnowledgeListResponse`). `total_matching`
 * is the count of items matching the current filters BEFORE the offset/limit
 * page cut — what a search paginates its results with.
 */
export interface KnowledgeListResponse {
  items: KnowledgeItem[];
  total_matching: number;
  summary: KnowledgeSummary;
}

/**
 * One not-yet-seen notification (GET /notifications, schemas.py
 * `Notification`). `kind`-specific fields are only populated for their own
 * kind:
 * - "declined" (#E7): paired with the automatic reroute (#206) — by the
 *   time this is shown, the system has already moved on to the next
 *   candidate, so it is informational, not a request to act.
 *   `declined_person_name` is set.
 * - "accepted" (#509, asker-side): a responder took the request.
 *   `responder_name` + `consult_method` are set — the latter decides
 *   whether the bell links to the chat thread or the session directly.
 * - "request_received" (#509, responder-side): a new request is waiting on
 *   this employee's decision. `asker_name` is set.
 */
export interface Notification {
  kind: "declined" | "accepted" | "request_received";
  /** the Recommendation row's id — also the ack target. */
  id: number;
  question_id: string;
  /** deep-link target; null for pre-session-tracking rows. */
  session_id?: string | null;
  message: string;
  created_at?: string | null;
  declined_person_name?: string | null;
  responder_name?: string | null;
  consult_method?: ConsultMethod | null;
  asker_name?: string | null;
}

/** GET /notifications payload (schemas.py `NotificationsResponse`). */
export interface NotificationsResponse {
  items: Notification[];
}

/**
 * POST /notifications/ack body — mark notifications as seen (#E7 declined,
 * #509 accepted/request_received). `asker_id` is required for
 * "declined"/"accepted"; `employee_id` is required for "request_received".
 */
export type NotificationAckRequest =
  | {
      kind: "declined" | "accepted";
      asker_id: EmployeeId;
      employee_id?: never;
      ids: number[];
    }
  | {
      kind: "request_received";
      employee_id: EmployeeId;
      asker_id?: never;
      ids: number[];
    };

/** POST /notifications/ack response (schemas.py `NotificationAckResponse`). */
export interface NotificationAckResponse {
  acknowledged: number;
}

/** GET /slack/authorize-url response (schemas.py `SlackAuthorizeUrlResponse`). */
export interface SlackAuthorizeUrlResponse {
  url: string;
}

/** GET /slack/status response (schemas.py `SlackStatusResponse`). */
export interface SlackStatusResponse {
  linked: boolean;
}

/** POST /slack/unlink response (schemas.py `SlackUnlinkResponse`). */
export interface SlackUnlinkResponse {
  ok: boolean;
}

// --------------------------------------------------------------------------- //
// domain models (shared by SSE data and final response)
// --------------------------------------------------------------------------- //

export interface Reason {
  type: string;
  detail: string;
}

export interface Recommendation {
  /** external "E###" form (schemas.format_employee_id). */
  person_id: string;
  name: string;
  dept?: string | null;
  score: number;
  confidence: string;
  reasons: Reason[];
}

// --------------------------------------------------------------------------- //
// handoff (GET /handoff/{session_id}) — responder-facing view (product-spec 画面4)
// --------------------------------------------------------------------------- //

/** The asking employee, enriched for the responder-facing handoff view. */
export interface HandoffAsker {
  /** external "E###" form (schemas.format_employee_id). */
  id: string;
  name?: string | null;
  dept?: string | null;
}

/**
 * Responder-facing payload for a session paused at the `send` interrupt
 * (schemas.py `HandoffResponse`). Read-only: fetching it does not advance the
 * graph — the responder acts via POST /answer (outcome=accepted|declined).
 */
export interface HandoffResponse {
  session_id: string;
  question: string;
  /** The durable question id — what the #247 retrospective is attributed to. */
  question_id?: string | null;
  asker: HandoffAsker;
  topics: string[];
  products: string[];
  situation?: string | null;
  missing: string[];
  /** the primary (handed-off) candidate — the person being asked. */
  responder?: Recommendation | null;
  draft: string;
  reuse_count: number;
  helpful_answer_count: number;
  /** Generation token echoed back on POST /answer so a stale outcome 409s (#94). */
  recommendation_id?: number | null;
  /** The asker's chosen consultation method; "chat" until they choose otherwise. */
  consult_method?: ConsultMethod;
}

// --------------------------------------------------------------------------- //
// chat (GET/POST /messages) — accepted-recommendation threads (#224)
// --------------------------------------------------------------------------- //

/** POST /messages body — send one chat message on an accepted thread. */
export interface SendMessageRequest {
  thread_id: number;
  sender_id: EmployeeId;
  body: string;
}

/** One chat message (schemas.py `MessageItem`). `sender_id` is the external "E###" form. */
export interface ChatMessage {
  id: number;
  thread_id: number;
  sender_id: string;
  body: string;
  created_at: string;
}

/**
 * One accepted thread for the chat list (schemas.py `MessageThreadSummary`),
 * newest activity first. `thread_id` is the accepted `Recommendation.id`.
 */
export interface ChatThreadSummary {
  thread_id: number;
  question_id: string;
  question_title: string;
  counterpart: HandoffAsker;
  last_message?: string | null;
  last_message_at?: string | null;
  created_at: string;
}

/** GET /messages/threads payload (schemas.py `MessageThreadListResponse`). */
export interface ChatThreadListResponse {
  items: ChatThreadSummary[];
}

/** GET /messages/threads/{id} payload (schemas.py `MessageThreadDetail`), oldest-first history. */
export interface ChatThreadDetail {
  thread_id: number;
  question_id: string;
  question_title: string;
  counterpart: HandoffAsker;
  messages: ChatMessage[];
  /** Deep link to this pair's shared Slack channel, or null if none exists yet. */
  slack_channel_url: string | null;
}

// --------------------------------------------------------------------------- //
// dashboard (GET /dashboard) — aggregate-only view (product-spec 画面5)
// --------------------------------------------------------------------------- //

export interface ResponderLoad {
  employee_id: number;
  name: string;
  answer_count: number;
}

export interface TopicCount {
  topic: string;
  count: number;
}

export interface OutcomeCounts {
  accepted: number;
  declined: number;
  pending: number;
}

/** Latest stored offline-evaluation metrics (推薦精度); null until `make eval` ran. */
export interface EvalSnapshot {
  top1_accuracy: number | null;
  recall_at_3: number | null;
  mrr: number | null;
  route_accuracy: number | null;
  created_at: string | null;
}

/** p50/p95 of per-question AI processing time in ms (schemas.py `ProcessingLatency`, #177). */
export interface ProcessingLatency {
  p50_ms: number | null;
  p95_ms: number | null;
  sample_size: number;
}

/**
 * GET /dashboard payload (schemas.py `DashboardResponse`). Aggregate-only — no
 * per-record listing (product-spec §241-251: summarise usage, never a monitoring
 * log of individual questions).
 */
/** #237: feedback counts per pipeline stage (どの段でどれだけずれているか). */
export interface FeedbackByStage {
  c1: number;
  c6: number;
  c7: number;
  total: number;
}

/** One point of the accumulation trend (#294): `"2026-09"` and its count. */
export interface MonthlyCount {
  month: string;
  count: number;
}

/**
 * GET /dashboard `knowledge_accumulation` (#294) — how much tacit knowledge became
 * explicit, and whether the loop is closing.
 *
 * Counts only what the runtime produced (captured answers #274, 直接相談
 * retrospectives #247), never the seeded corpus. `capture_rate` is the recovery
 * rate: of the hand-offs accepted this month, the share that left knowledge behind.
 */
export interface KnowledgeAccumulation {
  this_month: number;
  last_month: number;
  captured_answers: number;
  consult_retrospectives: number;
  accepted_handoffs: number;
  capture_rate: number;
  monthly: MonthlyCount[];
}

export interface DashboardResponse {
  total_employees: number;
  total_questions: number;
  total_answers: number;
  recommendation_count: number;
  recommendation_outcomes: OutcomeCounts;
  acceptance_rate: number;
  /** 自己解決率: fraction routed to an auxiliary route (prior_answer/document). */
  self_resolution_rate: number;
  /** 平均解決時間 in hours; null when nothing is resolved yet. */
  avg_resolution_hours: number | null;
  /** 負荷分散: share of answers by the single busiest responder. */
  top_responder_share: number;
  /** 応答速度: p50/p95 of AI processing time (ms) from recorded run events (#177). */
  processing_latency: ProcessingLatency;
  /** 推薦精度: latest offline eval snapshot; null when not yet measured. */
  latest_eval: EvalSnapshot | null;
  answers_per_responder: ResponderLoad[];
  topic_distribution: TopicCount[];
  /** #237: how often the asking side corrected each stage (c1/c6/c7). */
  feedback_by_stage: FeedbackByStage;
  knowledge_accumulation: KnowledgeAccumulation;
}

// --------------------------------------------------------------------------- //
// SSE event data (GET /events/{session_id}) — mirrors events.py
// --------------------------------------------------------------------------- //

export interface UnderstoodData {
  topics: string[];
  products: string[];
  situation?: string | null;
  question_type?: string | null;
  confidence: number;
  /**
   * #475 Screen 01: how many OTHER people previously asked in the same topic area
   * (reassurance — "you are not the only one"). 0 when the feature is off or nobody
   * else has; the UI hides the message at 0. Optional so older payloads still parse.
   */
  similar_asker_count?: number;
}

export interface FollowupData {
  question: string;
  missing: string[];
}

export interface RouteData {
  route: string;
  reason: string;
  confidence: number;
}

export interface RecommendData {
  recommendations: Recommendation[];
}

export interface DraftData {
  draft: string;
}

export interface DoneData {
  status: string;
  answer?: string | null;
  /** This run segment's processing time in ms (#177); null on a replay. */
  latency_ms?: number | null;
}

/**
 * One source a self-answer (#291) cited, shown in chat as a link or a label chip
 * (schemas.py `SourceCitation`).
 */
export interface SourceCitation {
  source_id: string;
  /**
   * Closed set from the backend contract (fragments.py): a past Q&A, an internal
   * doc, a daily report (#433), or a structured knowledge unit (#357). "daily" and
   * "knowledge" have no detail page yet, so the UI shows them as label chips
   * rather than links.
   *
   * Adding "knowledge" here is the enablement gate for #357 (#366): the renderer's
   * final branch is a catch-all that reads 「過去の回答」, so a `ku_*` citation would
   * be mislabelled as a past answer the moment `knowledge_retrieval_enabled` went
   * true.
   */
  kind: "qa" | "document" | "daily" | "knowledge";
}

/**
 * #413: a cited answer surfaced ALONGSIDE a person hand-off ("参考: 過去の類似
 * 回答"). Emitted as a `reference` event on the person route before `recommend`
 * when a grounded past answer exists — additive, never a substitute for the
 * hand-off (schemas.py `ReferenceData`).
 */
export interface ReferenceData {
  answer: string;
  citations: SourceCitation[];
}

export interface MessageData {
  status: string;
  message: string;
  /** For the "document" route: the cited document's id (GET /documents/{doc_id}). */
  doc_id?: string | null;
  /** Person already ranked as the document route's optional fallback (#351). */
  fallback_responder?: Recommendation | null;
  /**
   * #291/#293: sources the self-answer grounded on, rendered as links in chat.
   * Empty on non-self-answer terminals. Optional so older payloads parse.
   */
  citations?: SourceCitation[];
  /** This run segment's processing time in ms (#177); null on a replay. */
  latency_ms?: number | null;
}

/** GET /documents/{doc_id} payload (schemas.py `DocumentDetail`). */
export interface DocumentDetail {
  id: string;
  title?: string | null;
  body?: string | null;
  source?: string | null;
  updated_at?: string | null;
}

export interface ErrorData {
  error: string;
}

/**
 * Named SSE events emitted over /events, mapped to their `data` payload type.
 * (understood/followup/route/recommend/draft/done/message/error — see events.py.)
 */
export interface SseEventDataMap {
  understood: UnderstoodData;
  followup: FollowupData;
  route: RouteData;
  reference: ReferenceData;
  recommend: RecommendData;
  draft: DraftData;
  done: DoneData;
  message: MessageData;
  error: ErrorData;
}

export type SseEventName = keyof SseEventDataMap;

/**
 * The authenticated principal (GET /auth/me, POST /auth/login). `id` is the
 * external "E###" form for a regular user, or `null` for the admin (which is not
 * a DB employee). `is_admin` gates the dashboard and the demo user switcher.
 */
export interface Principal {
  id: string | null;
  name: string;
  dept?: string | null;
  is_admin: boolean;
}

/** POST /auth/login request body. */
export interface LoginRequest {
  email: string;
  password: string;
}

/** POST /auth/login response (schemas.py `LoginResponse`). */
export interface LoginResponse {
  access_token: string;
  token_type: "bearer";
  principal: Principal;
}

/** GET /topics — the closed topic vocabulary the C6 scorer joins on (#247). */
export interface TopicVocabularyResponse {
  topics: string[];
}

/** How far a 直接相談 got (#247). `unresolved` is recorded but is not evidence. */
export type ConsultResolution = "resolved" | "partial" | "unresolved";

/** The person a retrospective may be written about (#247). */
export interface ConsultResponder {
  person_id: string;
  name: string;
}

/**
 * GET /consult-retrospective/{session_id} — what the write-up form is built from
 * (#247).
 *
 * NOT `HandoffResponse`: that is the pending hand-off view and 404s the moment the
 * responder records an outcome, which is exactly when the face-to-face
 * consultation can finally have taken place. This one is read from the database
 * and stays valid afterwards.
 *
 * `responder` is null until someone accepts; `already_recorded` flips once a
 * write-up exists.
 */
export interface ConsultRetrospectiveContext {
  session_id: string;
  question_id: string;
  question: string;
  consult_method: ConsultMethod;
  responder: ConsultResponder | null;
  already_recorded: boolean;
}

/**
 * POST /consult-retrospective — the asker's write-up of a face-to-face 直接相談
 * (#247). `asker_id` is deliberately absent: the backend takes it from the token,
 * never the body, because this row becomes expertise evidence for `responder_id`.
 */
export interface ConsultRetrospectiveRequest {
  question_id: string;
  responder_id: string;
  topics: string[];
  asked?: string | null;
  answer_body: string;
  resolution: ConsultResolution;
}

/** POST /consult-retrospective response. */
export interface ConsultRetrospectiveAck {
  status: string;
  consult_id: number;
}

/** GET /sources — one of the organization's ingested documents (#72). */
export interface SourceItem {
  id: number;
  /** Where it came from: "manual" (added by hand), later "discord", "google", "notion". */
  provider: string;
  title: string | null;
  url: string | null;
  author_name: string | null;
  updated_at: string;
  /** How many searchable chunks the body was split into. */
  chunks: number;
  /** True for the member who added it, and for the admin. */
  can_delete: boolean;
}

export interface SourceListResponse {
  items: SourceItem[];
  /** All of the organization's sources, for paging past one response. */
  total: number;
}

/** POST /sources request (#72). */
export interface SourceCreateRequest {
  body: string;
  title: string | null;
  url: string | null;
}

/** POST /sources response: re-sending the same URL updates instead of adding. */
export interface SourceCreateResponse {
  status: "added" | "updated" | "unchanged";
  id: number;
  chunks: number;
}

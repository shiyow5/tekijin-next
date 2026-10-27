"""Application configuration.

Settings are loaded from environment variables (prefix ``TEKIJIN_``) with
sensible defaults for local development. Use :func:`get_settings` to obtain a
cached singleton.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, resolved relative to this file:
#   <repo>/backend/src/tekijin/config.py -> parents[3] == <repo>
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_FIXTURES_DIR = _REPO_ROOT / "fixtures" / "synthetic"

# INSECURE development defaults for auth (#241). Named so the fail-closed startup
# guard (``Settings.auth_enforced``) can detect "left at the default" without
# duplicating the literal — mirroring the ``strict_durability`` pattern.
DEV_AUTH_SECRET = "dev-insecure-change-me"
DEV_ADMIN_PASSWORD = "tekijin-admin"


class Settings(BaseSettings):
    """Runtime settings for the TEKIJIN backend.

    Field values are read from ``TEKIJIN_*`` environment variables when present,
    otherwise the defaults below are used.
    """

    model_config = SettingsConfigDict(
        env_prefix="TEKIJIN_",
        # Absolute path to the repository-root .env (the one .env.example targets),
        # so it is read consistently regardless of the process working directory.
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,  # immutable singleton: mutation must not leak across requests
    )

    app_env: str = "development"

    # Database (connection code lands in a later component).
    database_url: str = "postgresql+psycopg://tekijin:tekijin@localhost:5432/tekijin"

    # LLM serving via any OpenAI-compatible /v1 endpoint. The default is OpenRouter
    # (the DGX vLLM went away with the internship); a local llama-server or vLLM
    # works too — set ``llm_thinking_control="chat_template"`` for those.
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "qwen/qwen3.6-35b-a3b"
    llm_api_key: str = "dummy"

    # How ``llm_enable_thinking`` reaches the server. "chat_template" sends
    # ``chat_template_kwargs={"enable_thinking": ...}`` (vLLM / llama-server read it
    # through the Qwen3 template); "reasoning" sends OpenRouter's ``reasoning`` field
    # (only ``effort="none"`` turns thinking off there); "none" sends nothing, for an
    # endpoint that rejects unknown fields.
    llm_thinking_control: Literal["chat_template", "reasoning", "none"] = "reasoning"

    # OpenRouter's ``provider`` object, sent verbatim in the request body. Pin the
    # provider here: the same model is served at different quantizations and with
    # different tool/structured-output support, and some providers retain prompts.
    # Read from the environment as JSON, e.g.
    # ``{"order": ["akashml", "deepinfra"], "allow_fallbacks": false,
    #   "require_parameters": true, "data_collection": "deny"}``.
    llm_provider_routing: dict[str, Any] | None = None

    # ``with_structured_output`` method for every structured call. "function_calling"
    # is what was measured on the DGX; "json_schema" asks the server for constrained
    # decoding instead, which some providers honour more reliably than forced tools.
    llm_structured_output_method: Literal["function_calling", "json_schema"] = "function_calling"

    # Whether to let a reasoning model (Qwen3) emit its <think> chain before the
    # answer. Kept OFF by default: with thinking ON the C1/C2 structured-output
    # (tool-call) calls became both slow (C1+C2 p50 ≈ 83s) AND unreliable — the
    # reasoning pass sometimes suppressed the forced tool call, so
    # ``with_structured_output`` returned ``None`` and routing crashed (#140).
    # Passed to vLLM as ``chat_template_kwargs={"enable_thinking": ...}``; the
    # Qwen3 chat template reads it. This is a SINGLE GLOBAL switch shared by all
    # LLM nodes (C1 intent, C2 sufficiency, AND C7 draft) — flipping it on to
    # experiment with draft (C7) quality also re-enables thinking on C1/C2 and can
    # reintroduce the structured-output breakage, so change it deliberately.
    llm_enable_thinking: bool = False

    # Per-request timeout (seconds) for every LLM call (C1/C2/C7). A stuck vLLM
    # request otherwise hangs the run and holds a backpressure slot forever, so a
    # single GPU stall cascades into unavailability (#180 task 4). On timeout the
    # call raises and the run surfaces a graceful error event, freeing the slot.
    # ``None`` disables the bound (langchain's own default). The real Claude
    # fallback on timeout is a separate, credential-gated follow-up.
    llm_timeout_seconds: float | None = 60.0

    # Sampling temperature for the STRUCTURED calls (C1 intent / C2 sufficiency).
    # Default 0.0 so they are deterministic, as model-definition.md specifies
    # ("C1・C2 は低温"). ChatOpenAI otherwise sends its own default (0.7), which
    # made C1 non-deterministic and contributed to routing noise (#116 原因3).
    llm_temperature: float = 0.0

    # Separate temperature for the C7 hand-off DRAFT (free text). model-definition.md
    # wants "C7 は中温（自然さ）" — a deterministic draft reads stilted — so C7 does
    # NOT share the C1/C2 low temperature above. Kept a distinct knob so tuning the
    # draft's naturalness never re-introduces non-determinism into routing.
    llm_draft_temperature: float = 0.5

    # Hard cap on output tokens per LLM call (C1/C2/C7). Bounds a runaway
    # generation so C1 returns a short (possibly truncated) result instead of
    # burning the whole context and failing with ``finish_reason=length`` (#116
    # 原因2). 1024 comfortably covers the structured C1/C2 tool calls and a C7
    # hand-off draft. ``None`` sends no limit (the server default applies).
    llm_max_tokens: int | None = 1024

    # Client-side retries per LLM call. Default 0 so ``llm_timeout_seconds`` is a
    # HARD per-request bound: ChatOpenAI defaults to 2 retries and retries on a
    # timeout, which would make the real worst-case stall ``timeout × 3`` (~185s)
    # and hold a backpressure slot ~3× longer than intended (#180 task 4 review).
    llm_max_retries: int = 0

    # Which C1/C2/C7 implementation the API wires: "stub" = deterministic,
    # network-free defaults (CI/tests); "vllm" = real langchain-openai client
    # against ``llm_base_url``. A ``Literal`` so an invalid value is rejected at
    # startup (never a silent fallback). Default stub keeps imports/CI LLM-free.
    llm_backend: Literal["stub", "vllm"] = "stub"

    # Evidence-sufficiency critic (#70): reject to a graceful "no in-house expert"
    # terminal when the critic's 0–100 answerable score is BELOW this threshold.
    # Externalised (not hard-coded) because the measured accept band is wide and
    # flat (30–70) on a tiny reject set (#65/#67 §6). ``answerability_enabled``
    # gates the whole step OFF by default until it is wired + verified on the eval
    # (this is the component-only slice; the graph does not call it yet).
    answerability_enabled: bool = False
    answerability_threshold: int = 40

    # Self-answer (#291): when the retrieved past Q&A / documents already hold the
    # answer, reply DIRECTLY with a cited answer instead of handing off to a person
    # (the product pivot — "the answer is not always a person"). ENABLED after the
    # full-graph E2E verification (#380, DGX, real Qwen3.6 + Nemotron): firing ONLY
    # on the data-derived routes AFTER C5, it leaves person routing untouched
    # (person recall 1.000 -> 1.000) while citing a grounded answer on the data
    # rows (source recall 0.239, precision 0.739, grounded 0.261 — conservative,
    # low-hallucination). Safe by construction: it never intercepts a person query.
    self_answer_enabled: bool = True

    # #413: additive self-answer on the PERSON route. self_answer (#291) fires ONLY
    # on the data-derived routes (document/prior_answer) after C5 — so a knowledge
    # question that routes to a person (an expert exists) never shows a past-answer
    # citation, even when one grounds it. This makes System 1 also fire on the
    # person route: before the hand-off, if a low-relevance FLOOR is cleared, try a
    # grounded cited answer and surface it ALONGSIDE the recommendation. It NEVER
    # replaces the hand-off (person recall unchanged) and never marks the run
    # self-resolved. Requires self_answer to be wired (shares the composer).
    # DGX sizing (research_selfanswer_person.py): person-route compose grounded rate
    # 0.237 (avg 3.78 citations) — ~24% of person questions would gain a citation
    # that today shows none. OFF by default until the full-graph E2E confirms person
    # recall stays 1.000 and citations fire; the floor gates the compose LLM call so
    # no-data person questions add no latency.
    #
    # Before enabling (security review #413):
    # * The floor (0.20) is BELOW the data-route thresholds (DOCUMENT_SIM 0.28), so
    #   it composes on weak-relevance evidence a data route would not pick — re-check
    #   the false-positive rate on the DGX eval when tuning it.
    # * The corpus has no per-row ACL, so a past answer is treated as readable by any
    #   asker. Enabling surfaces past-answer summaries on the person route too, so
    #   confirm with the product owner that past answers are org-wide readable.
    # The compose call is best-effort: a failure degrades to a plain hand-off (the
    # additive_answer node swallows it), so person recall never regresses.
    #
    # Enabled by default (#413 → task3): the DGX full-graph floor sweep confirmed
    # floor 0.20 is the sweet spot — person route recall stays 1.000 at every floor
    # (additive is routing-safe by construction), and 0.20 fires on 17/76 person rows
    # at precision 0.740, while 0.15 collapses precision to 0.583 for one extra firing
    # and 0.30 fires on only 6/76. The product owner confirmed past answers are
    # org-wide readable (already surfaced by the knowledge library, #293).
    additive_self_answer_enabled: bool = True
    additive_self_answer_floor: float = 0.20

    # #475 Screen 01: attach the "N other people asked in this area" reassurance
    # count to the understood event. Read-only display signal (topic-overlap count
    # over the questions table, GIN-indexed) with no routing/scoring effect, so it
    # is enabled by default. The frontend hides it when the count is 0.
    similar_askers_enabled: bool = True

    # #327: corpus-count routing for prior_answer. Nemotron's answer cosine cannot
    # separate this route (PRIOR_ANSWER_SIM sits above the observed max — see
    # route.py / #119), so route on whether the top retrieved past answer is a
    # REUSED/canonical answer instead. ``None`` (default) = OFF: prior_answer stays
    # dormant and C5 behaves exactly as before. When set (e.g. 3), a top past
    # answer with ``reuse_count >=`` this value AND answer cosine >=
    # ``prior_answer_relevance_floor`` (a low noise floor, not a discriminator)
    # routes prior_answer. Calibrate the two on the DGX eval before enabling.
    prior_answer_reuse_min: int | None = None
    prior_answer_relevance_floor: float = 0.15

    # #355: include daily reports as C6 topic evidence. The eval gold derives from
    # projects + daily_reports(0.15), but the scorer was blind to daily reports —
    # so enriching daily activity (#326) could not lift R@3. This closes that
    # asymmetry. OFF by default (develop behaviour byte-identical); enable only
    # after DGX confirms a Pareto gain (primary R@3 up, alt not down).
    daily_evidence_enabled: bool = False

    # #433: search daily reports as a KNOWLEDGE SOURCE for System 1 (distinct from
    # #355 daily_evidence, which uses topic-overlap for the C6 scorer). When on, C4
    # runs a question↔daily dense search and self-answer / #413 additive can cite a
    # report's tacit knowledge (issue + content) alongside answers/documents. OFF by
    # default (no daily channel, RetrievalResult byte-identical). DGX sizing
    # (research_knowledge_source.py): person-route self-answer grounded rate
    # 0.265->0.347 (+0.082, daily cited on 6%) — chat was noise and even hurt, so
    # ONLY daily is wired. Requires the daily embedding column filled (make embed).
    #
    # Enabled by default (#433 → task3): the DGX full-graph run (--additive
    # --daily-knowledge, floor 0.20) held person route recall at 1.000 and raised the
    # self-answer grounded rate 0.217->0.261 (+0.044) with source recall +0.010. Daily
    # is routing-safe (C5 reads retrieval *_confidence, not the daily channel). The
    # daily embedding column starts NULL after migrate; deploy.sh's embed_missing step
    # (best-effort, only-missing) fills it on the next deploy, so the flag is inert
    # (no daily hits, never an error) only until that first post-enable deploy runs —
    # if that embed is skipped/fails, the channel simply stays empty until it succeeds.
    daily_knowledge_enabled: bool = True

    # #405: add a question↔past-answer similarity (qsim) term to the C6 score. The
    # scorer's topic_fit sees only the topic TAG and saturates (ADR-0006), so it
    # cannot re-rank on the specific question — and when C1 mispredicts the topic it
    # scores the gold expert against the wrong tag and drops them. qsim (max cosine
    # of the question vs the person's past answers, from C4's answer dense channel)
    # rescues those rows. ENABLED by default after the #405 DGX full-graph E2E
    # verification: Hit@3 0.742->0.788 (Top1/R@3/MRR up too), while RouteAcc and
    # person route recall stayed byte-identical at 1.000 (49/49) — routing is
    # untouched because C5 does not read the scorer, so this is safe by
    # construction. The scorer-isolation eval (scripts/research_c6_qsim.py, 2 runs)
    # localised the gain to the rows where C1 mispredicts the topic (Hit@3 on those
    # 0.444->0.778). Set False to restore the pre-#405 (tag-only) ranking.
    question_fit_enabled: bool = True

    # #87: score the WHOLE employee roster in C6 instead of only the people C4's
    # top chunks surfaced. C4's narrowing drops people who hold the evidence but
    # whose chunks did not rank, and that loss was measured (層2 R@3 -0.048 at
    # top-10, -0.020 at top-20/40 — i.e. it is the reachable SET, not the cut-off).
    # At a 40-person roster the scorer is a deterministic few-ms computation, so
    # there is no cost argument for narrowing. OFF by default until the full-graph
    # E2E measurement confirms Hit@3 improves without moving person route recall
    # (the route reads C4's set separately and must stay at 1.000 — ADR-0007).
    # MEASURED AND NOT ADOPTED (ADR-0009): on the real graph, 3 paired replicates
    # put Hit@3 at 0.7778 (C4 pool) vs 0.7626 (whole roster) — consistently one row
    # worse, never better; person route recall stayed 1.000 either way. C4's set is
    # not only a narrowing, it is the prior "this person was retrievable for this
    # question"; widening it lets a high-generic-topic_fit person with no
    # question-specific evidence displace the right one. #405's qsim only partly
    # offsets that: it is built from the ANSWER dense channel's hits (pool = 50 for
    # every current call site, where top_k = 10), so it is 0 for anyone whose past
    # answers did not reach that pool. On this corpus that is a different set from
    # `candidate_people` (top_k = 10) — neither contains the other — so widening to
    # the roster still adds people who can take no question-specific credit.
    # An earlier version of this comment said "qsim is 0 for anyone C4 did not
    # surface"; that is WRONG (ADR-0009 carries the correction) and it understated
    # qsim's reach — the true nonzero domain is LARGER than `candidate_people`.
    # So treat this sentence as secondary, exactly as ADR-0009 does: the decision
    # rests on the measured Hit@3 above, not on this mechanism being airtight.
    # The flag stays so the measurement is one command away when the scorer or
    # corpus changes.
    # At thousands of employees the pool should come from person_topic_edges by
    # topic rather than the whole table; that is a cost concern, not an accuracy one.
    score_all_employees: bool = False

    # #83: when the asker explicitly asks for someone at a given branch ("福岡の拠点で
    # 動ける方だと助かります"), treat it as a CONDITION in C6 rather than a scoring term.
    #
    # OFF, and NOT yet enablable — the measurement that would justify enabling it is
    # CONTAMINATED. The DGX full-graph A/B looks strong (Hit@3 0.7626 -> 0.8232 over 3
    # paired replicates, +0.0606 every time, routing byte-identical), and C1's branch
    # extraction measured 15/15 with 1 false fire in 72. But the C1 prompt was written
    # AFTER reading the eval's constrained rows: its two special rules (本部->本社,
    # 地方名->拠点) cover exactly the ids a naive extractor misses
    # (`ablation/robustness_results.json`: 11/13/15/17/19/21/24/25), and its negative
    # example is a verbatim lift of eval row 1. So both numbers measure a prompt fitted
    # to the rows it is scored on — the same defect that got "C1 few-shot Hit@3 0.803"
    # retracted (#384). Enabling needs held-out constraint phrasings authored without
    # reference to this prompt.
    #
    # Also unsettled: the committed bench has 「拠点一致を加点」 and 「拠点で絞ってから
    # 並べる」 at the SAME 0.9333/0.7727 (`robustness_results.json`), so a one-line
    # `proximity` weight bump may buy the same thing as this new code path.
    branch_constraint_enabled: bool = False

    # #247: the 直接相談 retrospective. ENABLED by default — this is a kill switch,
    # not a rollout gate: the feature is the only durable record a face-to-face
    # consultation leaves, so shipping it OFF would ship nothing. It exists because
    # this is the one UI-reachable write that feeds C6 expertise evidence; if the
    # write-ups ever turn out to be noisy (or abused despite the accepted-hand-off
    # check), the scoring input has to be stoppable without a rollback. Read-only
    # ``GET /consult-retrospective/{session_id}`` stays up when this is off, so the
    # screen still renders and the form's 503 branch can say the feature is
    # switched off rather than "retry" something that cannot succeed.
    consult_retrospective_enabled: bool = True

    # #357: knowledge framework. When the knowledge layer is wired into retrieval,
    # answer a question from structured knowledge units (problem → action → result,
    # with provenance) instead of, or before, pointing at a person. OFF by default —
    # this is the schema/CRUD skeleton slice; extraction, vector retrieval, and the
    # graph wiring land in later slices, and the graph does not call it yet. Enable
    # only after the extraction quality + A/B are verified on the eval.
    # #357: answer directly from structured knowledge units. Slice 4c wired the
    # ``knowledge_answer`` node into the graph (before C5), so flipping this to True
    # DOES make /ask try a knowledge answer live. OFF by default until the knowledge
    # corpus is populated (extract + embed + human-approve) AND the frontend renders
    # the ``knowledge`` citation kind (see #357 slice-4d gate). Off = pre-#357 graph.
    knowledge_retrieval_enabled: bool = False
    # #357 slice 4: the cosine-similarity floor a retrieved knowledge unit must clear
    # to be used in a knowledge answer. CALIBRATED on the eval (slice 4b A/B, DGX,
    # real Nemotron): CRM-topic queries score 0.224–0.305 against CRM case units,
    # non-CRM queries 0.015–0.312 (compressed cosines). A floor of 0.20 answers all
    # 3 CRM eval queries (topic-matched) with only 3/84 non-CRM false-fires; 0.35
    # (the initial guess) sat ABOVE the CRM range and would never fire. Small CRM
    # sample (n=3) — directional. Inert while knowledge_retrieval_enabled=False.
    knowledge_answer_min_similarity: float = 0.20

    # #371: fold the C1-extracted topics into the C4 retrieval query. The idea was
    # that a multi-facet question collapses onto the thicker-corpus facet under a
    # single dense query; folding topics in surfaced each facet on the RETRIEVAL
    # harness (gold_topics fed to the scorer — an ORACLE): R@3 0.7903 -> 0.8306.
    # BUT the full-graph E2E run (#380, real C1 predicting topics at acc@1=0.750)
    # showed the opposite: folding NOISY predicted topics into the query shifts the
    # C5 retrieval confidences and BREAKS routing — person recall 1.000 -> 0.776,
    # RouteAccuracy 0.833 -> 0.667, with Hit@3 unchanged (0.742). The oracle-harness
    # win did not survive real topic prediction. **DO NOT ENABLE** on the live graph
    # until C1 topic quality is high enough that expansion stops mis-routing (the
    # real lever is C1 intent, not retrieval). Kept OFF and inert; the c4_retrieve
    # branch is byte-for-byte dormant while False.
    query_expansion_enabled: bool = False

    # LangGraph checkpointer for session persistence / interrupt-resume:
    # "memory" = in-process MemorySaver (safe default, works without a DB);
    # "postgres" = PostgresSaver over ``database_url`` (production). A ``Literal``
    # so a typo is rejected at startup; a valid "postgres" that cannot be set up
    # still falls back to MemorySaver at runtime (see the checkpointer factory).
    checkpointer_backend: Literal["memory", "postgres"] = "memory"

    # Backpressure (#180): a SOFT, best-effort admission gate — NOT a hard concurrency
    # ceiling. When this many graph runs are already executing, /ask sheds NEW
    # questions with HTTP 503 (a fast, graceful "混雑中"). Two caveats by design: it is
    # check-then-run (a burst of near-simultaneous /ask→/events can transiently
    # overshoot before any slot is taken), and it gates ONLY new questions — resumes /
    # continuations of paused runs are never shed. The HARD per-GPU bound is vLLM's own
    # ``max-num-seqs`` scheduler; this just keeps the app from accepting far more than
    # the model can batch. Defaults to 8 to match a typical ``max-num-seqs``. 0
    # disables the gate (fine for local/stub dev).
    max_concurrent_runs: int = 8

    # Whether to FAIL-CLOSED on durability (#180): refuse to start on a non-durable
    # checkpointer (``memory``, or a ``postgres`` whose setup fails) instead of
    # silently degrading to in-memory — which would drop every session on the next
    # restart. ``None`` (default) derives from ``app_env`` (enforced when it is not
    # "development"). It is a SEPARATE knob from ``app_env`` on purpose: the DGX host
    # must run ``app_env=development`` for an unrelated reason (#108/#173 — the local
    # embedding model has no pinnable revision), so tying durability to ``app_env``
    # alone would leave it OFF exactly where it matters. Set
    # ``TEKIJIN_STRICT_DURABILITY=true`` on such a host to enforce it regardless, or
    # ``false`` as an explicit escape hatch in a throwaway prod-flavored env.
    strict_durability: bool | None = None

    def durability_enforced(self) -> bool:
        """True when a non-durable checkpointer must be a hard error, not a fallback."""

        if self.strict_durability is not None:
            return self.strict_durability
        return self.app_env != "development"

    # Embedding model used by the retrieval component (C3). Chosen by the on-DGX
    # benchmark (#61): Nemotron-3-Embed-1B was 1st of 5 on the primary metric
    # (層2 Recall@3 0.615 vs e5-large 0.530). See docs/benchmarks/README.md.
    embedding_model: str = "nvidia/Nemotron-3-Embed-1B-BF16"

    # Whether to prepend e5-style ``query:`` / ``passage:`` prefixes when
    # embedding. Correct for the e5 family AND Nemotron-3-Embed (both expect
    # ``query: `` / ``passage: ``); other models (e.g. ruri, BGE, Qwen3-Embedding
    # instruction-style) must NOT receive these prefixes or retrieval quality
    # degrades. Kept configurable because the embedding model is still being
    # benchmarked — flip this off when switching to a non-prefix model.
    embedding_use_e5_prefix: bool = True

    # Per-kind instruction prefixes, overriding the e5 ``query:``/``passage:`` pair
    # for that kind. Needed to reproduce an instruction-tuned fallback model's
    # benchmarked retrieval setup: the #61 Qwen3-Embedding bench prefixes QUERIES
    # with ``Instruct: <task>\nQuery: `` and PASSAGES with nothing (see
    # scripts/bench_embeddings.py). ``None`` (default) falls back to
    # ``embedding_use_e5_prefix`` for that kind; an empty string is a MEANINGFUL
    # override (= no prefix), distinct from ``None``. Set both when switching to
    # Qwen so index-time and query-time prefixes match the bench (#108).
    embedding_query_prefix: str | None = None
    embedding_passage_prefix: str | None = None

    # Whether to allow the embedding model to execute its own (remote) modeling
    # code at load time (``SentenceTransformer(..., trust_remote_code=True)``).
    # REQUIRED by the default Nemotron-3-Embed-1B (it ships custom modeling code);
    # without it ``make embed`` fails during model loading. SECURITY: this runs
    # code from the model repo, so keep it True only for a trusted, pinned default
    # and set it False when pointing ``embedding_model`` at an untrusted source.
    embedding_trust_remote_code: bool = True

    # Immutable model revision (git commit SHA or tag on the HF repo) to load.
    # SECURITY: with ``trust_remote_code=True``, loading the mutable default branch
    # means an upstream change/compromise would execute new code on the next cold
    # load. Pin a reviewed revision here for any real deployment so the executed
    # code is fixed. ``None`` loads the default branch (fine for local/dev).
    embedding_model_revision: str | None = None

    # Dimensionality of the embedding vectors produced by ``embedding_model``.
    # Nemotron-3-Embed-1B emits 2048-d vectors; this drives the width of every
    # ``pgvector`` column so the schema and the model agree. NOTE: the columns
    # stay ``vector`` (not ``halfvec``): ``vector`` stores up to 16000 dims and
    # the retrieval is brute-force, so 2048 is fine everywhere (incl. pgvector
    # 0.6.2). ``halfvec`` is only needed once an HNSW/ivfflat ANN index is added
    # (``vector`` indexes cap at 2000 dims) — deferred to #101.
    embedding_dim: int = 2048

    # RRF weight for the BM25 (sparse) channel in C4 hybrid search; the dense
    # channels stay at 1.0. Equal-weight RRF cost -0.170 層2 R@3 on eval_person v2
    # (queries are symptom-worded, so lexical BM25 ranks are noisy — #68 /
    # docs/benchmarks/ablation.md §3). Down-weighting BM25 to ~0.2 recovers dense
    # -only accuracy. A fixed low weight under-serves product-name / model-number /
    # error-code queries (where BM25 is the only signal); making it adaptive to the
    # dense signal strength is tracked in #114. 0.0 disables BM25 entirely.
    bm25_weight: float = 0.2

    # #114: make the BM25 RRF weight ADAPTIVE to the dense signal strength, to
    # reconcile the conflicting optima — symptom-worded eval queries want BM25 low
    # (``bm25_weight``), but product-name / model-number / error-code queries (where
    # the dense embedding is semantically uninformed and its top cosine is low) want
    # BM25 high or the exact match falls out of top-k (#68 warned of this).
    # ``bm25_weight_boosted`` is the weight when the dense channel is UNINFORMED
    # (its top cosine <= ``bm25_adapt_lo``); it decays linearly back to
    # ``bm25_weight`` as the dense confidence rises to ``bm25_adapt_hi``. ``None``
    # (default) keeps the FLAT ``bm25_weight`` everywhere — i.e. adaptivity is OFF.
    #
    # DGX tuning (2026-08-25, real Nemotron embeddings, scripts/research_bm25_sweep.py):
    # swept boosted ∈ {0.5,1.0,1.5} × window {0.15-0.35,0.10-0.25,0.20-0.45} on the
    # current eval. **boosted=0.5 / window 0.15-0.35 is the sweet spot** — layer-2
    # R@3 0.679→0.694 (+0.015), with-gold-topics 0.723→0.739, L2 0.708→0.736, no
    # L1/L3 regression, route accuracy unchanged (0.818). boosted=1.5 and a wider
    # window (0.20-0.45) regress; 0.10-0.25 is a no-op. So the DEFAULT window
    # (0.15-0.35) is already correct; the recommended boosted is 0.5.
    #
    # Still OFF by default: the gain is small and the PRIMARY target — product-name
    # / model-number queries — is not yet in the eval (that query set is #296's
    # remaining work), and the #70 misrejections (#37/#49) did NOT improve under any
    # config, so BM25 boosting alone is not their fix. Enable (set
    # ``TEKIJIN_BM25_WEIGHT_BOOSTED=0.5``) once the 型番/product-name eval confirms
    # the intended retrieval gain there. Re-run scripts/research_bm25_sweep.py to
    # re-tune when the corpus or embedding model changes.
    bm25_weight_boosted: float | None = None
    bm25_adapt_lo: float = 0.15
    bm25_adapt_hi: float = 0.35

    # Directory holding synthetic fixtures used for development/testing.
    fixtures_dir: Path = _DEFAULT_FIXTURES_DIR

    # CORS allowed origins. Explicit (wildcard "*" + credentials is rejected by
    # browsers) and an immutable tuple so the cached singleton cannot be mutated.
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)

    # --- Authentication (#241) ------------------------------------------------ #
    # HS256 signing secret for access tokens. The default is INSECURE (a fixed
    # dev value) — set TEKIJIN_AUTH_SECRET to a long random string in any real
    # deployment, or every issued token is forgeable. The startup guard
    # ``auth_enforced()`` refuses to boot on this default outside development.
    auth_secret: str = DEV_AUTH_SECRET

    # Access-token lifetime in hours (the "失効" of a stateless JWT). Logout is a
    # client-side token drop; there is no server session to revoke, so keep this
    # modest. 12h covers a demo/work session without a mid-use expiry.
    auth_token_ttl_hours: float = 12.0

    # The single ADMIN account. It is NOT a DB employee (the seeded roster stays
    # at 40 and the admin never becomes a recommendation candidate); it logs in
    # with these credentials and impersonates any employee via the demo switcher.
    # CHANGE admin_password in any real deployment (the default is a known value).
    admin_email: str = "admin@tekijin.local"
    admin_password: str = DEV_ADMIN_PASSWORD
    admin_name: str = "管理者"

    # Password seeded for EVERY employee (demo login). All synthetic, so a shared
    # demo password is acceptable and documented; override per environment.
    demo_user_password: str = "tekijin-demo"

    # Login brute-force throttle: at most this many FAILED attempts per email
    # within the rolling window before further attempts are refused (429).
    login_max_attempts: int = 5
    login_window_seconds: float = 300.0

    # Feedback flood throttle (#263): at most this many POST /feedback per actor
    # within the rolling window before further posts are refused (429). Feedback is
    # an append-only learning signal, so this bounds metric/learning-signal spam.
    feedback_max_per_window: int = 60
    feedback_window_seconds: float = 60.0

    # The ONE Slack workspace this deployment belongs to (``T...``), from
    # ``auth.test`` or the App's Basic Information. Slack's OAuth authenticates
    # any workspace's user against our client id, so the team the token comes
    # back with is the only thing separating a colleague from a stranger who
    # found the callback URL.
    #
    # Blank = accept any workspace, which is the pre-#406 behaviour kept for
    # local/demo setups that have no real workspace to name. Anything that turns
    # workspace membership into an AUTHORISATION decision must require it
    # explicitly rather than inherit this permissive default.
    slack_team_id: str = ""

    # Accept "Sign in with Slack" as a way to OBTAIN a session, not just to attach
    # a Slack identity to an existing one (#406 案A). Off by default: turning it on
    # makes workspace membership an authorisation boundary, which is only sound
    # once ``slack_team_id`` names the workspace — ``_enforce_secure_auth`` refuses
    # to boot on the combination rather than trusting an operator to notice.
    slack_login_enabled: bool = False

    # Treat the Slack workspace as the roster and reconcile `slack_links` against
    # it on a schedule (#406 step 3). OFF by default, and not merely out of
    # caution: #406 documents that registering everyone BEFORE evidence
    # extraction (#404) lands gives each new colleague ``topic_fit = 0``, so the
    # candidate list grows while the recommendation behind it does not — the
    # screen gets busier and the answers get worse. Turn this on only once there
    # is evidence for the people it would add.
    slack_user_sync_enabled: bool = False

    # Let that sync CREATE an employee for a Slack member who has none (#406 /
    # #402). A second flag on top of the first, because it is a much larger
    # statement: "reconcile the links of people we already know" becomes
    # "membership of this Slack workspace makes you an employee here", and that
    # should not ride along with the smaller one. Until #402 there was no
    # non-destructive way to add a person at all — ``make seed`` TRUNCATEs every
    # table — so someone who joined Slack could never sign in.
    #
    # NOTE: a created colleague has no evidence, so ``topic_fit`` is 0 and the
    # recommender will not surface them (#401). Creating the row makes them
    # reachable; #404 (evidence from Slack) is what makes them recommendable.
    slack_user_sync_create_employees: bool = False

    # Company email domains a Slack member must be in before the sync will CREATE
    # an employee for them. Empty allows any address, matching the other switches
    # here (restricting is opt-in). Narrows, but does not close, the squatting
    # risk documented in ``docs/slack-user-sync.md``: an address inside the
    # company domain that nobody holds yet can still be claimed by whoever puts
    # it on their Slack profile first.
    slack_user_sync_allowed_domains: tuple[str, ...] = ()

    # FastAPI's auto-docs (``/docs``, ``/redoc``, ``/openapi.json``). Default OFF:
    # ``/openapi.json`` publishes every endpoint's path, parameters and types, which
    # is a map of the protected surface even though no data leaks. It was reachable
    # on the internet-exposed deploy (#452/#457) purely because nobody set anything,
    # so the default has to be the safe one.
    #
    # A SEPARATE knob, not derived from ``app_env`` — the DGX runs
    # ``app_env=development`` for an unrelated reason (#108/#173), so gating on that
    # would leave production open. Same reasoning as ``strict_durability`` /
    # ``strict_auth``; set ``TEKIJIN_EXPOSE_API_DOCS=true`` on a dev box that wants them.
    expose_api_docs: bool = False

    # Fail-closed on insecure auth defaults (#241), mirroring ``strict_durability``.
    # ``None`` (default) derives from ``app_env`` (enforced when not "development").
    # A SEPARATE knob because the DGX host runs app_env=development for an unrelated
    # reason (#108/#173), so set TEKIJIN_STRICT_AUTH=true there to still refuse a
    # boot with the known default secret/admin password; or =false as an explicit
    # escape hatch in a throwaway prod-flavored env.
    strict_auth: bool | None = None

    def auth_enforced(self) -> bool:
        """True when insecure default auth secrets must be a hard startup error."""

        if self.strict_auth is not None:
            return self.strict_auth
        return self.app_env != "development"

    # --- Slack integration (chat <-> Slack DM, #388) --------------------------- #
    # "Sign in with Slack" OAuth app credentials for account linking, plus a bot
    # token for sending DM notifications once linked. All blank by default — no
    # real Slack App has been registered yet — which leaves the feature disabled
    # (see ``slack_configured()``) rather than failing at import time. Set these
    # via TEKIJIN_SLACK_* env vars once a Slack App exists.
    slack_client_id: str = ""
    slack_client_secret: str = ""
    # Must exactly match a Redirect URL registered on the Slack App (points at
    # this backend's GET /slack/oauth/callback, e.g. http://localhost:8000/slack/oauth/callback).
    slack_redirect_uri: str = ""
    # Bot token (xoxb-...) used only to send the DM notification itself — never
    # the linking user's own token, so there is nothing per-user to refresh.
    slack_bot_token: str = ""
    # Signing Secret (Slack App -> Basic Information), used to verify that a
    # POST /slack/events request genuinely came from Slack (#388: a reply typed
    # in the linked Slack DM lands back in the TEKIJIN thread). Requires the
    # backend to have a public URL Slack can reach (e.g. ngrok in local dev) —
    # a purely local/offline setup can leave this blank and use notifications
    # (TEKIJIN -> Slack) only.
    slack_signing_secret: str = ""
    # Where the OAuth callback sends the browser back to once linking finishes.
    slack_frontend_url: str = "http://localhost:3000"
    # #476 Screen 02: capture a resolved hand-off as a knowledge draft when a
    # participant reacts ✅ on the pair-channel thread. OFF (dormant) by default:
    # when False, the reaction handler is a no-op and no extraction runs, so the
    # Slack events path is byte-identical to today. Enable only once the extraction
    # backend (vLLM) is available and the draft quality is verified. Shared
    # pair-channels only (TEKIJIN never creates DMs), so this never reads a DM —
    # the #404 DM-consent question does not arise here by construction.
    #
    # Reaction→thread attribution (#508, resolved): a ✅ is attributed to the thread
    # the reacted MESSAGE belonged to via its recorded per-message anchor
    # (``slack_message_anchors``), not the channel's mutable ``current_thread_id`` —
    # correct even on a channel reused across hand-offs. A message with no anchor
    # (posted before capture was enabled) still falls back to ``current_thread_id``.
    slack_solve_capture_enabled: bool = False

    # ---- Discord (#60 D1, #66 members) ---------------------------------------- #
    # A Discord user becomes a member of the server's organization on first contact
    # (``data/members.py``); there is no user map to maintain.
    # Shared secret the Discord bot process sends in ``X-Tekijin-Bot-Secret`` when it
    # calls ``POST /internal/discord/ask``. Empty (default) = the endpoint is 404, so
    # Discord is OFF unless both this and the bot's token are configured.
    discord_bot_secret: str = ""
    # Web app URL the bot points to when a question needs the web (sending a
    # hand-off, answering a clarification).
    discord_web_url: str = "http://localhost:3000"
    # Read only by the bot process (``python -m tekijin.discord_bot``), never by the API.
    discord_bot_token: str = ""
    discord_backend_url: str = "http://127.0.0.1:8000"
    # The one Discord server TEKIJIN serves (#66): it is the organization, and
    # requests from any other server are refused. The bot also syncs its commands
    # to this server instantly instead of globally (which can take up to an hour).
    discord_guild_id: int | None = None

    def slack_configured(self) -> bool:
        """True once the OAuth app credentials needed to start linking exist.

        ``slack_bot_token`` is checked separately (:func:`slack_notifications_enabled`)
        since notifications and linking can be turned on independently.
        """

        return bool(self.slack_client_id and self.slack_client_secret and self.slack_redirect_uri)

    def slack_notifications_enabled(self) -> bool:
        """True once a bot token exists to actually send DM notifications."""

        return bool(self.slack_bot_token)


@lru_cache
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton."""
    return Settings()

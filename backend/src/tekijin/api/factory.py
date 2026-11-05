"""Build the default production :class:`AgentService` from settings.

Wires the real pieces: a sessionmaker over the configured database, the selected
checkpointer (memory/postgres with fallback), the lazy SentenceTransformer
embedder (no model loaded until first ``encode``), and the selected LLM backend
(stub/vllm). All construction here is cheap and import-light — no DB connection,
no model download, no LangChain OpenAI import on the stub path.
"""

from __future__ import annotations

from tekijin.api.checkpointer import make_checkpointer
from tekijin.api.service import AgentService
from tekijin.config import Settings, get_settings
from tekijin.data.db import get_engine, get_sessionmaker
from tekijin.llm.factory import make_llm_nodes, make_question_structurer
from tekijin.retrieval.embedding import SentenceTransformerEmbedder


def build_default_service(settings: Settings | None = None) -> AgentService:
    settings = settings or get_settings()
    session_factory = get_sessionmaker(get_engine(settings.database_url))
    intent_model, sufficiency_model, draft_model, answerability_model, self_answer_model = (
        make_llm_nodes(settings)
    )
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        # All settings-driven from THIS instance (not the cached global), so a
        # custom/hardened Settings passed to build_default_service is honored.
        use_e5_prefix=settings.embedding_use_e5_prefix,
        query_prefix=settings.embedding_query_prefix,
        passage_prefix=settings.embedding_passage_prefix,
        trust_remote_code=settings.embedding_trust_remote_code,
        revision=settings.embedding_model_revision,
        # app_env too, so the fail-closed guard checks THIS instance's env rather
        # than the global singleton (#108) — otherwise a hardened production Settings
        # built here could be validated against a development global and bypassed.
        app_env=settings.app_env,
    )
    return AgentService(
        session_factory=session_factory,
        checkpointer=make_checkpointer(settings),
        embedder=embedder,
        intent_model=intent_model,
        sufficiency_model=sufficiency_model,
        draft_model=draft_model,
        # #475 Screen 01: on-demand question re-drafter for the result-screen endpoint
        # (POST /handoff/structure). Built separately from the graph nodes — it never
        # runs on the C1 critical path. Always wired (stub or vLLM per llm_backend).
        question_structurer=make_question_structurer(settings),
        # #70: wire the evidence-sufficiency critic ONLY when enabled; otherwise
        # pass None so the graph compiles the pre-#70 flow (no critique node).
        # Default OFF until it is verified on the DGX eval (part3).
        answerability_model=answerability_model if settings.answerability_enabled else None,
        answerability_threshold=settings.answerability_threshold,
        # #291: wire the self-answer composer when enabled; else None keeps the
        # pre-#291 data-derived routes (document terminal / hand-off). ENABLED by
        # default after the #380 full-graph E2E verification (fires only on the
        # data-derived routes after C5 — person routing recall stays 1.000).
        self_answer_model=self_answer_model if settings.self_answer_enabled else None,
        # From THIS settings instance (not the cached global) so a custom Settings
        # is honored when the graph builds its C4 retriever (#68).
        bm25_weight=settings.bm25_weight,
        # #327: corpus-count routing for prior_answer from THIS settings instance
        # (None = OFF keeps prior_answer dormant, C5 unchanged).
        prior_answer_reuse_min=settings.prior_answer_reuse_min,
        prior_answer_relevance_floor=settings.prior_answer_relevance_floor,
        # #371: fold C1 topics into the C4 retrieval query. DORMANT (False) and must
        # stay OFF: the #380 full-graph E2E run showed folding real C1 topics
        # (acc@1=0.750) into the query BREAKS routing (person recall 1.000->0.776);
        # the retrieval-harness "+0.04" was an oracle-topic artifact. See config.py.
        query_expansion_enabled=settings.query_expansion_enabled,
        # #405: add the question↔past-answer term to C6. ENABLED by default after
        # the full-graph E2E verification (Hit@3 0.742->0.788, person route recall
        # 1.000 unchanged — C5 does not read the scorer, so routing is untouched).
        question_fit_enabled=settings.question_fit_enabled,
        # #413: additive cited answer on the person route (OFF until DGX full-graph
        # confirms person recall 1.000 and citations fire). Floors the compose call.
        additive_self_answer_enabled=settings.additive_self_answer_enabled,
        additive_self_answer_floor=settings.additive_self_answer_floor,
        score_all_employees=settings.score_all_employees,
        # #475 Screen 01: attach the "N other askers in this area" reassurance count.
        similar_askers_enabled=settings.similar_askers_enabled,
        # #118: offer verified knowledge to the self-answer composer ONLY when
        # enabled; None offers none. Default OFF until the circle is re-measured.
        knowledge_answer_min_similarity=(
            settings.knowledge_answer_min_similarity
            if settings.knowledge_retrieval_enabled
            else None
        ),
        # Backpressure admission limit (#180) from THIS settings instance.
        max_concurrent_runs=settings.max_concurrent_runs,
    )

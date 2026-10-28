"""The C1-C8 graph nodes (plus terminals and control nodes).

Each node is a method returning a *partial* :class:`AgentState` update. Deps
(LLM stubs, embedder, retriever, scorer) are injected so the same graph runs on
the deterministic stubs today and real models later. The two nodes that pause for
human input — ``ask`` (C2 clarification) and ``send`` (responder outcome) — use
LangGraph ``interrupt`` and resume via ``Command(resume=...)``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any, cast

from langgraph.types import interrupt

from tekijin.agent.protocols import (
    AnswerabilityModel,
    BranchSource,
    DraftModel,
    EmployeeSource,
    IntentModel,
    IntentResult,
    Retriever,
    SelfAnswerModel,
    SufficiencyModel,
)
from tekijin.agent.route import PRIOR_ANSWER, decide_route
from tekijin.agent.safety import scan_disallowed
from tekijin.agent.state import AgentState, empty_retrieval
from tekijin.agent.stubs import (
    INTENT_CONFIDENCE_THRESHOLD,
    MAX_FOLLOWUPS,
    collect_known_values,
    missing_required_slots,
)
from tekijin.data.questions import count_similar_prior_askers
from tekijin.knowledge.answer import answer_from_knowledge
from tekijin.retrieval.embedding import QUERY, Embedder
from tekijin.retrieval.fragments import FragmentSource, collect_cited_evidence
from tekijin.scorer.scorer import ExpertiseScorer

_QUESTION_TYPE_DEFAULT = "製品QA"


def _top_by_score(items: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The item with the highest ``score`` (deterministic, order-independent)."""

    if not items:
        return None
    return max(items, key=lambda item: float(item.get("score", 0.0)))


def answerability_evidence(recommendations: Sequence[Mapping[str, Any]]) -> list[str]:
    """One track-record summary line per ranked candidate, for the #70 critic.

    The evidence-sufficiency critic judges "can the company answer this in-house"
    from the *shown* candidates' track record — so we hand it exactly what C6
    produced: each candidate's name/dept plus its scored ``reasons`` details
    (self-reported skills, certifications, past answers…). Candidates with no
    reasons still contribute a line (name/dept), because the critic's job is to
    weigh how strong that record is — an empty ``recommendations`` yields ``[]``,
    which the critic reads as "nobody in-house" and rejects.
    """

    lines: list[str] = []
    for rec in recommendations:
        name = str(rec.get("name") or "").strip()
        dept = str(rec.get("dept") or "").strip()
        who = f"{name}（{dept}）" if dept else name
        details = [
            str(r.get("detail") or "").strip()
            for r in (rec.get("reasons") or [])
            if str(r.get("detail") or "").strip()
        ]
        line = f"{who}: {'; '.join(details)}" if details else who
        if line.strip():
            lines.append(line)
    return lines


def draft_context(state: Mapping[str, Any]) -> tuple[list[str], dict[str, str]]:
    """``(missing, known_values)`` for drafting a hand-off to a given responder.

    Factored out of :meth:`AgentNodes.c7_draft` so the ``/handoff/select``
    reselect path (#A1) can regenerate a draft for a different candidate using
    the exact same slot logic, without duplicating it.
    """

    question_type = state.get("question_type", _QUESTION_TYPE_DEFAULT)
    products = state.get("products") or []
    known_values = collect_known_values(state["question"], question_type, products)
    # A slot must never appear as both a filled premise and an open gap: drop
    # from `missing` anything we now surface under known_values, so the draft
    # cannot show the same slot as "確認済み" and "補足いただきたい" at once
    # (defensive dedup — C2 already recomputes `missing` on the re-understood
    # question via the ask->c1_intent edge, so they normally agree) (#175).
    missing = [slot for slot in (state.get("missing") or []) if slot not in known_values]
    return missing, known_values


class AgentNodes:
    """Bundles the graph's node implementations around their dependencies."""

    def __init__(
        self,
        *,
        intent_model: IntentModel,
        sufficiency_model: SufficiencyModel,
        draft_model: DraftModel,
        embedder: Embedder,
        retriever: Retriever,
        scorer: ExpertiseScorer,
        answerability_model: AnswerabilityModel | None = None,
        answerability_threshold: int = 40,
        self_answer_model: SelfAnswerModel | None = None,
        fragment_source: FragmentSource | None = None,
        prior_answer_reuse_min: int | None = None,
        prior_answer_relevance_floor: float = 0.15,
        knowledge_session: Any | None = None,
        knowledge_answer_min_similarity: float | None = None,
        similar_asker_session: Any | None = None,
        query_expansion_enabled: bool = False,
        question_fit_enabled: bool = False,
        additive_self_answer_enabled: bool = False,
        additive_self_answer_floor: float = 0.20,
        branch_constraint_enabled: bool = False,
        employee_branches: BranchSource | None = None,
        employee_source: EmployeeSource | None = None,
    ) -> None:
        self._intent = intent_model
        self._sufficiency = sufficiency_model
        self._draft = draft_model
        self._embedder = embedder
        self._retriever = retriever
        self._scorer = scorer
        # #70: optional evidence-sufficiency critic between C6 and C7. ``None``
        # (default) keeps the graph exactly as before — no critique node is even
        # added (see build_agent), so this is inert unless explicitly wired.
        self._answerability = answerability_model
        self._answerability_threshold = answerability_threshold
        # #291: optional self-answer composer + the source it re-hydrates evidence
        # from. Both None (default) -> no self_answer node is added (inert).
        self._self_answer = self_answer_model
        self._fragment_source = fragment_source
        # #357 slice 4c: optional knowledge-answer step. ``knowledge_answer_min_
        # similarity`` None (default) -> no knowledge_answer node is added (inert);
        # a float wires the node, which answers a query directly from approved
        # knowledge units before routing (bypassing the C5 separation problem, #327).
        self._knowledge_session = knowledge_session
        self._knowledge_floor = knowledge_answer_min_similarity
        # #475 Screen 01: read session used by c1_intent to count OTHER prior askers
        # in the same topic area. ``None`` (default) -> the count is never computed
        # and never attached, so ``understood`` is byte-identical when the feature
        # is off. Wired by build_agent only when ``similar_askers_enabled``.
        self._similar_asker_session = similar_asker_session
        # #327: corpus-count routing for prior_answer (None = OFF, dormant route).
        self._prior_answer_reuse_min = prior_answer_reuse_min
        self._prior_answer_relevance_floor = prior_answer_relevance_floor
        # #371: fold the C1 topics into the C4 retrieval query. False (default) keeps
        # c4_retrieve byte-for-byte the pre-#371 behaviour (raw query + reused C3
        # vector). See c4_retrieve for the multi-facet rationale.
        self._query_expansion_enabled = query_expansion_enabled
        # #405: pass C4's per-person question↔past-answer similarity to the C6
        # scorer as an additive question-fit term. False (default) -> the scorer is
        # never handed the map, so scores are develop-identical. Routing is unchanged
        # either way (C5 does not read the scorer).
        self._question_fit_enabled = question_fit_enabled
        # #83: honour an explicitly requested branch as a CONDITION in C6 rather than
        # a scoring term. False (default) ignores `constraint_branch` entirely, so
        # ranking is develop-identical. `employee_branches` resolves candidate ->
        # branch; without it the feature stays inert even when the flag is on.
        self._branch_constraint_enabled = branch_constraint_enabled
        self._employee_branches = employee_branches
        # #413: additive self-answer on the person route (shares the #291 composer +
        # fragment source). Only active when self_answer is wired AND this is set;
        # the floor gates the compose call so no-data person questions add no latency.
        self._additive_self_answer_enabled = additive_self_answer_enabled
        self._additive_self_answer_floor = additive_self_answer_floor
        # #87: score the WHOLE roster in C6 instead of only the people C4's top
        # chunks happened to surface. ``None`` (default) keeps the C4-derived pool,
        # so develop behaviour is byte-identical. See ``c6_score`` for why the pool
        # and the ROUTE signal must stay separate.
        self._employee_source = employee_source

    # -- entry: validate input, reset per-question control fields ---------
    def reset(self, state: AgentState) -> AgentState:
        """Validate the new question and clear per-question control fields.

        Runs at the START of every fresh ``invoke`` — but NOT on ``resume`` (a
        ``Command(resume=...)`` continues from the interrupted node, bypassing
        START). So a second question on the same ``thread_id`` starts clean
        (no inherited ``followup_count`` / ``declined_ids`` / stale route or
        answer), while the clarification and decline loops keep their state.
        """

        question = state.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question is required and must be a non-empty string")
        now = state.get("now")
        if not isinstance(now, dt.datetime):
            raise ValueError("now is required and must be a datetime")
        if now.tzinfo is not None:
            raise ValueError("now must be timezone-naive (matches stored timestamps)")

        return {
            "followup_count": 0,
            "declined_ids": [],
            "out_of_scope": False,
            "sufficient": False,
            "topics": [],
            "products": [],
            "situation": None,
            "question_type": _QUESTION_TYPE_DEFAULT,
            "intent_confidence": 0.0,
            "intent_unresolved": False,
            "missing": [],
            "followup_question": None,
            "retrieval": empty_retrieval(),
            "route": "person",
            "route_reason": "",
            "route_confidence": 0.0,
            "prior_answer_note": None,
            "pinned_responder_id": None,
            "referred_responder_id": None,
            # #291: clear the self-answer verdict so a second question on the same
            # thread_id never inherits a prior grounded answer / citations.
            "self_answer_grounded": False,
            "self_answer_text": None,
            "self_answer_citations": [],
            # #413: clear the additive (person-route) cited answer too.
            "additive_answer_text": None,
            "additive_citations": [],
            "recommendations": [],
            # #70: clear the critic's per-question verdict so a second question on
            # the same thread_id never reads a prior run's stale score/reason.
            "answerability_confidence": 0,
            "answerability_reason": None,
            "answerable": False,
            "draft": None,
            "outcome": None,
            "answer": None,
            "query_vector": [],
            # Per-question durable persistence identity: clear the rec ids
            # (a fresh question has no shown recommendations yet). ``question_id``
            # is intentionally NOT reset here — it is supplied on the fresh
            # ``invoke`` input and must survive this merge.
            "recommendation_ids": [],
            "primary_recommendation_id": None,
            "last_event": None,
        }

    # -- C1: intent understanding (LLM stub) ------------------------------
    def c1_intent(self, state: AgentState) -> AgentState:
        # Deterministic safety net BEFORE trusting the (prompt-dependent) model:
        # clear PII/secret solicitation or prompt injection is refused regardless of
        # which intent model is wired, so a model swap cannot regress it (#155). It
        # only ADDS rejections; softer out_of_scope cases stay the model's call.
        if scan_disallowed(state["question"]) is not None:
            return {
                "topics": [],
                "products": [],
                "situation": None,
                "question_type": _QUESTION_TYPE_DEFAULT,
                "out_of_scope": True,
                "intent_confidence": 0.0,
                "constraint_branch": None,
            }
        result = self._intent.analyze(state["question"], state.get("asker"))
        update: AgentState = {
            "topics": result.topics,
            "products": result.products,
            "situation": result.situation,
            "question_type": result.question_type,
            "out_of_scope": result.out_of_scope,
            "intent_confidence": result.confidence,
            "constraint_branch": result.constraint_branch,
        }
        # #475 Screen 01: attach the "N other people asked in this area" reassurance
        # count when wired. Read-only, cheap (GIN-indexed topic-overlap count), and
        # gated on a session so it stays inert (understood byte-identical) when off.
        # Never on the out_of_scope short-circuit above (nothing to reassure about).
        if self._similar_asker_session is not None and result.topics and not result.out_of_scope:
            asker = state.get("asker")
            update["similar_asker_count"] = count_similar_prior_askers(
                self._similar_asker_session,
                result.topics,
                exclude_asker_id=asker.get("id") if asker else None,
                exclude_question_id=state.get("question_id"),
            )
        return update

    # -- C2: sufficiency check (LLM stub) ---------------------------------
    def c2_sufficiency(self, state: AgentState) -> AgentState:
        intent = IntentResult(
            topics=state.get("topics", []),
            products=state.get("products", []),
            situation=state.get("situation"),
            question_type=state.get("question_type", _QUESTION_TYPE_DEFAULT),
            out_of_scope=state.get("out_of_scope", False),
            confidence=state.get("intent_confidence", 0.0),
        )
        followup_count = state.get("followup_count", 0)
        # Safety valve (#113): once C1 has confidently identified the topic, we can
        # already decide WHO to route to — the responder can ask for any missing
        # detail (現行製品/対象拠点数…). So don't let C2 block a confident, on-topic
        # consultation on estimate-style slots, no matter how a (prompt-sensitive)
        # model feels. This never fires on a vague/low-signal request (no topic, or
        # confidence below threshold), which still clarifies.
        can_route = bool(intent.topics) and intent.confidence >= INTENT_CONFIDENCE_THRESHOLD
        # Speed (#376): ``can_route`` is decidable from C1's output ALONE, and it
        # forces ``sufficient=True`` anyway — so on the common confident path SKIP the
        # C2 sufficiency LLM call entirely, removing one of the three serial
        # generations on the critical path. We still compute ``missing`` DETERMINISTICALLY
        # (LLM-free, same slot logic the rule model uses) so C7's hand-off draft keeps
        # its 「補足いただきたい点」hint — matching the pre-#376 behaviour, where the
        # model's ``missing`` flowed through even though ``sufficient`` was already
        # forced True by ``can_route``. ``followup_question`` is unused when sufficient
        # (graph goes to C3), and ``intent_unresolved`` is False whenever topics exist.
        if can_route:
            return {
                "sufficient": True,
                "missing": missing_required_slots(state["question"], intent),
                "followup_question": None,
                "intent_unresolved": False,
            }
        result = self._sufficiency.check(state["question"], intent, followup_count)
        # Graph-level termination guarantee: never ask more than MAX_FOLLOWUPS,
        # whatever the (possibly future vLLM) model returns.
        capped = followup_count >= MAX_FOLLOWUPS
        sufficient = result.sufficient or capped
        # If we have already asked once (capped) and STILL have no topic, the
        # intent is unresolved. Rather than silently search on nothing and land in
        # no_candidate, flag it so the graph routes to an explicit "couldn't
        # identify the request" terminal (see _after_c2 / unresolved_intent).
        intent_unresolved = capped and not (state.get("topics") or [])
        return {
            "sufficient": sufficient,
            "missing": result.missing,
            "followup_question": result.followup_question,
            "intent_unresolved": intent_unresolved,
        }

    # -- clarification: pause and ask the user one question ---------------
    def ask(self, state: AgentState) -> AgentState:
        reply = interrupt(
            {
                "followup_question": state.get("followup_question"),
                "missing": state.get("missing", []),
            }
        )
        # Only fold in a genuine, non-empty text reply; never interpolate a
        # non-string payload into the question.
        reply_text = reply.strip() if isinstance(reply, str) else ""
        enriched = f"{state['question']} {reply_text}".strip() if reply_text else state["question"]
        return {"question": enriched, "followup_count": state.get("followup_count", 0) + 1}

    # -- C3: embed the query ----------------------------------------------
    def c3_embed(self, state: AgentState) -> AgentState:
        vector = self._embedder.encode([state["question"]], kind=QUERY)[0]
        return {"query_vector": vector}

    # -- C4: hybrid retrieval ---------------------------------------------
    def c4_retrieve(self, state: AgentState) -> AgentState:
        question = state["question"]
        topics = state.get("topics") or []
        # #371: on a multi-facet question (e.g. "経理×データ基盤") the raw query's dense
        # signal collapses onto the facet with the thicker corpus, dropping the other
        # department's experts out of the top_k candidate pool — the measured cause of
        # ~2/3 of R@3 misses. Folding the C1 topics into the query surfaces each facet
        # (DGX: R@3 0.79->0.83). The expanded string must be RE-EMBEDDED, so the reused
        # C3 vector (which embeds only the raw question) is dropped on this path.
        if self._query_expansion_enabled and topics:
            expanded = f"{question} {' '.join(topics)}"
            retrieval = self._retriever.search(expanded)
        else:
            # Reuse the C3 embedding so the dense channels do not re-embed the query
            # (halves embedding calls under a real vLLM; BM25 still uses raw text).
            retrieval = self._retriever.search(question, query_vector=state.get("query_vector"))
        return {"retrieval": retrieval}

    # -- C5: route decision (deterministic) -------------------------------
    def c5_route(self, state: AgentState) -> AgentState:
        decision = decide_route(
            state.get("retrieval") or empty_retrieval(),
            prior_answer_reuse_min=self._prior_answer_reuse_min,
            prior_answer_relevance_floor=self._prior_answer_relevance_floor,
        )
        return {
            "route": decision.route,
            "route_reason": decision.reason,
            "route_confidence": decision.confidence,
        }

    # -- prior_answer (補助): pin the past responder, then hand off --------
    def prior_answer(self, state: AgentState) -> AgentState:
        retrieval = state.get("retrieval") or empty_retrieval()
        past = retrieval["past_answers"]
        top = _top_by_score(past)
        responder_id = top.get("responder_id") if top else None
        # The pin seats this person at rank 1, ahead of the scorer, so a departed
        # colleague must not reach it (#506). Checked against `departed_people`
        # rather than `candidate_people`: the pin deliberately reaches OUTSIDE
        # the pool — handing off to whoever answered before even when they did
        # not rank is the entire point — so pool membership is the wrong test.
        if responder_id is not None and responder_id in (retrieval.get("departed_people") or []):
            responder_id = None
        note = (
            f"過去にメンバーID {responder_id} が類似の質問に回答しています。本人に取り次ぎます。"
            if responder_id is not None
            else "過去の類似回答を参照します。"
        )
        # Pin the responder so C6/C7 hand off to THEM (本人に追加で聞く), rather
        # than letting a higher-scoring different person win.
        return {"prior_answer_note": note, "pinned_responder_id": responder_id}

    def _candidate_pool(self, retrieval: Mapping[str, Any]) -> list[int]:
        """The people C6 scores: the whole roster when wired, else C4's set (#87)."""

        if self._employee_source is None:
            return list(retrieval.get("candidate_people") or [])
        return [e.id for e in self._employee_source.list_employees()]

    def _requested_branch(self, state: AgentState) -> str | None:
        """The branch C6 must honour, or ``None`` when the #83 constraint is not live."""

        branch = state.get("constraint_branch")
        if not self._branch_constraint_enabled or not branch or self._employee_branches is None:
            return None
        return str(branch)

    def _note_unmet_branch(self, recs: list[dict[str, Any]], state: AgentState) -> None:
        """Stamp "we could not honour 拠点" on any rec that is NOT at the branch.

        Checked per-candidate rather than per-bucket, so the note lands exactly on the
        people it is true of — the prior_answer pin included (it is seated at rank 1
        and would otherwise violate the constraint with no explanation at all), and
        never on someone who IS at the branch.
        """

        branch = self._requested_branch(state)
        if branch is None or not recs:
            return
        assert self._employee_branches is not None  # guaranteed by _requested_branch
        ids = [r["person_id"] for r in recs]
        by_id = self._employee_branches.employees_by_ids(ids)
        note = f"{branch}に該当者が足りないため、他拠点から含めています"
        for rec in recs:
            if getattr(by_id.get(rec["person_id"]), "branch", None) == branch:
                continue
            rec["reasons"] = [
                *(rec.get("reasons") or []),
                {"type": "constraint", "detail": note},
            ]

    def _split_by_branch(
        self, candidates: list[int], state: AgentState
    ) -> tuple[list[int], list[int]]:
        """``(at the requested branch, everyone else)`` for the #83 constraint.

        Returns ``(candidates, [])`` when the constraint is not live. When it IS live
        but NOBODY in the pool is at that branch, this still returns everyone as
        "others" so the caller ranks them normally AND annotates them — "福岡には
        該当者がいません" is the case the asker most needs to hear, and 福岡 has only
        2 employees, so an empty match is common rather than exotic.
        """

        branch = self._requested_branch(state)
        if branch is None:
            return candidates, []
        by_id = self._employee_branches.employees_by_ids(candidates)  # type: ignore[union-attr]
        matching = [c for c in candidates if getattr(by_id.get(c), "branch", None) == branch]
        return matching, [c for c in candidates if c not in set(matching)]

    # -- C6: expertise scorer (deterministic) -----------------------------
    def c6_score(self, state: AgentState) -> AgentState:
        topics = state.get("topics") or []
        retrieval = state.get("retrieval") or empty_retrieval()
        declined = state.get("declined_ids") or []
        pinned = state.get("pinned_responder_id")
        asker = state.get("asker")
        asker_id = asker.get("id") if asker else None
        # A non-empty `recommendations` on entry means this is a reroute backfill
        # (see `reroute`, below): those candidates already survived a decline and
        # keep their rank/score/reasons untouched — only the freed slot(s) are
        # topped up from a fresh rank() call, never a full rescore (#D5/#206).
        existing = state.get("recommendations") or []
        existing_ids = {r["person_id"] for r in existing}

        # #518: a referral is CONSUMED by this one scoring pass — clear it (via
        # ``_scored``) in every return so a LATER, unrelated decline reverts to the
        # normal keep-survivors reroute (else ``reroute`` would keep wiping rank 2/3
        # on every subsequent decline for the rest of the question). No-op when no
        # referral is pending, so the return shape is byte-identical there.
        top_k = 3
        remaining = top_k - len(existing)
        if not topics or remaining <= 0:
            # Nothing to add: keep whatever survived the decline (possibly fewer
            # than 3), or [] on a genuinely fresh run with no topics at all.
            return self._scored(existing, state)

        # #405: hand the scorer C4's question↔past-answer similarity so it can add a
        # question-fit term. When the feature is off, the kwarg is omitted entirely,
        # so rank() is called exactly as before (develop byte-identical, and any
        # scorer double that predates #405 keeps working).
        # ``or {}`` so an enabled feature never silently degrades to dormant on a
        # malformed retrieval missing the key: an empty map -> every qsim is 0.0
        # (no boost) but the feature is genuinely ON, not accidentally None-disabled.
        qsim_kw: dict[str, Any] = (
            {"question_similarity": retrieval.get("person_question_similarity") or {}}
            if self._question_fit_enabled
            else {}
        )

        # prior_answer hands off to the pinned past responder — UNTIL they decline,
        # and never if the pin IS the asker (they cannot answer their own question).
        # In either case drop the pin and rely on the general candidate pool below
        # (never dead-end on a single decline or a self-referential pin).
        #
        # #83: the pin is seated at rank 1, which is who C7 drafts for and who `send`
        # interrupts on. So a pin at the WRONG branch would hand the request to Osaka
        # after the asker explicitly asked for 福岡 — the constraint would be violated
        # by the one candidate that matters, silently. Annotate it rather than drop it:
        # the pin exists because that person already answered a near-duplicate question
        # (#159), which is evidence the branch preference does not outweigh; but the
        # asker must be told the condition was not met.
        prior_pin: int | None = (
            pinned
            if (
                state.get("route") == PRIOR_ANSWER
                and pinned is not None
                and pinned not in declined
                and pinned != asker_id
                and pinned not in existing_ids
            )
            else None
        )
        # #518: a responder's explicit "他に適任者がいる → this person" is seated at
        # rank 1 the same way, but is NOT gated on the prior_answer route — a referral
        # happens on the person route. It takes precedence over a prior_answer pin
        # (the human's explicit choice wins). Dropped once the referred person is
        # declined / is the asker / is already shown, so it never dead-ends or loops;
        # None (the default) makes this whole block a no-op — c6 byte-identical.
        referred = state.get("referred_responder_id")
        referral_pin: int | None = (
            referred
            if (
                referred is not None
                and referred not in declined
                and referred != asker_id
                and referred not in existing_ids
            )
            else None
        )
        pin_id: int | None = referral_pin if referral_pin is not None else prior_pin

        fresh: list[dict[str, Any]] = []
        if pin_id is not None:
            # Guarantee the pinned past responder a slot regardless of how they'd
            # score against the general pool: they already answered a
            # near-duplicate question, a signal the scorer's generic evidence
            # cannot fully capture (#159 "fix G"). The remaining slots below are
            # then filled from the general pool so the asker still sees up to 3
            # candidates (#307) instead of dead-ending on this one person.
            pin_result = self._scorer.rank(
                topics, [pin_id], asker_id, state["now"], top_k=1, **qsim_kw
            )
            fresh = cast("list[dict[str, Any]]", pin_result["recommendations"])
            self._note_unmet_branch(fresh, state)
            # #518: when this pin is a responder referral, tell the asker WHY this
            # person is now rank 1 (a human vouched), so the referral is visible
            # rather than looking like a fresh scorer pick.
            if referral_pin is not None and fresh:
                fresh[0]["reasons"] = [
                    {"type": "referral", "detail": "担当者からの推薦"},
                    *fresh[0].get("reasons", []),
                ]
            remaining -= len(fresh)

        if remaining > 0:
            filled_ids = existing_ids | {r["person_id"] for r in fresh}
            # #87 proposed scoring the WHOLE roster here, on the theory that C4's
            # "who appears in the top chunks" narrowing drops people who HAVE the
            # evidence. MEASURED AND REJECTED (ADR-0009): on the real graph it is
            # consistently one row WORSE (Hit@3 0.7778 -> 0.7626 over 3 paired
            # replicates). `employee_source` keeps the lever so the measurement is one
            # command away, but it is OFF and should stay off absent new evidence.
            #
            # This deliberately changes ONLY the scoring pool. ``candidate_people``
            # is ALSO the C5 person-route signal (`route.py`: `if candidate_people:`),
            # and the route recall it produces is at 1.000 (ADR-0007) — so the route
            # keeps reading C4's set, untouched.
            pool = self._candidate_pool(retrieval)
            candidates = [p for p in pool if p not in declined and p not in filled_ids]
            if candidates:

                def rank(ids: list[int], top_k: int) -> list[dict[str, Any]]:
                    # All topics feed the scorer (aggregated topic_fit), not just topics[0].
                    # The scorer returns typed ScoredCandidate rows; AgentState keeps the
                    # looser list[dict[str, Any]] (also written as plain dicts elsewhere),
                    # so narrow the TypedDict-invariance gap with a cast — identical at
                    # runtime.
                    result = self._scorer.rank(
                        topics, ids, asker_id, state["now"], top_k=top_k, **qsim_kw
                    )
                    return cast("list[dict[str, Any]]", result["recommendations"])

                # #83: an explicitly requested branch is a CONDITION, not a bonus.
                # `Weights.proximity` adds one term to a linear sum, so a strong
                # candidate elsewhere simply outweighs it. Rank the branch's people
                # FIRST and only fall back to the rest to fill the remaining slots —
                # and say so in the reasons, so "福岡には該当者がいないため大阪の人を
                # 含めています" is visible rather than silently ignored.
                matching, others = self._split_by_branch(candidates, state)
                if others:
                    # Constrained: the branch's people fill the slots first, then the
                    # rest — and every rec NOT at the branch is annotated, so the asker
                    # is never silently handed someone elsewhere. `matching` may be
                    # empty (nobody at that branch at all), which is exactly when the
                    # note matters most.
                    picked = rank(matching, remaining) if matching else []
                    short = remaining - len(picked)
                    if short > 0:
                        picked = picked + rank(others, short)
                    self._note_unmet_branch(picked, state)
                    fresh = fresh + picked
                else:
                    fresh = fresh + rank(candidates, remaining)

        if not fresh:
            return self._scored(existing, state)
        return self._scored(existing + fresh, state)

    @staticmethod
    def _scored(recommendations: list[dict[str, Any]], state: AgentState) -> AgentState:
        """c6 result carrying ``recommendations``, clearing a consumed referral (#518)."""
        out: AgentState = {"recommendations": recommendations}
        if state.get("referred_responder_id") is not None:
            out["referred_responder_id"] = None
        return out

    # -- answerability critic (#70): can the company answer this in-house? -
    def answerability(self, state: AgentState) -> AgentState:
        """Rate the shown candidates' evidence 0–100 and decide accept/reject.

        Runs only when a critic is wired (build_agent adds this node then). A
        plausible topic does not imply an in-house expert (海外法務/知財/製造制御…),
        which the topic classifier cannot catch — so we score the *evidence* and
        reject BELOW the injected threshold to the ``no_expert`` terminal instead
        of handing off a weak match. The threshold decision is made here (not in
        the pure router) so the router only reads a boolean.
        """

        assert self._answerability is not None  # only reached when wired
        recs = state.get("recommendations") or []
        evidence = answerability_evidence(recs)
        result = self._answerability.assess(state["question"], evidence)
        return {
            "answerability_confidence": result.confidence,
            "answerability_reason": result.reason,
            "answerable": result.confidence >= self._answerability_threshold,
        }

    # -- C7: draft the request (LLM stub) ---------------------------------
    def c7_draft(self, state: AgentState) -> AgentState:
        top = (state.get("recommendations") or [])[0]
        missing, known_values = draft_context(state)
        draft = self._draft.draft(
            state["question"],
            top,
            state.get("asker"),
            missing,
            situation=state.get("situation"),
            topics=state.get("topics") or [],
            known_values=known_values,
        )
        return {"draft": draft}

    # -- send: pause for the responder's outcome --------------------------
    def send(self, state: AgentState) -> AgentState:
        recs = state.get("recommendations") or []
        top = recs[0] if recs else None
        outcome = interrupt({"draft": state.get("draft"), "responder": top})
        return {"outcome": outcome}

    # -- reroute: the top pick declined; try the next candidate -----------
    def reroute(self, state: AgentState) -> AgentState:
        recs = state.get("recommendations") or []
        declined = list(state.get("declined_ids") or [])
        if recs:
            declined.append(recs[0]["person_id"])
        # Keep the already-shown survivors (rank 2/3, unchanged) instead of
        # wiping the whole set: c6_score backfills only the freed slot rather
        # than rescoring everyone from scratch (#D5/#206).
        #
        # #518: EXCEPT a responder referral, which must put the NAMED person at rank 1.
        # c6_score appends the referral pin AFTER any survivors, so keeping rank 2/3
        # would leave the named person at the bottom. Dropping the survivors makes
        # ``existing`` empty, so the pin is seated at rank 1 and the pool backfills the
        # rest. Gated on the referral being set, so a plain decline is unchanged.
        kept = [] if state.get("referred_responder_id") is not None else recs[1:]
        return {"declined_ids": declined, "outcome": None, "draft": None, "recommendations": kept}

    # -- C8: graph update (minimal, deterministic) ------------------------
    def c8_update(self, state: AgentState) -> AgentState:
        # The person_topic_edges online write lands in a later issue; here C8 is a
        # deterministic no-op beyond recording the successful hand-off, so the run
        # stays reproducible.
        recs = state.get("recommendations") or []
        name = recs[0]["name"] if recs else "担当者"
        return {"answer": f"{name}さんに取り次ぎました。回答をお待ちください。"}

    # -- terminals --------------------------------------------------------
    def out_of_scope(self, state: AgentState) -> AgentState:
        return {
            "answer": "恐れ入りますが、このご質問にはTEKIJINではお答えできません。"
            "組織の活動や、メンバーの知見に関することをお尋ねください。"
        }

    # -- self-answer (#291): compose a cited answer from retrieved data ----
    def knowledge_answer(self, state: AgentState) -> AgentState:
        """Try to answer directly from structured knowledge units (#357 slice 4c).

        Runs (when wired) right after C3 embeds the query, BEFORE routing — so it
        applies to every route, sidestepping the C5 person/document separation the
        #327 measurement proved unfixable (ADR-0007). Searches approved knowledge
        units by the query embedding and, if one clears the similarity floor,
        composes a grounded answer deterministically (no LLM, no hallucination) and
        terminates at ``self_answered``. No relevant knowledge -> ``grounded=False``
        and the run proceeds to normal retrieval/routing (never a degraded answer).
        Reuses the #291 ``self_answer_*`` state + terminal so this is a drop-in.
        """

        assert self._knowledge_session is not None and self._knowledge_floor is not None
        query_vec = state.get("query_vector") or []
        if not query_vec:
            return {"self_answer_grounded": False}
        result = answer_from_knowledge(
            self._knowledge_session, query_vec, min_similarity=self._knowledge_floor
        )
        if result is None or not result.grounded:
            return {"self_answer_grounded": False}
        citations = [{"source_id": sid, "kind": "knowledge"} for sid in result.cited_source_ids]
        return {
            "self_answer_grounded": True,
            "self_answer_text": result.answer,
            "self_answer_citations": citations,
        }

    def self_answer(self, state: AgentState) -> AgentState:
        """Try to answer directly from the retrieved sources (data-derived routes).

        Runs only when a composer is wired (build_agent adds this node then), on the
        document / prior_answer routes — where C4 found strong data. Re-hydrates the
        top hits as id-paired evidence and asks the composer for a grounded, cited
        answer. ``grounded`` True -> the graph terminates at ``self_answered`` with
        the answer + source links; False -> fall back to the original route (a human
        hand-off for the tacit-knowledge case). Never fabricates an answer.
        """

        assert self._self_answer is not None and self._fragment_source is not None
        retrieval = state.get("retrieval") or empty_retrieval()
        evidence = collect_cited_evidence(self._fragment_source, retrieval)
        result = self._self_answer.compose(state["question"], evidence)
        if not result.grounded:
            return {"self_answer_grounded": False}
        # The composer already filters cited ids to the supplied evidence, so every
        # id resolves here; keep only resolvable ids (never mislabel an unknown one).
        kind_by_id = {e.source_id: e.kind for e in evidence}
        citations = [
            {"source_id": sid, "kind": kind_by_id[sid]}
            for sid in result.cited_source_ids
            if sid in kind_by_id
        ]
        return {
            "self_answer_grounded": True,
            "self_answer_text": result.answer,
            "self_answer_citations": citations,
        }

    def self_answered(self, state: AgentState) -> AgentState:
        # Terminal for a grounded self-answer: surface the composed text and carry
        # the citations forward so the SSE message renders a link per source (#291).
        return {
            "answer": state.get("self_answer_text") or "",
            "self_answer_citations": state.get("self_answer_citations") or [],
        }

    # -- additive self-answer (#413): cite past knowledge ON the person route ---
    def additive_answer(self, state: AgentState) -> AgentState:
        """Compose a cited answer to surface ALONGSIDE the person hand-off.

        Runs on the PERSON route (wired only when additive self-answer is enabled).
        Unlike ``self_answer`` it never terminates and never marks the run
        self-resolved — the hand-off always follows (person recall unchanged). A low
        relevance FLOOR gates the compose LLM call so a person question with no
        relevant data adds no latency; a grounded result is attached to
        ``additive_answer_text`` / ``additive_citations`` for additive display, and
        an ungrounded one leaves them empty (plain hand-off, as before).
        """

        assert self._self_answer is not None and self._fragment_source is not None
        retrieval = state.get("retrieval") or empty_retrieval()
        # Gate: only spend a compose call when at least one data channel is weakly
        # relevant. Below the floor -> skip (no LLM, no citation), just hand off.
        top_conf = max(
            float(retrieval.get("answer_confidence") or 0.0),
            float(retrieval.get("document_confidence") or 0.0),
        )
        if top_conf < self._additive_self_answer_floor:
            return {}
        evidence = collect_cited_evidence(self._fragment_source, retrieval)
        # This is a BEST-EFFORT additive step: the hand-off MUST always follow, so a
        # composer failure (e.g. the LLM returns unparseable structured output ->
        # ValueError) must degrade to "no citation" rather than crash the person run
        # and take the hand-off down with it. The whole safety premise of #413 is
        # that person recall never regresses; swallow and proceed to c6_score.
        try:
            result = self._self_answer.compose(state["question"], evidence)
        except Exception:  # noqa: BLE001 — additive best-effort, never break hand-off
            return {}
        if not result.grounded:
            return {}
        kind_by_id = {e.source_id: e.kind for e in evidence}
        citations = [
            {"source_id": sid, "kind": kind_by_id[sid]}
            for sid in result.cited_source_ids
            if sid in kind_by_id
        ]
        return {"additive_answer_text": result.answer, "additive_citations": citations}

    def document(self, state: AgentState) -> AgentState:
        docs = (state.get("retrieval") or empty_retrieval())["documents"]
        top = _top_by_score(docs)
        doc_id = top["doc_id"] if top else None
        where = f"（文書ID: {doc_id}）" if doc_id else ""
        answer = f"組織の文書に該当がありそうです{where}。該当箇所をご確認ください。"
        # #279: C6 ran on the document route too, so a real expert behind a
        # weak-profile document is offered as a fallback rather than dead-ending
        # at zero person-recall. Self-resolution stays the main line (no hand-off
        # interrupt); the person is a "if this does not solve it" backstop.
        recs = state.get("recommendations") or []
        fallback = recs[0] if recs else None
        if fallback:
            answer += f"（解決しない場合は{fallback['name']}さんにも取り次げます）"
        return {"answer": answer, "document_id": doc_id, "fallback_responder": fallback}

    def no_candidate(self, state: AgentState) -> AgentState:
        return {
            "answer": "現時点で適任者が見つかりませんでした。条件を変えて、もう一度お試しください。"
        }

    def no_expert(self, state: AgentState) -> AgentState:
        # #70: reached when C6 DID rank candidates but the answerability critic
        # judged the in-house track record insufficient — a "社内に痕跡が無い領域"
        # (海外法務/知財/製造制御…) where a plausible topic hides a real gap. Fail
        # gracefully; distinct from ``no_candidate`` (nobody scored at all).
        #
        # Deliberately a FIXED message: the critic's ``answerability_reason`` is
        # kept in state for server-side logging only and is NOT interpolated here.
        # It is free LLM text derived from the (now suppressed) candidates' names/
        # departments, so echoing it would re-surface the very people this reject
        # path exists to withhold — defeating the recommend/persist suppression.
        return {
            "answer": "ご相談の領域に詳しいメンバーの記録が、組織の中に見つかりませんでした。"
            "恐れ入りますが、組織の外の専門家へのご相談もご検討ください。"
        }

    def unresolved_intent(self, state: AgentState) -> AgentState:
        # Reached when even after one clarification we could not identify a topic.
        # Fail gracefully instead of silently returning "no expert found".
        return {
            "answer": "ご相談内容を特定できませんでした。恐れ入りますが、"
            "もう少し具体的なご相談内容をお知らせください。"
        }

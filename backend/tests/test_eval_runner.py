"""Tests for the offline evaluation runner (#33 / technical-spec §7).

Layers:
* pure metric functions — exact, hand-computed expectations;
* dataset loader — the primary ``eval_person.json`` parses + strict validation;
* the runner — orchestration via a deterministic stub ranker, plus a real
  retrieval+scorer pipeline pass over the seed (a regression floor, not the spec
  target: with NULL fixture embeddings BM25 drives retrieval, so absolute numbers
  are lower than the fully-embedded target — we assert the pipeline is wired and
  clears a floor rather than betting CI on absolute accuracy).
"""

from __future__ import annotations

import datetime as dt

import pytest

from tekijin.eval.dataset import VALID_ROUTES, EvalQuery, load_eval_queries
from tekijin.eval.metrics import (
    QueryResult,
    decision_class,
    evaluate,
    evaluate_alt,
    evaluate_by_difficulty,
    evaluate_decisions,
    evaluate_source_recall,
    evaluate_topics,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    source_precision,
    source_recall,
    top1_hit,
    topic_hit_at_k,
)
from tekijin.eval.runner import RankResult, format_report, run_eval

# Anchored just after the fixtures' latest answer (2026-08-21) so the scorer's
# 7-day load window contains recent activity (matches the CLI's EVAL_NOW).
NOW = dt.datetime(2026, 8, 22, 0, 0, 0)


def _qr(
    ranked,
    gold,
    predicted_route="person",
    gold_route="person",
    difficulty="L1",
    gold_alt=None,
    predicted_topics=None,
    gold_topics=None,
    cited_source_ids=None,
    gold_source=None,
) -> QueryResult:
    return QueryResult(
        ranked_experts=ranked,
        gold_experts=gold,
        predicted_route=predicted_route,
        gold_route=gold_route,
        difficulty=difficulty,
        gold_experts_alt=gold_alt or [],
        predicted_topics=predicted_topics or [],
        gold_topics=gold_topics or [],
        cited_source_ids=cited_source_ids or [],
        gold_source=gold_source or [],
    )


def _q(**kw) -> EvalQuery:
    base = {
        "id": 1,
        "query": "q",
        "gold_topics": ["t"],
        "gold_experts": [1],
        "gold_route": "person",
        "difficulty": "L1",
        "expect_abstain": False,
        "gold_experts_alt": [],
    }
    base.update(kw)
    return EvalQuery(**base)


# --------------------------------------------------------------------------- #
# metrics (pure)
# --------------------------------------------------------------------------- #
def test_top1_hit() -> None:
    assert top1_hit(_qr([3, 1, 2], [3, 9])) is True
    assert top1_hit(_qr([1, 3], [3, 9])) is False
    assert top1_hit(_qr([], [3])) is False  # empty ranking never hits


def test_recall_at_k_normalises_by_min_k_and_gold() -> None:
    assert recall_at_k(_qr([1, 2, 9, 4], [1, 2, 3]), 3) == pytest.approx(2 / 3)
    # gold larger than k: denominator capped at k so a full top-k can reach 1.0.
    assert recall_at_k(_qr([1, 2, 3], [1, 2, 3, 4]), 3) == pytest.approx(1.0)
    assert recall_at_k(_qr([1, 2], []), 3) == 0.0  # no gold -> 0.0


def test_recall_at_k_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k(_qr([1], [1]), 0)


def test_hit_at_k_credits_at_least_one_gold_in_topk() -> None:
    # Product-true routing metric (#371 follow-up): success == the user got ≥1
    # genuinely useful expert in the shown top-k, regardless of how many of the
    # 2–4 gold were caught. Diverges from fractional recall exactly on multi-gold.
    assert hit_at_k(_qr([1, 2, 9, 4], [1, 2, 3]), 3) is True  # 2 of gold in top3
    assert hit_at_k(_qr([9, 8, 7, 1], [1, 2]), 3) is False  # gold only at rank 4
    assert hit_at_k(_qr([9, 1], [1, 2, 3, 4]), 3) is True  # 1-of-4 gold == product success
    assert hit_at_k(_qr([], [3]), 3) is False  # empty ranking never hits


def test_hit_at_k_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        hit_at_k(_qr([1], [1]), 0)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(_qr([9, 3, 1], [3])) == pytest.approx(1 / 2)  # first hit at index 1
    assert reciprocal_rank(_qr([3, 9], [3])) == pytest.approx(1.0)
    assert reciprocal_rank(_qr([9, 8], [3])) == 0.0  # miss


def test_topic_hit_at_k() -> None:
    r = _qr([], [], predicted_topics=["B", "A", "C"], gold_topics=["A"])
    assert topic_hit_at_k(r, 1) is False  # top guess "B" is not gold
    assert topic_hit_at_k(r, 3) is True  # gold "A" is within the top 3
    # exact top-1 hit
    assert topic_hit_at_k(_qr([], [], predicted_topics=["A"], gold_topics=["A", "Z"]), 1) is True
    # no prediction never hits
    assert topic_hit_at_k(_qr([], [], predicted_topics=[], gold_topics=["A"]), 3) is False


def test_topic_hit_at_k_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError):
        topic_hit_at_k(_qr([], [], predicted_topics=["A"], gold_topics=["A"]), 0)


def test_evaluate_topics_averages_over_gold_topic_rows_only() -> None:
    results = [
        _qr([], [], predicted_topics=["A"], gold_topics=["A"]),  # acc@1 hit, acc@3 hit
        _qr([], [], predicted_topics=["X", "B"], gold_topics=["B"]),  # acc@1 miss, acc@3 hit
        _qr([], [], predicted_topics=["Y", "Z", "Q"], gold_topics=["A"]),  # both miss
        # abstain row: no gold topics -> excluded from the denominator entirely.
        _qr([], [], predicted_topics=["A"], gold_topics=[]),
    ]
    ta = evaluate_topics(results)
    assert ta.n_topic == 3  # the goldless row is not counted
    assert ta.acc_at_1 == pytest.approx(1 / 3)  # only the first row
    assert ta.acc_at_3 == pytest.approx(2 / 3)  # first two rows
    assert ta.as_dict() == {
        "n_topic": 3,
        "acc_at_1": pytest.approx(1 / 3),
        "acc_at_3": pytest.approx(2 / 3),
    }


def test_evaluate_topics_empty_when_no_gold_topics() -> None:
    ta = evaluate_topics([_qr([], [], predicted_topics=["A"], gold_topics=[])])
    assert ta.n_topic == 0
    assert ta.acc_at_1 == 0.0 and ta.acc_at_3 == 0.0


def test_evaluate_excludes_goldless_and_nonabc_routes() -> None:
    results = [
        # top1 hit, recall@3 = 1/2, route match (person)
        _qr([1, 5, 6], [1, 2], predicted_route="person", gold_route="person"),
        # first hit at rank 2, recall@3 = 1/2, route miss (predicted document)
        _qr([9, 2, 8], [2, 7], predicted_route="document", gold_route="person"),
        # abstain query: no gold experts (excluded from ranking) AND gold_route
        # "none" (excluded from route accuracy) — contributes only to n.
        _qr([], [], predicted_route="person", gold_route="none"),
    ]
    m = evaluate(results)
    assert m.n == 3
    assert m.n_ranked == 2  # goldless query excluded from ranking metrics
    assert m.n_routed == 2  # "none"-route query excluded from route accuracy
    assert m.n_abstain == 1  # the "none"-route query
    assert m.top1_accuracy == pytest.approx(0.5)
    assert m.recall_at_3 == pytest.approx(0.5)
    # Both ranked rows have ≥1 gold in top3 -> Hit@3 == 1.0, diverging from the
    # fractional Recall@3 of 0.5: the product succeeded on both (a useful expert
    # was shown), which fractional recall under-credits on multi-gold rows.
    assert m.hit_at_3 == pytest.approx(1.0)
    assert m.mrr == pytest.approx((1.0 + 0.5) / 2)
    assert m.route_accuracy == pytest.approx(0.5)  # 1 of 2 A/B/C routes match
    # the abstain query produced no experts -> declined correctly.
    assert m.abstain_accuracy == pytest.approx(1.0)

    d = m.as_dict()
    assert set(d) == {
        "n",
        "n_ranked",
        "n_routed",
        "n_abstain",
        "top1_accuracy",
        "recall_at_3",
        "hit_at_3",
        "mrr",
        "route_accuracy",
        "abstain_accuracy",
    }
    assert d["n"] == 3 and d["n_ranked"] == 2 and d["n_routed"] == 2


def test_abstain_accuracy_counts_only_empty_rankings() -> None:
    results = [
        _qr([], [], gold_route="none"),  # declined -> correct
        _qr([7], [], gold_route="none"),  # produced an expert -> failed to abstain
    ]
    m = evaluate(results)
    assert m.n_abstain == 2
    assert m.abstain_accuracy == pytest.approx(0.5)


def test_evaluate_by_difficulty_splits_per_layer() -> None:
    results = [
        _qr([1], [1], difficulty="L1"),  # L1 top1 hit
        _qr([9], [1], difficulty="L2"),  # L2 miss
        _qr([2], [2], difficulty="L2"),  # L2 hit
    ]
    by_layer = evaluate_by_difficulty(results)
    assert set(by_layer) == {"L1", "L2"}
    assert by_layer["L1"].top1_accuracy == pytest.approx(1.0)
    assert by_layer["L2"].top1_accuracy == pytest.approx(0.5)  # 1 of 2
    assert by_layer["L2"].n_ranked == 2


def test_evaluate_alt_uses_alternate_gold_only_where_present() -> None:
    results = [
        _qr([5], [1], gold_alt=[5]),  # alt hit (primary would miss)
        _qr([9], [2], gold_alt=[2]),  # alt miss
        _qr([1], [1], gold_alt=[]),  # no alt -> excluded from the alt run
    ]
    alt = evaluate_alt(results)
    assert alt.n_ranked == 2  # only the two rows with alt labels
    assert alt.top1_accuracy == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# #297 recall-centric metrics: C5 decision recall + C7' source recall
# --------------------------------------------------------------------------- #
def test_decision_class_maps_routes_and_terminals() -> None:
    # data-derived routes + the grounded self-answer terminal -> self_answer
    assert decision_class("document") == "self_answer"
    assert decision_class("prior_answer") == "self_answer"
    assert decision_class("self_answered") == "self_answer"
    # person -> route (取次ぎ)
    assert decision_class("person") == "route"
    # abstain route + the no-expert terminal -> abstain
    assert decision_class("none") == "abstain"
    assert decision_class("no_expert") == "abstain"
    # unknown label -> None (never a silent hit)
    assert decision_class("banana") is None


def test_source_recall_per_row() -> None:
    # cited 1 of 2 gold -> 0.5
    r = _qr([], [], gold_source=["doc_1", "doc_2"], cited_source_ids=["doc_1", "doc_9"])
    assert source_recall(r) == pytest.approx(0.5)
    # should self-answer but cited nothing -> 0.0 (the miss stays in the average)
    assert source_recall(_qr([], [], gold_source=["doc_1"], cited_source_ids=[])) == 0.0
    # no citation obligation -> None (excluded from the denominator)
    assert source_recall(_qr([], [], gold_source=[], cited_source_ids=[])) is None


def test_source_precision_per_row() -> None:
    # cited 2, one gold -> 0.5 (hallucination signal)
    r = _qr([], [], gold_source=["doc_1"], cited_source_ids=["doc_1", "doc_9"])
    assert source_precision(r) == pytest.approx(0.5)
    # cited nothing -> None (no precision to score)
    assert source_precision(_qr([], [], gold_source=["doc_1"], cited_source_ids=[])) is None
    # every citation gold -> 1.0
    assert source_precision(_qr([], [], gold_source=["a", "b"], cited_source_ids=["a"])) == 1.0


def test_evaluate_decisions_per_class_recall() -> None:
    results = [
        # self_answer gold: one hit (document→document), one miss (prior_answer→person)
        _qr([], [], predicted_route="document", gold_route="document"),
        _qr([], [], predicted_route="person", gold_route="prior_answer"),
        # route gold: one hit, one miss (person→document)
        _qr([1], [1], predicted_route="person", gold_route="person"),
        _qr([], [], predicted_route="document", gold_route="person"),
        # abstain gold: one hit via the no_expert terminal
        _qr([], [], predicted_route="no_expert", gold_route="none"),
    ]
    dr = evaluate_decisions(results)
    assert dr.n == 5
    assert dr.per_class["self_answer"].recall == pytest.approx(0.5)  # 1/2
    assert dr.per_class["route"].recall == pytest.approx(0.5)  # 1/2
    assert dr.per_class["abstain"].recall == pytest.approx(1.0)  # 1/1
    # macro averages the three present classes
    assert dr.macro_recall == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    d = dr.as_dict()
    assert d["per_class"]["self_answer"] == {"support": 2, "hits": 1, "recall": 0.5}


def test_evaluate_decisions_omits_absent_classes() -> None:
    # only route-gold rows present: self_answer/abstain must NOT appear as 0/0
    dr = evaluate_decisions([_qr([1], [1], predicted_route="person", gold_route="person")])
    assert set(dr.per_class) == {"route"}
    assert dr.macro_recall == pytest.approx(1.0)


def test_evaluate_source_recall_aggregate() -> None:
    results = [
        # obligated + fully cited
        _qr([], [], gold_route="document", gold_source=["a", "b"], cited_source_ids=["a", "b"]),
        # obligated + half cited, one hallucinated citation
        _qr([], [], gold_route="prior_answer", gold_source=["c", "d"], cited_source_ids=["c", "z"]),
        # obligated but did NOT self-answer (routed to a person): recall 0, not cited
        _qr([1], [1], gold_route="document", gold_source=["e"], cited_source_ids=[]),
        # not obligated (person route, no gold_source): excluded entirely
        _qr([1], [1], gold_route="person", gold_source=[], cited_source_ids=[]),
    ]
    sr = evaluate_source_recall(results)
    assert sr.n == 3  # three citation-obligated rows
    assert sr.n_cited == 2  # two of them cited something
    # recall: (1.0 + 0.5 + 0.0) / 3
    assert sr.recall == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    # precision over cited rows only: (1.0 + 0.5) / 2
    assert sr.precision == pytest.approx((1.0 + 0.5) / 2)
    # 2 of 3 obligated rows produced any citation
    assert sr.grounded_rate == pytest.approx(2 / 3)
    d = sr.as_dict()
    assert d["n"] == 3 and d["n_cited"] == 2
    assert d["recall"] == pytest.approx(0.5) and d["grounded_rate"] == pytest.approx(2 / 3)


def test_evaluate_source_recall_empty_without_obligations() -> None:
    sr = evaluate_source_recall([_qr([1], [1], gold_route="person")])
    assert sr.n == 0 and sr.n_cited == 0
    assert sr.recall == 0.0 and sr.precision == 0.0 and sr.grounded_rate == 0.0


# --------------------------------------------------------------------------- #
# dataset loader (primary eval_person.json + strict validation)
# --------------------------------------------------------------------------- #
def test_load_bundled_eval_person() -> None:
    queries = load_eval_queries()
    assert len(queries) == 87  # eval_person.json v2 (#296 で型番6件を L3 に追加)
    assert all(isinstance(q, EvalQuery) for q in queries)
    assert {q.gold_route for q in queries} <= VALID_ROUTES
    # L4 abstain queries carry no gold experts and route "none".
    l4 = [q for q in queries if q.difficulty == "L4"]
    assert l4 and all(not q.gold_experts and q.gold_route == "none" for q in l4)
    # The independent alternate labels are present on a sizeable slice (README: 45).
    assert sum(1 for q in queries if q.gold_experts_alt) >= 40


def test_load_eval_queries_rejects_missing_keys(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('[{"id": 1, "query": "q"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="missing keys"):
        load_eval_queries(bad)


def test_load_eval_queries_rejects_non_list(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"id": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON list"):
        load_eval_queries(bad)


def test_load_eval_queries_rejects_non_object_row(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[123]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_eval_queries(bad)


def _row(**over) -> str:
    import json

    base = {
        "id": 1,
        "query": "q",
        "gold_topics": ["t"],
        "gold_experts": [1],
        "gold_route": "person",
        "difficulty": "L1",
    }
    base.update(over)
    return json.dumps([base])


def test_load_eval_queries_rejects_bad_route(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(_row(gold_route="preson"), encoding="utf-8")  # typo
    with pytest.raises(ValueError, match="gold_route"):
        load_eval_queries(bad)


def test_load_eval_queries_rejects_bad_difficulty(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(_row(difficulty="l1"), encoding="utf-8")  # wrong case
    with pytest.raises(ValueError, match="difficulty"):
        load_eval_queries(bad)


def test_load_eval_queries_rejects_non_string_query(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(_row(query=None), encoding="utf-8")  # null must not become "None"
    with pytest.raises(ValueError, match="expected a string"):
        load_eval_queries(bad)


def test_load_eval_queries_rejects_boolean_expert_id(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(_row(gold_experts=[True]), encoding="utf-8")  # bool must not become 1
    with pytest.raises(ValueError, match="expected an integer"):
        load_eval_queries(bad)


def test_load_eval_queries_rejects_non_list_field(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(_row(gold_topics="t"), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a list"):
        load_eval_queries(bad)


def test_load_eval_queries_rejects_non_bool_expect_abstain(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(_row(expect_abstain="yes"), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a boolean"):
        load_eval_queries(bad)


# --------------------------------------------------------------------------- #
# pipeline: prior_answer responder pinning (mirrors the production graph)
# --------------------------------------------------------------------------- #
def test_pinned_responder_picks_highest_scoring_past_answer() -> None:
    from tekijin.eval.pipeline import _pinned_responder

    retrieval = {
        "past_answers": [
            {"qa_id": "a", "score": 0.4, "responder_id": 7},
            {"qa_id": "b", "score": 0.9, "responder_id": 3},  # highest score
            {"qa_id": "c", "score": 0.8, "responder_id": None},
        ],
    }
    assert _pinned_responder(retrieval) == 3  # type: ignore[arg-type]
    assert _pinned_responder({"past_answers": []}) is None  # type: ignore[arg-type]
    # The SINGLE top-scored answer has no responder -> no pin (production then
    # falls back to the full pool); we must NOT fall through to a lower-scored one.
    top_none = {
        "past_answers": [
            {"qa_id": "a", "score": 0.9, "responder_id": None},  # top score, no responder
            {"qa_id": "b", "score": 0.4, "responder_id": 7},
        ]
    }
    assert _pinned_responder(top_none) is None  # type: ignore[arg-type]


def test_pipeline_ranker_pins_responder_on_prior_answer_route() -> None:
    from tekijin.eval.pipeline import PipelineRanker

    class _FakeRetriever:
        def search(self, query: str):  # noqa: ARG002
            # High answer confidence + a past answer -> decide_route == prior_answer.
            return {
                "past_answers": [{"qa_id": "a", "score": 0.95, "responder_id": 3}],
                "documents": [],
                "candidate_people": [1, 2, 3, 4],
                "answer_confidence": 0.99,
                "document_confidence": 0.0,
                "people_confidence": 0.2,
            }

    calls: list[list[int]] = []

    class _FakeScorer:
        def rank(self, topics, candidate_ids, asker_id, now, *, top_k=3, question_similarity=None):  # noqa: ARG002
            calls.append(list(candidate_ids))
            return {"recommendations": [{"person_id": pid} for pid in candidate_ids]}

    ranker = PipelineRanker(retriever=_FakeRetriever(), scorer=_FakeScorer(), now=NOW)
    result = ranker(_q(gold_route="prior_answer"))

    assert result.route == "prior_answer"
    # #307: the pinned past responder is scored alone first (guaranteed a slot),
    # then the remaining candidate pool (minus the pin) backfills the rest —
    # mirroring nodes.c6_score so the eval reflects what the product shows.
    assert calls == [[3], [1, 2, 4]]
    assert result.ranked_experts == [3, 1, 2, 4]


class _StaticRetriever:
    def __init__(self, payload):
        self._payload = payload

    def search(self, query: str):  # noqa: ARG002
        return self._payload


class _RecordingScorer:
    def __init__(self):
        self.called = False

    def rank(self, topics, candidate_ids, asker_id, now, *, top_k=3, question_similarity=None):  # noqa: ARG002
        self.called = True
        return {"recommendations": [{"person_id": pid} for pid in candidate_ids]}


def test_pipeline_ranker_presents_no_experts_on_document_route() -> None:
    from tekijin.eval.pipeline import PipelineRanker

    # High document confidence + weak people/answer -> decide_route == document.
    retriever = _StaticRetriever(
        {
            "past_answers": [],
            "documents": [{"doc_id": "d1", "score": 0.9}],
            "candidate_people": [1, 2, 3],
            "answer_confidence": 0.0,
            "document_confidence": 0.99,
            "people_confidence": 0.0,
        }
    )
    scorer = _RecordingScorer()
    result = PipelineRanker(retriever=retriever, scorer=scorer, now=NOW)(_q(gold_route="document"))

    assert result.route == "document"
    assert result.ranked_experts == []  # document is a terminal: no experts shown
    assert scorer.called is False  # C6 never runs on the document route


def test_predict_topics_from_retrieval_rank_weighted_vote() -> None:
    from tekijin.eval.pipeline import predict_topics_from_retrieval

    def _retrieval(*qa_ids):
        return {
            "past_answers": [
                {"qa_id": qa, "score": 1.0 - i * 0.1, "responder_id": i}
                for i, qa in enumerate(qa_ids)
            ],
            "documents": [],
        }

    topics = {"a1": ["ネットワーク"], "a2": ["セキュリティ"], "a3": ["セキュリティ"]}
    # Frequency accumulates: セキュリティ (ranks 1+2 -> 1/62+1/63) outweighs a lone
    # ネットワーク at rank 0 (1/61) under the default k=60.
    assert predict_topics_from_retrieval(_retrieval("a1", "a2", "a3"), topics) == [
        "セキュリティ",
        "ネットワーク",
    ]
    # But rank still matters: when ネットワーク is backed twice at the top and
    # セキュリティ only once below it, ネットワーク wins.
    topics2 = {"a1": ["ネットワーク"], "a2": ["ネットワーク"], "a3": ["セキュリティ"]}
    assert predict_topics_from_retrieval(_retrieval("a1", "a2", "a3"), topics2) == [
        "ネットワーク",
        "セキュリティ",
    ]


def test_predict_topics_from_retrieval_empty_without_answers() -> None:
    from tekijin.eval.pipeline import predict_topics_from_retrieval

    # Documents carry no topic, so a document-only retrieval predicts nothing.
    retrieval = {"past_answers": [], "documents": [{"doc_id": "d1", "score": 0.9}]}
    assert predict_topics_from_retrieval(retrieval, {"a1": ["x"]}) == []
    # An answer with no known topic mapping contributes nothing either.
    retrieval2 = {"past_answers": [{"qa_id": "unknown", "score": 0.9, "responder_id": 1}]}
    assert predict_topics_from_retrieval(retrieval2, {"a1": ["x"]}) == []


def test_predict_topics_from_retrieval_caps_voting_input() -> None:
    from tekijin.eval.pipeline import predict_topics_from_retrieval

    # 25 answers all tagged "TAIL", plus one "HEAD" answer sitting just past the
    # vote_depth cut. With vote_depth=2 only the first two (both TAIL) vote, so the
    # HEAD answer beyond the cut contributes nothing — proving the cap bounds the
    # voting INPUT (the reference semantics), not just the output list.
    past = [{"qa_id": f"t{i}", "score": 1.0 - i * 0.01, "responder_id": i} for i in range(25)]
    past.append({"qa_id": "head", "score": 0.0, "responder_id": 99})
    retrieval = {"past_answers": past, "documents": []}
    topics = {f"t{i}": ["TAIL"] for i in range(25)} | {"head": ["HEAD"]}
    assert predict_topics_from_retrieval(retrieval, topics, vote_depth=2) == ["TAIL"]
    # Without a cut the head answer is reached and its topic appears too.
    assert set(predict_topics_from_retrieval(retrieval, topics, vote_depth=100)) == {"TAIL", "HEAD"}


def test_pipeline_ranker_populates_predicted_topics() -> None:
    from tekijin.eval.pipeline import PipelineRanker

    retriever = _StaticRetriever(
        {
            "past_answers": [{"qa_id": "a1", "score": 0.95, "responder_id": 3}],
            "documents": [],
            "candidate_people": [3],
            "answer_confidence": 0.99,
            "document_confidence": 0.0,
            "people_confidence": 0.1,
        }
    )
    ranker = PipelineRanker(
        retriever=retriever,
        scorer=_RecordingScorer(),
        now=NOW,
        answer_topics={"a1": ["ネットワーク・VPN"]},
    )
    result = ranker(_q(gold_route="prior_answer", gold_topics=["ネットワーク・VPN"]))
    assert result.predicted_topics == ["ネットワーク・VPN"]


def _answer_dto(**over):
    from tekijin.data.dto import AnswerDTO

    base = {
        "id": "ans1",
        "question_id": "q1",
        "responder_id": 1,
        "body": "b",
        "topic": None,
        "reuse_count": 0,
        "was_helpful": None,
        "created_at": None,
        "has_embedding": False,
    }
    base.update(over)
    return AnswerDTO(**base)


def _question_dto(**over):
    from tekijin.data.dto import QuestionDTO

    base = {
        "id": "q1",
        "asker_id": 1,
        "body": "b",
        "topics": (),
        "status": None,
        "created_at": None,
        "has_embedding": False,
    }
    base.update(over)
    return QuestionDTO(**base)


class _FakeTopicRepo:
    def __init__(self, answers, questions):
        self._answers = answers
        self._questions = questions

    def list_answers(self):
        return self._answers

    def list_questions(self):
        return self._questions


def test_build_answer_topics_prefers_answer_topic_then_question() -> None:
    from tekijin.eval.pipeline import build_answer_topics

    repo = _FakeTopicRepo(
        answers=[
            # own topic wins outright
            _answer_dto(id="a_own", topic="セキュリティ", question_id="q_multi"),
            # NULL topic -> falls back to the linked question's topics array
            _answer_dto(id="a_fallback", topic=None, question_id="q_multi"),
            # NULL topic AND no matching question -> empty list, not a crash
            _answer_dto(id="a_orphan", topic=None, question_id="q_missing"),
        ],
        questions=[_question_dto(id="q_multi", topics=("ネットワーク・VPN", "クラウド移行"))],
    )
    mapping = build_answer_topics(repo)  # type: ignore[arg-type]
    assert mapping["a_own"] == ["セキュリティ"]
    assert mapping["a_fallback"] == ["ネットワーク・VPN", "クラウド移行"]
    assert mapping["a_orphan"] == []


def test_build_answer_topics_fallback_agrees_with_repository(session, seed_counts) -> None:
    """The NULL-topic fallback must resolve the SAME topics ``answers_by_topics`` uses.

    Insert a runtime-style answer (``topic`` NULL) on a question that has a topics
    array, then assert build_answer_topics maps it to that array AND that
    ``Repository.answers_by_topics`` treats the answer as evidence for each — locking
    the two code paths together (they both implement the NULL-topic fallback rule).
    """

    from tekijin.data.repository import Repository
    from tekijin.eval.pipeline import build_answer_topics
    from tekijin.models.tables import Answer, Question

    session.add(
        Question(id="q_rt", asker_id=1, body="拠点間のVPNが不安定", topics=["ネットワーク・VPN"])
    )
    session.flush()  # persist the question before its answer references it (FK order)
    session.add(
        Answer(id="a_rt", question_id="q_rt", responder_id=1, body="MTUを見直す", topic=None)
    )
    session.flush()

    repo = Repository(session)
    mapping = build_answer_topics(repo)
    assert mapping["a_rt"] == ["ネットワーク・VPN"]
    # The repository's topic-evidence query agrees: the NULL-topic answer counts
    # for the question's topic.
    assert any(a.id == "a_rt" for a in repo.answers_by_topics(["ネットワーク・VPN"]))


def test_pipeline_ranker_returns_no_experts_when_topics_empty() -> None:
    from tekijin.eval.pipeline import PipelineRanker

    retriever = _StaticRetriever(
        {
            "past_answers": [],
            "documents": [],
            "candidate_people": [1, 2, 3],
            "answer_confidence": 0.0,
            "document_confidence": 0.0,
            "people_confidence": 0.1,
        }
    )
    scorer = _RecordingScorer()
    # Unsupported-topic row: gold_topics empty -> mirror c6_score returning nothing.
    result = PipelineRanker(retriever=retriever, scorer=scorer, now=NOW)(
        _q(gold_topics=[], gold_route="person")
    )

    assert result.route == "person"
    assert result.ranked_experts == []
    assert scorer.called is False


# --------------------------------------------------------------------------- #
# runner (stub ranker — deterministic orchestration)
# --------------------------------------------------------------------------- #
def test_run_eval_with_perfect_stub() -> None:
    queries = [
        _q(id=1, gold_experts=[3, 1], gold_route="person"),
        _q(id=2, gold_experts=[5], gold_route="document"),
    ]

    def perfect(query: EvalQuery) -> RankResult:
        return RankResult(ranked_experts=list(query.gold_experts), route=query.gold_route)

    report = run_eval(queries, perfect)
    assert report.metrics.top1_accuracy == pytest.approx(1.0)
    assert report.metrics.recall_at_3 == pytest.approx(1.0)
    assert report.metrics.mrr == pytest.approx(1.0)
    assert report.metrics.route_accuracy == pytest.approx(1.0)
    assert len(report.results) == 2


def test_run_eval_with_empty_stub_scores_zero() -> None:
    queries = [_q(id=1, gold_experts=[3], gold_route="person")]
    report = run_eval(queries, lambda _q: RankResult(ranked_experts=[], route="document"))
    assert report.metrics.top1_accuracy == 0.0
    assert report.metrics.recall_at_3 == 0.0
    assert report.metrics.mrr == 0.0
    assert report.metrics.route_accuracy == 0.0


def test_run_eval_carries_topics_and_scores_stage_a() -> None:
    queries = [
        _q(id=1, gold_topics=["A"], gold_experts=[1]),
        _q(id=2, gold_topics=["B"], gold_experts=[2]),
    ]

    def ranker(query: EvalQuery) -> RankResult:
        # Query 1's top predicted topic is correct; query 2's is not (but B is #2).
        predicted = ["A"] if query.id == 1 else ["X", "B"]
        return RankResult(list(query.gold_experts), query.gold_route, predicted_topics=predicted)

    report = run_eval(queries, ranker)
    # Per-query records keep the topics for drill-down.
    assert report.results[0].predicted_topics == ["A"]
    assert report.results[0].gold_topics == ["A"]
    # Stage-A aggregate: 1/2 acc@1 (only query 1), 2/2 acc@3 (B is within top 3).
    assert report.topic_accuracy.n_topic == 2
    assert report.topic_accuracy.acc_at_1 == pytest.approx(0.5)
    assert report.topic_accuracy.acc_at_3 == pytest.approx(1.0)


def test_run_eval_carries_source_recall_and_decisions() -> None:
    queries = [
        _q(id=1, gold_route="document", gold_source=["doc_1", "doc_2"]),
        _q(id=2, gold_route="person", gold_source=[], gold_experts=[7]),
    ]

    def ranker(query: EvalQuery) -> RankResult:
        if query.id == 1:
            # self-answers, citing one of the two gold sources
            return RankResult([], "self_answered", cited_source_ids=["doc_1"])
        return RankResult([7], "person")

    report = run_eval(queries, ranker)
    # per-query records keep the citation inputs for drill-down
    assert report.results[0].cited_source_ids == ["doc_1"]
    assert report.results[0].gold_source == ["doc_1", "doc_2"]
    # C7' source recall over the single obligated row: 1 of 2 gold cited
    assert report.source_recall.n == 1
    assert report.source_recall.recall == pytest.approx(0.5)
    assert report.source_recall.grounded_rate == pytest.approx(1.0)
    # C5 decision recall: self_answered→self_answer hit, person→route hit
    assert report.decision_recall.per_class["self_answer"].recall == pytest.approx(1.0)
    assert report.decision_recall.per_class["route"].recall == pytest.approx(1.0)


def test_format_report_contains_recall_centric_blocks() -> None:
    queries = [
        _q(id=1, gold_route="document", gold_source=["doc_1"]),
        _q(id=2, gold_route="person", gold_experts=[2]),
    ]
    report = run_eval(
        queries,
        lambda q: RankResult(
            list(q.gold_experts),
            "self_answered" if q.id == 1 else "person",
            cited_source_ids=["doc_1"] if q.id == 1 else [],
        ),
    )
    text = format_report(report)
    assert "C5 振り分け recall" in text
    assert "自己回答" in text and "取次ぎ" in text
    assert "C7' 出典 recall" in text
    assert "precision" in text and "grounded率" in text


def test_format_report_contains_all_metrics_and_breakdowns() -> None:
    queries = [
        _q(id=1, difficulty="L1", gold_experts=[1], gold_experts_alt=[1]),
        _q(id=2, difficulty="L2", gold_experts=[2], gold_experts_alt=[9]),
    ]
    report = run_eval(
        queries,
        lambda q: RankResult(list(q.gold_experts), "person", predicted_topics=list(q.gold_topics)),
    )
    text = format_report(report)
    for label in ("Top-1 Accuracy", "Hit@3", "Recall@3", "MRR", "Route Accuracy"):
        assert label in text
    assert "層別" in text and "L1" in text and "L2" in text  # per-layer breakdown
    assert "第2正解" in text  # anti-circularity (alt) line
    assert "段A" in text and "acc@1" in text and "acc@3" in text  # stage-A topic block
    # report structure
    assert set(report.by_difficulty) == {"L1", "L2"}
    assert report.metrics_alt.n_ranked == 2
    assert report.topic_accuracy.n_topic == 2  # both rows carry gold_topics (["t"])


# --------------------------------------------------------------------------- #
# integration: the real pipeline over the seed (regression floor)
# --------------------------------------------------------------------------- #
def test_run_eval_real_pipeline_over_seed(seed_counts, session, fake_embedder) -> None:
    from tekijin.eval.pipeline import build_pipeline_ranker

    queries = load_eval_queries()
    ranker = build_pipeline_ranker(session, fake_embedder, now=NOW)
    report = run_eval(queries, ranker)

    m = report.metrics
    assert m.n == 87  # ran over every query (#296: 型番6件を追加)
    assert m.n_ranked > 0 and m.n_routed > 0
    for value in (m.top1_accuracy, m.recall_at_3, m.mrr, m.route_accuracy):
        assert 0.0 <= value <= 1.0
    # Layer-wise breakdown + anti-circularity run are produced.
    assert {"L1", "L2", "L3"} <= set(report.by_difficulty)
    assert report.metrics_alt.n_ranked > 0
    # Stage-A topic hit-rate is produced (#71): scored over the queries carrying
    # gold topics, with valid fractions. Documents carry no topic so predictions
    # come from retrieved past answers over the seed. The abstain (L4) rows have
    # empty gold_topics, so the denominator sits strictly between 0 and n.
    ta = report.topic_accuracy
    assert 0 < ta.n_topic <= m.n - m.n_abstain
    assert 0.0 <= ta.acc_at_1 <= ta.acc_at_3 <= 1.0
    # Abstain layer is measured (15 L4 rows) and the rate is a valid fraction.
    assert m.n_abstain == 15
    assert 0.0 <= m.abstain_accuracy <= 1.0
    # Sanity: the alt (answers-derived) Recall@3 is not a perfect echo of the
    # primary — if it were 1.0 the scorer would just be reproducing labels.
    assert report.metrics_alt.recall_at_3 < 0.99
    # Regression floors with headroom below the deterministic observed values on
    # the NON-leaky set. These catch a real retrieval/scoring regression without
    # pinning the exact numbers (which move when the scorer weights are retuned).
    # Absolute spec targets (§7) need real embeddings on the seed rows; here the
    # fixtures store NULL vectors so BM25 drives retrieval and routing degenerates
    # to the person line (route accuracy == the person fraction among routed queries).
    #
    # #158 で評価セットを2段階に変えたので、その都度実測して床を引き直している。
    #
    #     指標        #158前   制約15件   ＋L3 20件
    #     Top-1      0.643    0.536      0.606
    #     Recall@3   0.545    0.551      0.568
    #     MRR        0.670    0.598      0.659
    #     route      0.696    0.696      0.742
    #
    # 制約を 15 件に増やした段では 10 件の gold が 4 名から 1〜2 名に絞られ
    # （gold 人数の合計 40 → 19）、Top-1 と MRR が下がった。**検索やスコアラーの
    # 劣化ではない**（該当クエリでランダムに 1 名選んだときの Top-1 期待値も
    # 0.100 → 0.048 に半減している）。その後 L3 を 10 件足したところ、
    # 新しい L3 は gold が 3〜4 名で当てやすいぶん、いずれも戻った。
    #
    # 床は現在の実測から余裕を取って置く。**下げるのは実測が下がったときだけにし、
    # 理由をここに書く**（一度、下がっていない Recall@3 の床まで下げてしまった）。
    #
    # #95 で資格・スキル・案件をスコアラーの根拠から外して、実測が下がった。
    #
    #     指標        #95前    #95後
    #     Top-1      0.606    0.470
    #     Recall@3   0.563    0.455
    #     MRR        0.659    0.563
    #     route      0.681    0.681
    #
    # 主 gold は `projects` から作るので、案件を根拠に使わなくなれば当たりにくくなる。
    # 回答から作る第2の gold では Top-1 0.721→0.738、Recall@3 0.492→0.503 と少し上がった。
    # スコアラーの劣化ではなく、会社データの評価セットが案件に寄っているぶんが抜けた。
    assert m.top1_accuracy >= 0.42
    assert m.recall_at_3 >= 0.40
    assert m.mrr >= 0.50
    assert m.route_accuracy >= 0.65

    # #297 recall-centric metrics are produced over the real seed. The LLM-free
    # pipeline ranker never self-answers, so C7' source recall is 0 over the 23
    # citation-obligated rows (document 16 + prior_answer 7) — this asserts the
    # obligation denominator is wired, not that the (absent) self-answer works.
    sr = report.source_recall
    assert sr.n == 23  # document 16 (#296: 型番6件追加) + prior_answer 7 carry gold_source
    assert sr.recall == 0.0 and sr.n_cited == 0 and sr.grounded_rate == 0.0
    # C5 decision recall covers all three product decisions on this set.
    dr = report.decision_recall
    assert set(dr.per_class) == {"self_answer", "route", "abstain"}
    assert dr.n == m.n  # every VALID_ROUTES row maps to a decision
    for cr in dr.per_class.values():
        assert 0.0 <= cr.recall <= 1.0

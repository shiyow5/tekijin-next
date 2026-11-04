"""Integration tests for the C6 scorer against live PostgreSQL + pgvector.

Uses the shared ``engine`` / ``seed_counts`` / ``session`` fixtures (conftest):
CI's pgvector service via ``TEKIJIN_DATABASE_URL`` or an ephemeral ``pgserver``
locally, skipped when neither is available.

Evidence is isolated with a SYNTHETIC topic that matches nothing in the seed
(no answer or retrospective maps to it), so every candidate starts from
zero and the only evidence is what each test inserts. ``now`` is injected, so the
7-day load window and recency are fully deterministic. Inserted rows are flushed
(visible in-session) but never committed — each test rolls back on close.
"""

from __future__ import annotations

import datetime as dt

from tekijin.data.repository import Repository
from tekijin.models.tables import Answer, Question, Recommendation
from tekijin.scorer import ExpertiseScorer

# Well past the newest seed answer (2026-08-21), so no seeded activity falls in
# the 7-day load window: load is then driven purely by rows each test inserts.
NOW = dt.datetime(2026, 9, 15, 12, 0, 0)
TOPIC = "__SCORER_TEST_TOPIC__"  # matches nothing in the seed


# --------------------------------------------------------------------------- #
# insert helpers (flushed, not committed)
# --------------------------------------------------------------------------- #
def _add_question(session, qid: str, asker_id: int = 1) -> None:
    session.add(Question(id=qid, asker_id=asker_id, body="q", topics=[TOPIC], status="open"))
    session.flush()


def _add_answer(
    session,
    aid: str,
    qid: str,
    responder_id: int,
    *,
    helpful: bool,
    reuse: int,
    created: dt.datetime,
) -> None:
    session.add(
        Answer(
            id=aid,
            question_id=qid,
            responder_id=responder_id,
            body="a",
            topic=TOPIC,
            reuse_count=reuse,
            was_helpful=helpful,
            created_at=created,
        )
    )
    session.flush()


def _add_evidence(session, eid: str, employee_id: int, topic: str = TOPIC) -> None:
    """One helpful on-topic answer dated outside the 7-day load window.

    It gives the candidate topic evidence without adding to their load, so tests
    that compare load or other terms start from identical evidence.
    """

    session.add(Question(id=f"q_{eid}", asker_id=1, body="q", topics=[topic], status="open"))
    session.flush()
    session.add(
        Answer(
            id=eid,
            question_id=f"q_{eid}",
            responder_id=employee_id,
            body="a",
            topic=topic,
            reuse_count=0,
            was_helpful=True,
            created_at=NOW - dt.timedelta(days=30),
        )
    )
    session.flush()


def _add_recommendation(
    session, employee_id: int, *, created: dt.datetime, outcome: str | None, qid: str
) -> None:
    session.add(
        Recommendation(
            question_id=qid,
            employee_id=employee_id,
            rank=1,
            score=0.5,
            outcome=outcome,
            created_at=created,
        )
    )
    session.flush()


def _scores_by_person(result: dict) -> dict[int, float]:
    return {r["person_id"]: r["score"] for r in result["recommendations"]}


# --------------------------------------------------------------------------- #
# output shape / top_k
# --------------------------------------------------------------------------- #
def test_rank_output_shape_and_top_k(seed_counts, session) -> None:
    _add_question(session, "q_shape")
    # Five candidates with differing strengths (one old answer + varying fresh ones).
    for i, emp in enumerate([1, 2, 3, 4, 5]):
        _add_evidence(session, f"ev_shape_{emp}", emp)
        for j in range(i):  # emp 1 gets 0, emp 5 gets 4 helpful answers
            _add_answer(
                session, f"a_shape_{emp}_{j}", "q_shape", emp, helpful=True, reuse=1, created=NOW
            )

    scorer = ExpertiseScorer(Repository(session))
    result = scorer.rank(TOPIC, [1, 2, 3, 4, 5], asker_id=None, now=NOW, top_k=3)

    recs = result["recommendations"]
    assert set(result) == {"recommendations"}
    assert len(recs) == 3
    for rec in recs:
        assert set(rec) == {"person_id", "name", "dept", "score", "confidence", "reasons"}
        assert isinstance(rec["person_id"], int)
        assert isinstance(rec["score"], float)
        assert rec["confidence"] in {"高", "低"}
        assert rec["reasons"] and all(set(r) == {"type", "detail"} for r in rec["reasons"])
    # Sorted by descending score.
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
    # The strongest candidate (emp 5, most helpful answers) leads.
    assert recs[0]["person_id"] == 5


# --------------------------------------------------------------------------- #
# expert ranks above a bare candidate
# --------------------------------------------------------------------------- #
def test_expert_outranks_thin_candidate(seed_counts, session) -> None:
    _add_question(session, "q_exp")
    # Expert: one old answer + three fresh helpful, reused answers.
    _add_evidence(session, "ev_exp", 1)
    for j in range(3):
        _add_answer(session, f"a_exp_{j}", "q_exp", 1, helpful=True, reuse=2, created=NOW)
    # Thin candidate: only the one old answer.
    _add_evidence(session, "ev_thin", 2)

    scorer = ExpertiseScorer(Repository(session))
    result = scorer.rank(TOPIC, [2, 1], asker_id=None, now=NOW, top_k=3)

    assert result["recommendations"][0]["person_id"] == 1
    scores = _scores_by_person(result)
    assert scores[1] > scores[2]
    # The expert's reasons mention the answers evidence.
    expert = result["recommendations"][0]
    assert any(r["type"] == "answers" for r in expert["reasons"])
    assert expert["confidence"] == "高"


# --------------------------------------------------------------------------- #
# load penalty
# --------------------------------------------------------------------------- #
def test_high_load_lowers_score(seed_counts, session) -> None:
    _add_question(session, "q_load")
    # Identical evidence for both candidates.
    for emp in (1, 2):
        _add_evidence(session, f"ev_load_{emp}", emp)
        _add_answer(session, f"a_load_{emp}", "q_load", emp, helpful=True, reuse=1, created=NOW)
    # emp 2 is busy this week (8 recent recommendations).
    for _ in range(8):
        _add_recommendation(
            session, 2, created=NOW - dt.timedelta(days=1), outcome="accepted", qid="q_load"
        )

    scorer = ExpertiseScorer(Repository(session))
    result = scorer.rank(TOPIC, [1, 2], asker_id=None, now=NOW, top_k=3)

    scores = _scores_by_person(result)
    assert scores[1] > scores[2]  # load drags emp 2 below the identical emp 1
    assert result["recommendations"][0]["person_id"] == 1


def test_declined_recommendations_count_as_load(seed_counts, session) -> None:
    # Spec (db-schema.md / model-definition C8 / technical-spec §5): a decline
    # lowers only availability (余裕度), so it DOES count toward the 7-day load
    # window — it just never counts as expertise evidence. So emp 1 (8 recent
    # declines) must rank BELOW emp 2 (identical evidence, no load).
    _add_question(session, "q_decl")
    for emp in (1, 2):
        _add_evidence(session, f"ev_decl_{emp}", emp)
    for _ in range(8):
        _add_recommendation(
            session, 1, created=NOW - dt.timedelta(days=1), outcome="declined", qid="q_decl"
        )

    scorer = ExpertiseScorer(Repository(session))
    result = scorer.rank(TOPIC, [1, 2], asker_id=None, now=NOW, top_k=3)

    scores = _scores_by_person(result)
    assert scores[1] < scores[2]  # declined load drags emp 1 down
    assert result["recommendations"][0]["person_id"] == 2


def test_a_decline_never_costs_expertise_only_availability(seed_counts, session) -> None:
    """断り≠非専門, locked from the other side (#261).

    ``test_declined_recommendations_count_as_load`` fixes that a decline DOES cost
    availability. Nothing fixed the half that actually protects people: that it
    costs no EXPERTISE. Hold the load window identical and vary only the outcome —
    eight declines against eight accepts — and the scores must come out equal.

    Without this, a future "declines are a negative signal" change would pass the
    whole suite: the existing test only asserts a decline is not FREE.
    """

    _add_question(session, "q_decl_sym")
    for emp in (1, 2):
        _add_evidence(session, f"ev_decl_sym_{emp}", emp)
    # Same count, same window, same everything — only `outcome` differs.
    for emp, outcome in ((1, "declined"), (2, "accepted")):
        for _ in range(8):
            _add_recommendation(
                session,
                emp,
                created=NOW - dt.timedelta(days=1),
                outcome=outcome,
                qid="q_decl_sym",
            )

    scorer = ExpertiseScorer(Repository(session))
    scores = _scores_by_person(scorer.rank(TOPIC, [1, 2], asker_id=None, now=NOW, top_k=3))

    assert scores[1] == scores[2], (
        "a decline cost expertise: declining is a statement about availability, "
        "not about what the person knows"
    )


def test_declining_does_not_erase_the_answers_a_person_already_gave(seed_counts, session) -> None:
    """The evidence a decline must not touch is the strongest kind there is.

    Structurally this cannot regress by accident — ``collect_topic_evidence`` takes
    no recommendations at all — but that IS the guarantee, so it is worth pinning:
    an expert who declines eight times still outranks a novice with none.
    """

    _add_question(session, "q_decl_ans")
    _add_answer(
        session,
        "a_decl_ans",
        "q_decl_ans",
        1,
        helpful=True,
        reuse=3,
        created=NOW - dt.timedelta(days=30),  # outside the 7-day load window
    )
    for _ in range(8):
        _add_recommendation(
            session, 1, created=NOW - dt.timedelta(days=1), outcome="declined", qid="q_decl_ans"
        )

    scorer = ExpertiseScorer(Repository(session))
    result = scorer.rank(TOPIC, [1, 2], asker_id=None, now=NOW, top_k=3)
    scores = _scores_by_person(result)

    assert scores[1] > scores[2], "declines outweighed a helpful answer"
    expert = next(r for r in result["recommendations"] if r["person_id"] == 1)
    assert any(r["type"] == "answers" for r in expert["reasons"])


def test_old_recommendations_outside_window_ignored(seed_counts, session) -> None:
    _add_question(session, "q_old")
    for emp in (1, 2):
        _add_evidence(session, f"ev_old_{emp}", emp)
    # emp 2's recs are 10 days old — outside the 7-day load window.
    for _ in range(8):
        _add_recommendation(
            session, 2, created=NOW - dt.timedelta(days=10), outcome="accepted", qid="q_old"
        )

    scorer = ExpertiseScorer(Repository(session))
    scores = _scores_by_person(scorer.rank(TOPIC, [1, 2], asker_id=None, now=NOW, top_k=3))
    assert scores[1] == scores[2]  # old load ignored -> tie


# --------------------------------------------------------------------------- #
# candidate scoping / unknown ids
# --------------------------------------------------------------------------- #
def test_only_input_candidates_are_scored(seed_counts, session) -> None:
    _add_question(session, "q_scope")
    for emp in (1, 2):
        _add_evidence(session, f"ev_scope_{emp}", emp)

    scorer = ExpertiseScorer(Repository(session))
    result = scorer.rank(TOPIC, [1, 2, 999999], asker_id=None, now=NOW, top_k=3)

    person_ids = {r["person_id"] for r in result["recommendations"]}
    assert person_ids <= {1, 2}  # unknown id 999999 dropped; nobody outside input
    assert 999999 not in person_ids


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_rank_is_deterministic(seed_counts, session) -> None:
    _add_question(session, "q_det")
    for emp in (1, 2, 3):
        _add_evidence(session, f"ev_det_{emp}", emp)
        _add_answer(session, f"a_det_{emp}", "q_det", emp, helpful=True, reuse=1, created=NOW)

    scorer = ExpertiseScorer(Repository(session))
    first = scorer.rank(TOPIC, [1, 2, 3], asker_id=None, now=NOW, top_k=3)
    second = scorer.rank(TOPIC, [1, 2, 3], asker_id=None, now=NOW, top_k=3)
    assert first == second


# --------------------------------------------------------------------------- #
# fix 3: the asker is never recommended to themselves
# --------------------------------------------------------------------------- #
def test_asker_excluded_from_candidates(seed_counts, session) -> None:
    for emp in (1, 2):
        _add_evidence(session, f"ev_asker_{emp}", emp)

    scorer = ExpertiseScorer(Repository(session))
    result = scorer.rank(TOPIC, [1, 2], asker_id=1, now=NOW, top_k=3)

    person_ids = {r["person_id"] for r in result["recommendations"]}
    assert person_ids == {2}  # asker (1) dropped, only 2 remains


def test_asker_only_candidate_yields_empty(seed_counts, session) -> None:
    _add_evidence(session, "ev_solo", 5)
    scorer = ExpertiseScorer(Repository(session))
    result = scorer.rank(TOPIC, [5], asker_id=5, now=NOW, top_k=3)
    assert result["recommendations"] == []


# --------------------------------------------------------------------------- #
# #405: question↔past-answer similarity (qsim) re-ranks and is dormant when None
# --------------------------------------------------------------------------- #
def test_question_similarity_reranks_and_adds_reason(seed_counts, session) -> None:
    """A candidate whose past answers match the QUESTION text is lifted, even when
    both candidates carry identical topic evidence (saturated topic_fit ties). The
    boost is ``weights.question_fit * qsim`` and is surfaced as a reason."""
    from tekijin.scorer.weights import DEFAULT_WEIGHTS

    for emp in (1, 2):
        _add_evidence(session, f"ev_qsim_{emp}", emp)  # identical topic evidence

    scorer = ExpertiseScorer(Repository(session))
    # Baseline: identical evidence -> tie broken by ascending person_id (1 leads).
    base = _scores_by_person(scorer.rank(TOPIC, [1, 2], asker_id=None, now=NOW, top_k=3))
    assert base[1] == base[2]

    # qsim favours #2: it must now outrank #1 and gain a question_fit reason.
    res = scorer.rank(
        TOPIC, [1, 2], asker_id=None, now=NOW, top_k=3, question_similarity={2: 0.8, 1: 0.0}
    )
    scores = _scores_by_person(res)
    assert scores[2] > scores[1]
    assert scores[2] == round(base[2] + DEFAULT_WEIGHTS.question_fit * 0.8, 4)
    assert scores[1] == base[1]  # qsim 0.0 -> unchanged
    two = next(r for r in res["recommendations"] if r["person_id"] == 2)
    qfit = [r for r in two["reasons"] if r["type"] == "question_fit"]
    assert qfit  # the boost is explained, not silent


def test_question_similarity_below_floor_scores_without_reason(seed_counts, session) -> None:
    """A noise-level qsim still nudges the score (continuous signal, as measured)
    but must NOT show the assertive "一致" reason — the reason is floored (#405)."""
    from tekijin.scorer.weights import DEFAULT_WEIGHTS, QUESTION_FIT_REASON_FLOOR

    _add_evidence(session, "ev_qsim_floor", 4)
    scorer = ExpertiseScorer(Repository(session))
    base = _scores_by_person(scorer.rank(TOPIC, [4], asker_id=None, now=NOW, top_k=3))

    tiny = QUESTION_FIT_REASON_FLOOR / 2  # below the reason floor
    res = scorer.rank(TOPIC, [4], asker_id=None, now=NOW, top_k=3, question_similarity={4: tiny})
    rec = res["recommendations"][0]
    # Score is nudged by the continuous term...
    assert rec["score"] == round(base[4] + DEFAULT_WEIGHTS.question_fit * tiny, 4)
    # ...but the misleading "matches" reason is suppressed below the floor.
    assert all(r["type"] != "question_fit" for r in rec["reasons"])


def test_question_similarity_absent_is_byte_identical(seed_counts, session) -> None:
    """Omitting ``question_similarity`` keeps develop behaviour byte-identical:
    no score change, no question_fit reason (dormant by default)."""
    _add_evidence(session, "ev_qsim_off", 3)
    scorer = ExpertiseScorer(Repository(session))

    without = scorer.rank(TOPIC, [3], asker_id=None, now=NOW, top_k=3)
    empty = scorer.rank(TOPIC, [3], asker_id=None, now=NOW, top_k=3, question_similarity=None)

    assert _scores_by_person(without) == _scores_by_person(empty)
    rec = without["recommendations"][0]
    assert all(r["type"] != "question_fit" for r in rec["reasons"])


# --------------------------------------------------------------------------- #
# fix 5: an answer tagged for a DIFFERENT subtopic is not counted
# --------------------------------------------------------------------------- #
def test_answer_tagged_other_subtopic_is_not_evidence(seed_counts, session) -> None:
    topic_a = "__SCORER_TOPIC_A__"
    topic_b = "__SCORER_TOPIC_B__"
    # A multi-topic question (topics include B), but the answer is tagged for A.
    session.add(
        Question(id="q_multi", asker_id=1, body="q", topics=[topic_a, topic_b], status="open")
    )
    session.flush()
    # emp 3: an answer whose own topic is A (must NOT count for B).
    # Dated outside the 7-day load window so its (topic-independent) load does not
    # confound the comparison — this test is purely about evidence filtering.
    session.add(
        Answer(
            id="a_tagged_a",
            question_id="q_multi",
            responder_id=3,
            body="a",
            topic=topic_a,
            reuse_count=5,
            was_helpful=True,
            created_at=NOW - dt.timedelta(days=30),
        )
    )
    # emp 4: control — no evidence at all.
    session.flush()

    scorer = ExpertiseScorer(Repository(session))
    result = scorer.rank(topic_b, [3, 4], asker_id=None, now=NOW, top_k=3)
    scores = _scores_by_person(result)

    # The A-tagged answer is ignored for B, so emp 3 == emp 4 (no evidence).
    assert scores[3] == scores[4]
    emp3 = next(r for r in result["recommendations"] if r["person_id"] == 3)
    assert not any(r["type"] == "answers" for r in emp3["reasons"])


def test_null_topic_answer_uses_question_topics(seed_counts, session) -> None:
    # An answer with NO own topic DOES count via the question's topics (fallback).
    topic_b = "__SCORER_TOPIC_B2__"
    session.add(Question(id="q_null", asker_id=1, body="q", topics=[topic_b], status="open"))
    session.flush()
    session.add(
        Answer(
            id="a_null_topic",
            question_id="q_null",
            responder_id=3,
            body="a",
            topic=None,  # falls back to the question's topics
            reuse_count=1,
            was_helpful=True,
            created_at=NOW,
        )
    )
    # emp 4: control — no evidence.
    session.flush()

    scorer = ExpertiseScorer(Repository(session))
    result = scorer.rank(topic_b, [3, 4], asker_id=None, now=NOW, top_k=3)
    scores = _scores_by_person(result)
    assert scores[3] > scores[4]  # the null-topic answer counts for emp 3


# --------------------------------------------------------------------------- #
# fix 10: empty candidates and the 7-day load-window boundary
# --------------------------------------------------------------------------- #
def test_empty_candidates_yield_empty(seed_counts, session) -> None:
    scorer = ExpertiseScorer(Repository(session))
    assert scorer.rank(TOPIC, [], asker_id=None, now=NOW, top_k=3) == {"recommendations": []}


def test_load_window_boundary_is_inclusive(seed_counts, session) -> None:
    from tekijin.scorer.weights import LOAD_WINDOW_DAYS

    for emp in (1, 2):
        _add_evidence(session, f"ev_bnd_{emp}", emp)
    _add_question(session, "q_bnd")
    # emp 1: a rec EXACTLY at the window edge (now - 7d) -> counted (>= since).
    _add_recommendation(
        session,
        1,
        created=NOW - dt.timedelta(days=LOAD_WINDOW_DAYS),
        outcome="accepted",
        qid="q_bnd",
    )
    # emp 2: a rec one second before the edge -> outside the window.
    _add_recommendation(
        session,
        2,
        created=NOW - dt.timedelta(days=LOAD_WINDOW_DAYS, seconds=1),
        outcome="accepted",
        qid="q_bnd",
    )

    scorer = ExpertiseScorer(Repository(session))
    scores = _scores_by_person(scorer.rank(TOPIC, [1, 2], asker_id=None, now=NOW, top_k=3))
    # Only emp 1's boundary rec is inside the window, so emp 1 carries the load.
    assert scores[1] < scores[2]


# --------------------------------------------------------------------------- #
# fix 1: rows created AFTER now (upper bound) are not counted — offline eval
# --------------------------------------------------------------------------- #
def test_future_recommendations_not_counted_as_load(seed_counts, session) -> None:
    _add_question(session, "q_future")
    for emp in (1, 2):
        _add_evidence(session, f"ev_fut_{emp}", emp)
    # emp 1 has 8 recs dated AFTER now — a replayed historical ``now`` must not
    # let them into the [now-7d, now] window.
    for _ in range(8):
        _add_recommendation(
            session, 1, created=NOW + dt.timedelta(days=1), outcome="accepted", qid="q_future"
        )

    scorer = ExpertiseScorer(Repository(session))
    scores = _scores_by_person(scorer.rank(TOPIC, [1, 2], asker_id=None, now=NOW, top_k=3))
    assert scores[1] == scores[2]  # future rows excluded by the upper bound


# --------------------------------------------------------------------------- #
# fix C: multiple topics are aggregated (not just topics[0])
# --------------------------------------------------------------------------- #
def test_multi_topic_aggregates_expertise(seed_counts, session) -> None:
    topic_a = "__MT_TOPIC_A__"
    topic_b = "__MT_TOPIC_B__"
    # emp 3: an answer on A only. emp 4: answers on A AND B.
    _add_evidence(session, "mt_a3", 3, topic=topic_a)
    _add_evidence(session, "mt_a4", 4, topic=topic_a)
    _add_evidence(session, "mt_b4", 4, topic=topic_b)

    scorer = ExpertiseScorer(Repository(session))

    # Both topics: emp 4's extra B-answer lifts it above emp 3.
    both = scorer.rank([topic_a, topic_b], [3, 4], asker_id=None, now=NOW, top_k=3)
    both_scores = _scores_by_person(both)
    assert both["recommendations"][0]["person_id"] == 4
    assert both_scores[4] > both_scores[3]

    # Single topic A only: the B-answer is ignored, so they tie.
    only_a = scorer.rank(topic_a, [3, 4], asker_id=None, now=NOW, top_k=3)
    only_scores = _scores_by_person(only_a)
    assert only_scores[3] == only_scores[4]


def test_rank_accepts_single_topic_string(seed_counts, session) -> None:
    # Back-compat: a single topic string still works as before.
    _add_evidence(session, "mt_single", 5)
    scorer = ExpertiseScorer(Repository(session))
    result = scorer.rank(TOPIC, [5], asker_id=None, now=NOW, top_k=3)
    assert result["recommendations"][0]["person_id"] == 5

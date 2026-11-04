"""Integration tests for the retrieval layer against live PostgreSQL + pgvector.

Uses the shared ``engine`` / ``seed_counts`` / ``session`` fixtures (conftest):
CI's pgvector service via ``TEKIJIN_DATABASE_URL`` or an ephemeral ``pgserver``
locally, skipped when neither is available. Dense vectors are injected with the
deterministic ``fake_embedder`` — no model download.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from tekijin.agent.route import PERSON, PRIOR_ANSWER_SIM, decide_route
from tekijin.data.repository import Repository
from tekijin.models.tables import Answer, Document, EmployeeProfile, Question
from tekijin.retrieval import dense
from tekijin.retrieval.indexing import embed_corpus
from tekijin.retrieval.retriever import HybridRetriever


def _vec(fake_embedder, text: str) -> list[float]:
    """Deterministic embedding for a single text via the injected fake."""

    return fake_embedder.encode([text])[0]


# --------------------------------------------------------------------------- #
# repository extension
# --------------------------------------------------------------------------- #
def test_list_answers(seed_counts, session) -> None:
    answers = Repository(session).list_answers()
    assert len(answers) == 150
    assert all(a.responder_id is not None for a in answers)


def test_list_profiles(seed_counts, session) -> None:
    profiles = Repository(session).list_profiles()
    assert len(profiles) == 40
    ids = [p.employee_id for p in profiles]
    assert ids == sorted(ids)  # ordered by employee id


def test_answers_by_ids_resolves_a_subset(seed_counts, session) -> None:
    repo = Repository(session)
    all_answers = repo.list_answers()
    wanted = [all_answers[0].id, all_answers[5].id]
    resolved = repo.answers_by_ids(wanted)
    assert set(resolved) == set(wanted)
    assert resolved[wanted[0]].body == all_answers[0].body
    # Unknown ids are simply absent; empty input queries nothing.
    assert repo.answers_by_ids([*wanted, "does-not-exist"]).keys() == set(wanted)
    assert repo.answers_by_ids([]) == {}


def test_questions_by_ids_resolves_a_subset(seed_counts, session) -> None:
    repo = Repository(session)
    all_questions = repo.list_questions()
    wanted = [all_questions[1].id, all_questions[3].id]
    resolved = repo.questions_by_ids(wanted)
    assert set(resolved) == set(wanted)
    assert resolved[wanted[0]].body == all_questions[1].body
    assert repo.questions_by_ids([]) == {}


def test_documents_by_ids_resolves_a_subset(seed_counts, session) -> None:
    repo = Repository(session)
    all_documents = repo.list_documents()
    wanted = [all_documents[0].id]
    resolved = repo.documents_by_ids(wanted)
    assert set(resolved) == set(wanted)
    assert resolved[wanted[0]].title == all_documents[0].title
    assert repo.documents_by_ids([]) == {}


def test_collect_context_fragments_over_live_repository(seed_counts, session) -> None:
    # End-to-end of the #69 read path: a RetrievalResult's ids re-hydrate to text.
    from tekijin.agent.state import empty_retrieval
    from tekijin.retrieval.fragments import collect_context_fragments

    repo = Repository(session)
    answer = repo.list_answers()[0]
    document = repo.list_documents()[0]
    retrieval = empty_retrieval()
    retrieval["past_answers"] = [
        {"qa_id": answer.id, "score": 0.9, "responder_id": answer.responder_id}
    ]
    retrieval["documents"] = [{"doc_id": document.id, "score": 0.7}]

    fragments = collect_context_fragments(repo, retrieval)
    assert fragments  # non-empty
    joined = "\n".join(fragments)
    if answer.body:
        assert answer.body[:20] in joined


# --------------------------------------------------------------------------- #
# dense search
# --------------------------------------------------------------------------- #
def test_dense_search_ranks_nearest_first(seed_counts, session, fake_embedder) -> None:
    a, b, c = session.scalars(select(Answer).order_by(Answer.id).limit(3)).all()
    a.embedding = _vec(fake_embedder, "alpha unique token")
    b.embedding = _vec(fake_embedder, "beta different token")
    c.embedding = _vec(fake_embedder, "gamma other token")
    session.flush()

    query_vec = _vec(fake_embedder, "alpha unique token")
    hits = dense.search(session, query_vec, "answers", top_k=5)

    # Only the three embedded rows are eligible; NULL-embedding rows are excluded.
    assert len(hits) == 3
    assert hits[0][0] == a.id
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)  # identical vector -> cosine 1
    # Descending similarity.
    sims = [s for _, s in hits]
    assert sims == sorted(sims, reverse=True)


def test_dense_search_excludes_null_embeddings(seed_counts, session, fake_embedder) -> None:
    # Nothing embedded yet -> no eligible rows.
    query_vec = _vec(fake_embedder, "anything")
    assert dense.search(session, query_vec, "documents", top_k=5) == []


def test_dense_search_employee_profiles_returns_employee_id(
    seed_counts, session, fake_embedder
) -> None:
    prof = session.scalars(select(EmployeeProfile).order_by(EmployeeProfile.employee_id)).first()
    prof.embedding = _vec(fake_embedder, "expert network vpn")
    session.flush()
    hits = dense.search(session, _vec(fake_embedder, "expert network vpn"), "employee_profiles", 5)
    assert hits and hits[0][0] == prof.employee_id
    assert isinstance(hits[0][0], int)


def test_dense_search_rejects_unknown_target(seed_counts, session, fake_embedder) -> None:
    with pytest.raises(ValueError, match="unknown target"):
        dense.search(session, _vec(fake_embedder, "x"), "widgets", 5)


# --------------------------------------------------------------------------- #
# indexing (embed_corpus)
# --------------------------------------------------------------------------- #
def test_embed_corpus_fills_missing_embeddings(seed_counts, session, fake_embedder) -> None:
    n_answers_with_body = session.scalars(select(Answer).where(Answer.body.isnot(None))).all()
    expected_answers = len([a for a in n_answers_with_body if (a.body or "").strip()])

    counts = embed_corpus(session, fake_embedder)

    assert counts["documents"] == 36  # #296: 型番文書6件追加
    assert counts["answers"] == expected_answers
    assert counts["questions"] >= 1
    assert counts["employee_profiles"] >= 1
    # Every answer that had text now carries a vector.
    assert all(a.embedding is not None for a in n_answers_with_body if (a.body or "").strip())


def test_embed_corpus_only_missing_is_idempotent(seed_counts, session, fake_embedder) -> None:
    first = embed_corpus(session, fake_embedder)
    assert sum(first.values()) > 0
    second = embed_corpus(session, fake_embedder)  # only_missing default -> nothing left
    assert sum(second.values()) == 0


def test_embed_corpus_reembeds_when_not_only_missing(seed_counts, session, fake_embedder) -> None:
    embed_corpus(session, fake_embedder)
    again = embed_corpus(session, fake_embedder, only_missing=False)
    assert again["documents"] == 36  # re-embedded despite existing vectors (#296: 30->36)


# --------------------------------------------------------------------------- #
# HybridRetriever end-to-end
# --------------------------------------------------------------------------- #
def test_hybrid_retriever_end_to_end(seed_counts, session, fake_embedder) -> None:
    # Give one answer a UNIQUE body so it is the unambiguous top hit — no
    # dependence on how Postgres orders rows that share a (reused) body.
    target = session.scalars(
        select(Answer).where(Answer.body.isnot(None)).order_by(Answer.id)
    ).first()
    target.body = "UNIQZ4242 singular marker phrase"
    session.flush()
    embed_corpus(session, fake_embedder)  # populate dense vectors for every row

    retriever = HybridRetriever(fake_embedder, session, top_k=5)
    result = retriever.search("UNIQZ4242 singular marker phrase")

    assert set(result) == {
        "past_answers",
        "documents",
        "candidate_people",
        "answer_confidence",
        "document_confidence",
        "people_confidence",
        "person_question_similarity",
        # #506: responders of past_answers who have left, so the prior_answer pin
        # can refuse them without consulting the candidate pool it is allowed to
        # reach past.
        "departed_people",
    }
    # Confidences are absolute cosine similarities in [0, 1].
    for key in ("answer_confidence", "document_confidence", "people_confidence"):
        assert 0.0 <= result[key] <= 1.0
    # Exact-body query -> the answer channel confidence is ~1.0 (identical vector).
    assert result["answer_confidence"] == pytest.approx(1.0, abs=1e-6)

    past = result["past_answers"]
    assert past, "expected at least one fused past answer"
    top = past[0]
    assert set(top) == {"qa_id", "score", "responder_id", "reuse_count"}  # #327
    assert top["score"] > 0
    assert isinstance(top["responder_id"], int)
    # Unique body -> the exact-match answer is the unambiguous rank-0 hit.
    assert top["qa_id"] == target.id

    # documents section is well-formed (may be empty for this query)
    assert isinstance(result["documents"], list)
    for doc in result["documents"]:
        assert set(doc) == {"doc_id", "score"}

    # candidate people are employee ids; the top answer's responder leads.
    people = result["candidate_people"]
    assert people and all(isinstance(p, int) for p in people)
    assert people[0] == top["responder_id"]
    assert len(people) == len(set(people))  # de-duplicated
    assert len(people) <= 5

    # #405: per-person question↔past-answer similarity, keyed by responder_id.
    # The exact-body query matches the target answer at cosine ~1.0, so its
    # responder's max question similarity is ~1.0 and all values are in [0, 1].
    qsim = result["person_question_similarity"]
    assert isinstance(qsim, dict)
    assert all(isinstance(k, int) and 0.0 <= v <= 1.0 for k, v in qsim.items())
    assert qsim.get(top["responder_id"]) == pytest.approx(1.0, abs=1e-6)


def test_hybrid_retriever_document_channel(seed_counts, session, fake_embedder) -> None:
    embed_corpus(session, fake_embedder)
    doc = session.scalars(select(Document).where(Document.body.isnot(None))).first()

    retriever = HybridRetriever(fake_embedder, session, top_k=5)
    result = retriever.search(f"{doc.title or ''} {doc.body or ''}")

    doc_ids = [d["doc_id"] for d in result["documents"]]
    assert doc.id in doc_ids


def test_hybrid_retriever_respects_top_k(seed_counts, session, fake_embedder) -> None:
    embed_corpus(session, fake_embedder)
    q = session.scalars(select(Question).where(Question.body.isnot(None))).first()
    retriever = HybridRetriever(fake_embedder, session, top_k=2)
    result = retriever.search(q.body)
    assert len(result["past_answers"]) <= 2
    assert len(result["documents"]) <= 2
    assert len(result["candidate_people"]) <= 2


# --------------------------------------------------------------------------- #
# fix 1: a term living only in the QUESTION still surfaces the answer
# --------------------------------------------------------------------------- #
def test_hybrid_retriever_matches_term_in_question_only(
    seed_counts, session, fake_embedder
) -> None:
    # Rare term appears only in the question; the answer is a generic procedure
    # with no lexical or dense overlap with the query.
    question = Question(
        id="q_term_only", asker_id=1, body="ZQX8888 rare model discussion", topics=["misc"]
    )
    answer = Answer(
        id="ans_generic",
        question_id="q_term_only",
        responder_id=7,
        body="follow the standard onboarding procedure",
    )
    session.add(question)
    session.flush()  # insert the question before its answer (FK order)
    session.add(answer)
    session.flush()
    embed_corpus(session, fake_embedder)

    # BM25 indexes the question body alongside the answer (fix 1), so a term living
    # only in the question is recoverable via the sparse channel. Pin bm25_weight=1.0
    # to isolate THAT indexing behavior from #68's default down-weighting
    # (bm25_weight=0.2 deliberately lets dense outrank a lone BM25 hit — the
    # model-number/exact-match tradeoff is tracked in #114).
    retriever = HybridRetriever(fake_embedder, session, top_k=10, bm25_weight=1.0)
    result = retriever.search("ZQX8888")

    qa_ids = {p["qa_id"] for p in result["past_answers"]}
    assert "ans_generic" in qa_ids  # recovered via question-body BM25 (fix 1)


def test_hybrid_retriever_adaptive_bm25_recovers_model_number(
    seed_counts, session, fake_embedder
) -> None:
    # #114: the same model-number recovery WITHOUT hardcoding bm25_weight=1.0. With
    # the base weight left at the #68 default (0.2) but adaptivity ON, the weak dense
    # signal on a bare model number raises BM25 for that channel, so the exact hit
    # survives — the flat-0.2 default (which the test above pins 1.0 to work around)
    # would drop it.
    question = Question(
        id="q_term_only2", asker_id=1, body="ZQX8888 rare model discussion", topics=["misc"]
    )
    answer = Answer(
        id="ans_generic2",
        question_id="q_term_only2",
        responder_id=7,
        body="follow the standard onboarding procedure",
    )
    session.add(question)
    session.flush()
    session.add(answer)
    session.flush()
    embed_corpus(session, fake_embedder)

    retriever = HybridRetriever(fake_embedder, session, top_k=10)  # base bm25_weight=0.2
    # Enable adaptivity with a window whose lo=1.0 boosts at ANY cosine (<=1.0). This
    # ISOLATES the end-to-end plumbing (settings -> _fuse -> weighted RRF) — it does
    # NOT exercise per-cosine interpolation or which channel's confidence is used
    # (those are covered by test_adaptive_bm25_weight_* and the wiring test below).
    retriever._bm25_boosted = 1.0
    retriever._bm25_adapt_lo, retriever._bm25_adapt_hi = 1.0, 1.001
    result = retriever.search("ZQX8888")

    assert "ans_generic2" in {p["qa_id"] for p in result["past_answers"]}


def test_search_passes_each_channels_own_confidence_to_its_fuse(
    seed_counts, session, fake_embedder
) -> None:
    # #114 review: prove search() wires the RIGHT per-channel dense confidence into
    # each fusion call — answers<-answer_confidence, docs<-document_confidence,
    # people<-people_confidence — independent of the cosine scale. A mixed-up channel
    # would boost/starve the wrong pool; the unconditional-window test above cannot
    # catch it, so capture the dense_confidence handed to _fuse and compare it to the
    # confidences the SAME run reports back.
    retriever = HybridRetriever(fake_embedder, session, top_k=10)
    seen: list[float] = []
    original = retriever._fuse

    def _spy(dense_rankings, sparse_ranking, *, dense_confidence):
        seen.append(dense_confidence)
        return original(dense_rankings, sparse_ranking, dense_confidence=dense_confidence)

    retriever._fuse = _spy  # type: ignore[method-assign]
    result = retriever.search("VPN 3拠点の相談")

    # search() fuses in this order: past answers, documents, then people (via
    # _fused_ids -> _fuse). Each must have received its OWN channel's confidence.
    assert len(seen) == 3
    assert seen[0] == result["answer_confidence"]
    assert seen[1] == result["document_confidence"]
    assert seen[2] == result["people_confidence"]


# --------------------------------------------------------------------------- #
# fix 2: a term living only in a PROFILE surfaces that person (sparse channel)
# --------------------------------------------------------------------------- #
def test_hybrid_retriever_profile_sparse_channel(seed_counts, session, fake_embedder) -> None:
    prof = session.scalars(select(EmployeeProfile).order_by(EmployeeProfile.employee_id)).first()
    prof.description = f"{prof.description or ''} WQZ7777rareterm"
    session.flush()
    embed_corpus(session, fake_embedder)

    # DEFAULT top_k=10: round-robin interleaving (fix B) must keep the exact
    # profile match inside the result even though many answer responders precede
    # it — no reliance on an inflated top_k to avoid the crowd-out.
    retriever = HybridRetriever(fake_embedder, session, top_k=10)
    result = retriever.search("WQZ7777rareterm")

    assert prof.employee_id in result["candidate_people"]  # via profile BM25 (fix 2)


# --------------------------------------------------------------------------- #
# fix 6: channels are pooled BEFORE RRF, so a shared just-below-top_k item wins
# --------------------------------------------------------------------------- #
def test_hybrid_retriever_fuses_before_truncating_to_top_k(
    seed_counts, session, fake_embedder, monkeypatch
) -> None:
    from tekijin.retrieval import retriever as retriever_mod
    from tekijin.retrieval.sparse import BM25Index

    ans_a, ans_x, ans_b = session.scalars(select(Answer).order_by(Answer.id).limit(3)).all()

    # Each channel truncates to the top_k it is *called with* (mimics real search).
    def fake_dense(_session, _vec, target, top_k):
        if target == "answers":
            return [(ans_a.id, 1.0), (ans_x.id, 0.9)][:top_k]
        return []

    def fake_bm25(_self, _query, top_k):
        return [(ans_b.id, 1.0), (ans_x.id, 0.9)][:top_k]

    monkeypatch.setattr(retriever_mod.dense, "search", fake_dense)
    monkeypatch.setattr(BM25Index, "search", fake_bm25)

    # top_k=1: if channels were cut to 1 BEFORE fusion, dense=[A], sparse=[B] and
    # X would be lost. Pooling first lets X (rank 1 in both) win via RRF.
    retriever = HybridRetriever(fake_embedder, session, top_k=1)
    result = retriever.search("anything")

    assert [p["qa_id"] for p in result["past_answers"]] == [ans_x.id]


# --------------------------------------------------------------------------- #
# fix E: a question-matched answer surfaces at default top_k even when the
# direct-answer dense pool is full and BM25 finds no overlap.
# --------------------------------------------------------------------------- #
def test_hybrid_retriever_question_match_surfaces_at_default_top_k(
    seed_counts, session, fake_embedder, monkeypatch
) -> None:
    # The bag-of-tokens fake embedder couples dense similarity to BM25 overlap
    # (both read the question body), so a pure-DB test cannot isolate the
    # question-dense channel. Stub the two search primitives over the REAL repo
    # to reproduce the exact regression: direct-answer pool full, no BM25 hit,
    # the answer reachable ONLY via its strongly-matching question.
    from tekijin.retrieval import retriever as retriever_mod
    from tekijin.retrieval.sparse import BM25Index

    answers = session.scalars(select(Answer).order_by(Answer.id)).all()
    special = answers[-1]  # not among the first 50 dummy ids below
    pool_ids = [a.id for a in answers[:50]]  # fills the direct-answer dense pool

    def fake_dense(_session, _vec, target, top_k):
        if target == "answers":
            return [(aid, 1.0) for aid in pool_ids][:top_k]  # 50 unrelated hits
        if target == "questions":
            return [(special.question_id, 1.0)][:top_k]  # the paraphrased question
        return []  # employee_profiles: irrelevant here

    def fake_bm25(_self, _query, _top_k):
        return []  # no lexical overlap on any channel

    monkeypatch.setattr(retriever_mod.dense, "search", fake_dense)
    monkeypatch.setattr(BM25Index, "search", fake_bm25)

    # DEFAULT top_k=10. Under the old concat approach the special answer sat at
    # dense rank 51 (RRF ~1/111, below all 50 pool ids) and was dropped; as an
    # independent third ranking it ranks ~0 and lands in the top results.
    retriever = HybridRetriever(fake_embedder, session, top_k=10)
    result = retriever.search("paraphrased query with no lexical overlap")

    assert special.id in {p["qa_id"] for p in result["past_answers"]}


# --------------------------------------------------------------------------- #
# answer_confidence gate: a near-duplicate question with NO answers must not
# raise answer_confidence (nobody to hand off to) -> C5 does not pick prior_answer
# --------------------------------------------------------------------------- #
def test_answer_confidence_ignores_answerless_questions(
    seed_counts, session, fake_embedder
) -> None:
    query = "ZQXUNIQ alpha beta gamma delta"  # unique tokens -> no seed collision
    # Question A: an EXACT near-duplicate of the query, but with NO answers.
    session.add(
        Question(id="q_noans", asker_id=1, body=query, topics=["ネットワーク・VPN"], status="open")
    )
    # Question B: unrelated text, but it HAS an answer (also unrelated to the query).
    session.add(
        Question(
            id="q_withans",
            asker_id=1,
            body="無関係な別の話題",
            topics=["セキュリティ"],
            status="open",
        )
    )
    session.flush()
    session.add(
        Answer(
            id="ans_withans",
            question_id="q_withans",
            responder_id=7,
            body="無関係な回答本文",
            topic="セキュリティ",
        )
    )
    session.flush()
    embed_corpus(session, fake_embedder)  # populate dense vectors

    retriever = HybridRetriever(fake_embedder, session, top_k=10)
    result = retriever.search(query)

    # The answerless near-duplicate (cosine ~1.0) does NOT leak into confidence:
    # only questions that actually have an answer count.
    assert result["answer_confidence"] < PRIOR_ANSWER_SIM
    assert result["answer_confidence"] < 0.5  # far from the ~1.0 of the answerless dup
    # ...so C5 does not short-circuit to prior_answer.
    assert decide_route(result).route == PERSON

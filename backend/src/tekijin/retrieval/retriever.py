"""Hybrid retriever (component C4): Dense + BM25 fused with RRF.

Ties the pieces together: embed the query (C3), run dense pgvector search and
in-memory BM25 in parallel channels, and fuse each channel's ranked ids with
Reciprocal Rank Fusion. Produces the C4 output shape consumed by the route
selector (#30) and the agent (#31):

    {
      "past_answers":    [{"qa_id", "score", "responder_id"}, ...],
      "documents":       [{"doc_id", "score"}, ...],
      "candidate_people": [employee_id, ...],
    }

``candidate_people`` is built by round-robin interleaving (responder-first) the
responders of the fused past answers with the profile-match ranking — dense and
sparse (BM25) self-introduction hits fused with RRF — then de-duplicated. The
interleave (rather than "all responders, then profiles") keeps an exact profile
match from being crowded out of ``top_k`` by many answer responders; see
:meth:`_aggregate_people`.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import zip_longest
from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from tekijin.config import get_settings
from tekijin.data.repository import Repository
from tekijin.retrieval import dense
from tekijin.retrieval.bm25_cache import cached_bm25_index
from tekijin.retrieval.embedding import QUERY, Embedder
from tekijin.retrieval.fusion import adaptive_bm25_weight, rrf

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids agent<-retrieval cycle)
    from tekijin.agent.state import DocumentHit, PastAnswer, RetrievalResult

# How many candidates each channel (dense / sparse) retrieves *before* fusion.
# RRF is applied to these pools and only then truncated to ``top_k``: cutting
# each channel to ``top_k`` first would discard items sitting just below the cut
# that RRF would otherwise lift above a single channel's top hits.
CANDIDATE_POOL = 50


class HybridRetriever:
    """Dense + sparse hybrid search over past answers, documents, and people.

    Args:
        embedder: Query embedder (C3). Injected so tests can use a fake.
        session: Active session for dense pgvector search and repository reads.
        top_k: Max hits per result section.
        rrf_k: RRF constant (spec default 60).
        bm25_weight: RRF weight for the sparse (BM25) channel; dense channels stay
            at 1.0. ``None`` reads ``settings.bm25_weight`` (default 0.2). Equal
            weight (1.0) over-trusts BM25 on symptom-worded queries — see #68.
    """

    def __init__(
        self,
        embedder: Embedder,
        session: Session,
        *,
        top_k: int = 10,
        rrf_k: int = 60,
        bm25_weight: float | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if rrf_k <= 0:
            raise ValueError(f"rrf_k must be positive, got {rrf_k}")
        settings = get_settings()
        resolved_weight = settings.bm25_weight if bm25_weight is None else bm25_weight
        if resolved_weight < 0:
            raise ValueError(f"bm25_weight must be non-negative, got {resolved_weight}")
        self._embedder = embedder
        self._session = session
        self._repo = Repository(session)
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._bm25_weight = resolved_weight
        # #114 adaptive-BM25 params (read from settings; None boosted = flat/OFF).
        # A boosted value below the base would DOWN-weight BM25 exactly where dense
        # is weakest — the opposite of the intent — so reject it rather than silently
        # inverting the curve.
        self._bm25_boosted = settings.bm25_weight_boosted
        if self._bm25_boosted is not None and self._bm25_boosted < resolved_weight:
            raise ValueError(
                f"bm25_weight_boosted ({self._bm25_boosted}) must be >= bm25_weight "
                f"({resolved_weight}); it is the weight when dense is WEAK"
            )
        self._bm25_adapt_lo = settings.bm25_adapt_lo
        self._bm25_adapt_hi = settings.bm25_adapt_hi
        # Fail loudly on a backwards/degenerate window WHEN adaptivity is on, mirroring
        # the boosted<base guard above. Otherwise ``adaptive_bm25_weight``'s hi<=lo
        # fallback would silently make an "enabled" feature inert — the exact kind of
        # misconfiguration the DGX tuning must not miss. (When boosted is None the
        # window is unused, so lo/hi are left unchecked.)
        if self._bm25_boosted is not None and self._bm25_adapt_lo >= self._bm25_adapt_hi:
            raise ValueError(
                f"bm25_adapt_lo ({self._bm25_adapt_lo}) must be < bm25_adapt_hi "
                f"({self._bm25_adapt_hi}) when bm25_weight_boosted is set"
            )
        # Per-channel retrieval depth before fusion (see CANDIDATE_POOL).
        self._pool = max(top_k * 5, CANDIDATE_POOL)

    def _fuse(
        self,
        dense_rankings: list[list[Any]],
        sparse_ranking: list[Any],
        *,
        dense_confidence: float,
    ) -> list[tuple[Any, float]]:
        # Dense channels weighted 1.0; the BM25 sparse channel is down-weighted so
        # its noisier ranks (on symptom-worded queries) cannot swamp dense (#68).
        # #114: when THIS channel's dense signal is weak (low top cosine), raise the
        # BM25 weight so an exact term/model-number hit is not starved out of top-k.
        bm25_weight = adaptive_bm25_weight(
            dense_confidence,
            base=self._bm25_weight,
            boosted=self._bm25_boosted,
            lo=self._bm25_adapt_lo,
            hi=self._bm25_adapt_hi,
        )
        # RRF over the full channel pools; truncate to top_k only afterwards.
        rankings = [*dense_rankings, sparse_ranking]
        weights = [1.0] * len(dense_rankings) + [bm25_weight]
        return rrf(rankings, k=self._rrf_k, weights=weights)[: self._top_k]

    def _fused_ids(
        self, dense_rankings: list[list[Any]], sparse_ranking: list[Any], *, dense_confidence: float
    ) -> list[Any]:
        return [
            id_
            for id_, _ in self._fuse(
                dense_rankings, sparse_ranking, dense_confidence=dense_confidence
            )
        ]

    def _dense_hits(self, query_vec: Sequence[float], target: str) -> list[tuple[Any, float]]:
        """``(id, cosine_similarity)`` for ``target``, best-first."""

        return dense.search(self._session, query_vec, target, self._pool)

    @staticmethod
    def _top_similarity(*hit_lists: list[tuple[Any, float]]) -> float:
        """Highest cosine similarity across the given dense hit lists (``0.0`` if
        all empty). This is the channel's absolute confidence for C5."""

        return max((sim for hits in hit_lists for _, sim in hits), default=0.0)

    @staticmethod
    def _question_mapped_answer_ids(
        question_hits: list[tuple[Any, float]],
        answers_by_question: dict[str, list[str]],
    ) -> list[Any]:
        """Answer ids ranked by DENSE similarity of their *question*.

        A user often paraphrases the question rather than the answer, so a dense
        search over answer bodies alone (which are frequently generic procedures)
        can miss the right answer. This expands each question hit to its answers,
        preserving the question ranking (first occurrence wins on duplicates). It
        is fed to RRF as an independent ranking — NOT appended behind the
        direct-answer pool — so an answer whose question matches strongly ranks
        near the top of its own list and gains a large RRF contribution.
        """

        ranked: list[Any] = []
        seen: set[Any] = set()
        for question_id, _sim in question_hits:
            for answer_id in answers_by_question.get(question_id, ()):
                if answer_id not in seen:
                    seen.add(answer_id)
                    ranked.append(answer_id)
        return ranked

    @staticmethod
    def _answer_text(answer: Any, question_body_by_id: dict[str, str | None]) -> str:
        """BM25 index text for an answer: its question's body plus its own.

        Terms a user searches by (model numbers, jargon) often live only in the
        *question*, while the answer is a generic procedure. Indexing both keeps
        those answers reachable. Falls back to the answer body alone when the
        linked question has no body.
        """

        question_body = question_body_by_id.get(answer.question_id)
        answer_body = answer.body or ""
        if question_body:
            return f"{question_body} {answer_body}".strip()
        return answer_body

    def search(self, query: str, *, query_vector: Sequence[float] | None = None) -> RetrievalResult:
        """Retrieve fused past answers, documents, and candidate people.

        ``query_vector`` is the C3 embedding of ``query``; pass it to avoid a
        second embedding call (the dense channels reuse it, BM25 uses the raw
        ``query``). When omitted, the query is embedded here.
        """

        query_vec = (
            query_vector
            if query_vector is not None
            else self._embedder.encode([query], kind=QUERY)[0]
        )

        answers = self._repo.list_answers()
        questions = self._repo.list_questions()
        documents = self._repo.list_documents()
        profiles = self._repo.list_profiles()
        responder_of = {a.id: a.responder_id for a in answers}
        # #327: reuse_count per answer — the corpus-count signal C5 routes prior_answer on.
        reuse_of = {a.id: int(a.reuse_count or 0) for a in answers}
        question_body_by_id = {q.id: q.body for q in questions}
        answers_by_question: dict[str, list[str]] = {}
        for a in answers:
            answers_by_question.setdefault(a.question_id, []).append(a.id)

        # Content-signature cache (#56): identical corpora reuse the built index,
        # skipping SudachiPy tokenization + BM25 fit (a write changes the pairs and
        # rebuilds automatically). The tokenizer/dictionary is shared per process.
        answer_index = cached_bm25_index(
            "answers", ((a.id, self._answer_text(a, question_body_by_id)) for a in answers)
        )
        document_index = cached_bm25_index(
            "documents", ((d.id, f"{d.title or ''} {d.body or ''}") for d in documents)
        )
        profile_index = cached_bm25_index(
            "profiles", ((p.employee_id, p.description or "") for p in profiles)
        )

        # Dense searches (each returns (id, cosine_similarity), best-first).
        answer_hits = self._dense_hits(query_vec, "answers")
        question_hits = self._dense_hits(query_vec, "questions")
        document_hits = self._dense_hits(query_vec, "documents")
        people_hits = self._dense_hits(query_vec, "employee_profiles")

        # Absolute per-channel confidence for C5 (top cosine similarity). The
        # answer channel is the closest of a matching past *answer* or a *question
        # that actually has at least one answer* — a near-duplicate question with
        # NO answer must not raise answer_confidence (nobody to hand off to).
        answered_question_hits = [
            (qid, sim) for qid, sim in question_hits if answers_by_question.get(qid)
        ]
        answer_confidence = self._top_similarity(answer_hits, answered_question_hits)
        document_confidence = self._top_similarity(document_hits)
        people_confidence = self._top_similarity(people_hits)

        # --- past answers -------------------------------------------------- #
        # Three independent rankings fused by RRF: dense over answer bodies, dense
        # over question bodies (mapped to their answers), and sparse BM25.
        sparse_answers = [id_ for id_, _ in answer_index.search(query, self._pool)]
        dense_answers = [id_ for id_, _ in answer_hits]
        question_answers = self._question_mapped_answer_ids(question_hits, answers_by_question)
        fused_answers = self._fuse(
            [dense_answers, question_answers], sparse_answers, dense_confidence=answer_confidence
        )
        past_answers: list[PastAnswer] = [
            {
                "qa_id": id_,
                "score": score,
                "responder_id": responder_of.get(id_),
                "reuse_count": reuse_of.get(id_, 0),
            }
            for id_, score in fused_answers
        ]

        # --- documents ----------------------------------------------------- #
        sparse_docs = [id_ for id_, _ in document_index.search(query, self._pool)]
        fused_docs = self._fuse(
            [[id_ for id_, _ in document_hits]], sparse_docs, dense_confidence=document_confidence
        )
        documents_out: list[DocumentHit] = [
            {"doc_id": id_, "score": score} for id_, score in fused_docs
        ]

        # #405: per-person question↔past-answer similarity. Reuse the answer dense
        # channel (already computed above): each answer_hit is (answer_id, cosine of
        # the QUESTION vs that answer body), so the max over a responder's hits is
        # how well their past answers match THIS question — the signal C6 adds to
        # break topic_fit saturation and rescue C1 topic mispredictions. No extra
        # embedding; answers outside the dense pool are simply absent (treated as 0).
        person_question_similarity: dict[int, float] = {}
        for answer_id, sim in answer_hits:
            responder_id = responder_of.get(answer_id)
            if responder_id is None:
                continue
            if sim > person_question_similarity.get(responder_id, 0.0):
                person_question_similarity[responder_id] = sim

        # --- candidate people ---------------------------------------------- #
        dense_people = [id_ for id_, _ in people_hits]
        sparse_people = [id_ for id_, _ in profile_index.search(query, self._pool)]
        fused_people = self._fused_ids(
            [dense_people], sparse_people, dense_confidence=people_confidence
        )
        # Departed colleagues are dropped HERE rather than from the answer and
        # profile corpora (#506). Their answers stay searchable — the knowledge
        # in them outlives the employment — but they stop being offered as
        # someone to ask, because the hand-off has nowhere to go: the Slack link
        # that would have delivered it was removed by the same departure.
        active = self._repo.active_employee_ids()
        departed_people = sorted(
            {
                responder
                for answer in past_answers
                if (responder := answer.get("responder_id")) is not None and responder not in active
            }
        )
        candidate_people = [
            person
            for person in self._aggregate_people(past_answers, fused_people)
            if person in active
        ]

        return {
            "past_answers": past_answers,
            "documents": documents_out,
            "candidate_people": candidate_people,
            "departed_people": departed_people,
            "answer_confidence": answer_confidence,
            "document_confidence": document_confidence,
            "people_confidence": people_confidence,
            "person_question_similarity": person_question_similarity,
        }

    def _aggregate_people(
        self,
        past_answers: Sequence[PastAnswer],
        profile_people: list[Any],
    ) -> list[Any]:
        """Merge responders and profile matches, then de-duplicate and cap.

        Responders (people who answered similar questions) are the stronger
        signal, but concatenating *all* of them ahead of profile matches lets a
        long responder list push every profile hit past ``top_k`` — so an exact
        profile match could vanish from the result. Instead the two ranked lists
        are round-robin interleaved, responder-first (responder, profile,
        responder, profile, …). Responders keep a slight edge (they lead and take
        the even slots) while a top profile match always lands at position 1,
        safely inside ``top_k``.
        """

        responders: list[Any] = []
        for answer in past_answers:
            responder_id = answer["responder_id"]
            if responder_id is not None and responder_id not in responders:
                responders.append(responder_id)

        people: list[Any] = []
        for responder_id, employee_id in zip_longest(responders, profile_people):
            for candidate in (responder_id, employee_id):
                if candidate is not None and candidate not in people:
                    people.append(candidate)
        return people[: self._top_k]

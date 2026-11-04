"""Tests for knowledge-unit embedding + vector search (#357 slice 3).

Integration tests against the seeded PostgreSQL fixtures, using the dim-matched
``fake_embedder`` so no model download is needed.
"""

from __future__ import annotations

import pytest

from tekijin.data.knowledge import (
    get_knowledge_unit_by_source,
    search_knowledge_units,
    set_review_status,
    upsert_knowledge_unit,
)
from tekijin.knowledge.index import embed_knowledge_units, unit_text
from tekijin.models.tables import KnowledgeUnit


def _vec(fake_embedder, text: str) -> list[float]:
    return fake_embedder.encode([text])[0]


def _upsert(session, source_id: str, problem: str, action: str, result: str | None = None):
    upsert_knowledge_unit(
        session,
        kind="case",
        problem=problem,
        action=action,
        result=result,
        topics=["CRM・営業支援"],
        source_type="slack_thread",
        source_id=source_id,
    )
    session.flush()


# --------------------------------------------------------------------------- #
# unit_text (pure)
# --------------------------------------------------------------------------- #
def test_unit_text_composes_present_parts() -> None:
    full = KnowledgeUnit(kind="case", problem="P", action="A", result="R")
    assert unit_text(full) == "P\nA\nR"
    no_result = KnowledgeUnit(kind="case", problem="P", action="A", result=None)
    assert unit_text(no_result) == "P\nA"
    empty = KnowledgeUnit(kind="case", problem=None, action=None, result=None)
    assert unit_text(empty) == ""


# --------------------------------------------------------------------------- #
# embed_knowledge_units
# --------------------------------------------------------------------------- #
def test_embed_knowledge_units_fills_missing(seed_counts, session, fake_embedder) -> None:
    _upsert(session, "90001", "CRM が定着しない", "SFA/CRM を提案")
    _upsert(session, "90002", "顧客情報の一元管理", "デモを実施")
    embedded = embed_knowledge_units(session, fake_embedder)
    assert embedded == 2
    for sid in ("90001", "90002"):
        dto = get_knowledge_unit_by_source(session, "slack_thread", sid)
        assert dto is not None and dto.has_embedding is True
    # only_missing default: a second run embeds nothing new.
    assert embed_knowledge_units(session, fake_embedder) == 0
    # --all re-embeds every unit.
    assert embed_knowledge_units(session, fake_embedder, only_missing=False) == 2


def test_embed_knowledge_units_skips_empty_text(seed_counts, session, fake_embedder) -> None:
    # A unit with no problem/action (unit_text == "") is not embeddable and skipped.
    upsert_knowledge_unit(
        session,
        kind="case",
        problem=None,
        action=None,
        topics=[],
        source_type="slack_thread",
        source_id="90099",
    )
    session.flush()
    assert embed_knowledge_units(session, fake_embedder) == 0
    assert embed_knowledge_units(session, fake_embedder, batch_size=1) == 0


def test_embed_knowledge_units_skips_rejected(seed_counts, session, fake_embedder) -> None:
    # A human-rejected unit is never embedded (waste avoidance, #354).
    _upsert(session, "90201", "keep this", "action a")
    _upsert(session, "90202", "discard this", "action b")
    rejected = get_knowledge_unit_by_source(session, "slack_thread", "90202")
    set_review_status(session, rejected.id, "rejected")
    session.flush()
    embedded = embed_knowledge_units(session, fake_embedder)
    assert embedded == 1  # only the non-rejected unit
    assert get_knowledge_unit_by_source(session, "slack_thread", "90201").has_embedding is True
    assert get_knowledge_unit_by_source(session, "slack_thread", "90202").has_embedding is False


def test_embed_knowledge_units_rejects_bad_batch_size(seed_counts, session, fake_embedder) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        embed_knowledge_units(session, fake_embedder, batch_size=0)


# --------------------------------------------------------------------------- #
# search_knowledge_units
# --------------------------------------------------------------------------- #
def test_search_ranks_nearest_and_gates_on_review(seed_counts, session, fake_embedder) -> None:
    _upsert(session, "91001", "alpha unique token", "resolve alpha")
    _upsert(session, "91002", "beta different token", "resolve beta")
    _upsert(session, "91003", "gamma other token", "resolve gamma")
    # Approve two; leave 91003 unreviewed.
    for sid in ("91001", "91002"):
        dto = get_knowledge_unit_by_source(session, "slack_thread", sid)
        set_review_status(session, dto.id, "approved")
    session.flush()
    embed_knowledge_units(session, fake_embedder)

    query = _vec(fake_embedder, "alpha unique token resolve alpha")
    hits = search_knowledge_units(session, query, top_k=5)
    ids = [u.source_id for u, _sim in hits]
    # Only approved units are returned; the exact-text match ranks first.
    assert "91003" not in ids  # unreviewed is gated out
    assert ids and ids[0] == "91001"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)  # cosine similarity of identical text

    # review_status=None returns every status (admin view) → the unreviewed appears.
    every = search_knowledge_units(session, query, top_k=5, review_status=None)
    assert "91003" in {u.source_id for u, _ in every}


def test_search_excludes_null_embeddings(seed_counts, session, fake_embedder) -> None:
    # An approved but NOT-yet-embedded unit never surfaces (NULL embedding excluded).
    _upsert(session, "92001", "unembedded problem", "unembedded action")
    dto = get_knowledge_unit_by_source(session, "slack_thread", "92001")
    set_review_status(session, dto.id, "approved")
    session.flush()  # deliberately NOT embedded
    hits = search_knowledge_units(session, _vec(fake_embedder, "unembedded problem"), top_k=5)
    assert "92001" not in {u.source_id for u, _ in hits}


def test_search_rejects_bad_review_status(seed_counts, session, fake_embedder) -> None:
    with pytest.raises(ValueError, match="unknown review status"):
        search_knowledge_units(session, _vec(fake_embedder, "x"), review_status="bogus")

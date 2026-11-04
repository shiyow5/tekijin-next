"""Tests for answering from structured knowledge units (#357 slice 4a).

Unit tests cover the deterministic composer (no DB, no LLM); integration tests
cover the retrieve→compose primitive against the seeded DB with the fake embedder.
"""

from __future__ import annotations

from tekijin.data.dto import KnowledgeUnitDTO
from tekijin.data.knowledge import (
    get_knowledge_unit_by_source,
    set_review_status,
    upsert_knowledge_unit,
)
from tekijin.knowledge.answer import (
    answer_from_knowledge,
    compose_knowledge_answer,
    knowledge_citation_id,
)
from tekijin.knowledge.index import embed_knowledge_units


def _unit(uid: int, problem="CRM 定着", action="SFA 提案", result="受注", industry="製造業"):
    return KnowledgeUnitDTO(
        id=uid,
        kind="case",
        problem=problem,
        action=action,
        result=result,
        topics=("CRM・営業支援",),
        industry=industry,
        source_type="slack_thread",
        source_id=str(1000 + uid),
        confidence=0.9,
        review_status="approved",
        has_embedding=True,
        created_at=None,
    )


def _vec(fake_embedder, text: str) -> list[float]:
    return fake_embedder.encode([text])[0]


# --------------------------------------------------------------------------- #
# compose_knowledge_answer (pure, no LLM)
# --------------------------------------------------------------------------- #
def test_compose_empty_is_ungrounded() -> None:
    res = compose_knowledge_answer([])
    assert res.grounded is False and res.answer == "" and res.cited_source_ids == []


def test_compose_builds_grounded_answer_with_citations() -> None:
    res = compose_knowledge_answer([_unit(1), _unit(2)])
    assert res.grounded is True
    # The answer is built from the units' fields (grounded by construction).
    assert "課題: CRM 定着" in res.answer and "打ち手: SFA 提案" in res.answer
    assert "結果: 受注" in res.answer and "【製造業】" in res.answer
    # Each unit is cited as ku_{id}.
    assert res.cited_source_ids == ["ku_1", "ku_2"]


def test_compose_respects_top_n() -> None:
    res = compose_knowledge_answer([_unit(i) for i in range(1, 6)], top_n=2)
    assert res.cited_source_ids == ["ku_1", "ku_2"]


def test_compose_omits_absent_optional_fields() -> None:
    res = compose_knowledge_answer([_unit(9, result=None, industry=None)])
    assert res.grounded is True
    assert "結果:" not in res.answer and "【" not in res.answer
    assert knowledge_citation_id(_unit(9)) == "ku_9"


def test_compose_never_leaks_literal_none() -> None:
    # problem/action are str|None; a None must never render as the literal "None"
    # in a user-facing answer (defensive guard for manual writes / future kinds).
    res = compose_knowledge_answer(
        [_unit(3, problem=None, action=None, result=None, industry=None)]
    )
    assert "None" not in res.answer
    assert res.grounded is True and res.cited_source_ids == ["ku_3"]


# --------------------------------------------------------------------------- #
# answer_from_knowledge (retrieve → compose, DB)
# --------------------------------------------------------------------------- #
def _store_approved(session, source_id, problem, action, embed=True, approve=True):
    upsert_knowledge_unit(
        session,
        kind="case",
        problem=problem,
        action=action,
        result="受注",
        topics=["CRM・営業支援"],
        source_type="slack_thread",
        source_id=source_id,
        confidence=0.9,
    )
    session.flush()
    if approve:
        dto = get_knowledge_unit_by_source(session, "slack_thread", source_id)
        set_review_status(session, dto.id, "approved")
        session.flush()


def test_answer_from_knowledge_returns_grounded_when_relevant(
    seed_counts, session, fake_embedder
) -> None:
    _store_approved(session, "95001", "alpha unique problem", "alpha action")
    _store_approved(session, "95002", "beta other problem", "beta action")
    embed_knowledge_units(session, fake_embedder)

    query = _vec(fake_embedder, "alpha unique problem alpha action")
    res = answer_from_knowledge(session, query, top_k=5, top_n=1)
    assert res is not None and res.grounded is True
    assert "alpha unique problem" in res.answer
    # Cites a knowledge unit (ku_ prefix).
    assert res.cited_source_ids and res.cited_source_ids[0].startswith("ku_")


def test_answer_from_knowledge_none_below_similarity_floor(
    seed_counts, session, fake_embedder
) -> None:
    _store_approved(session, "95101", "gamma problem", "gamma action")
    embed_knowledge_units(session, fake_embedder)
    # An unrelated query with a floor above any achievable similarity → no answer.
    query = _vec(fake_embedder, "totally unrelated words xyzzy")
    assert answer_from_knowledge(session, query, min_similarity=0.99) is None


def test_answer_from_knowledge_none_when_no_approved(seed_counts, session, fake_embedder) -> None:
    # Unreviewed unit exists and is embedded, but the default approved-gate hides it.
    _store_approved(session, "95201", "delta problem", "delta action", approve=False)
    embed_knowledge_units(session, fake_embedder)
    query = _vec(fake_embedder, "delta problem delta action")
    assert answer_from_knowledge(session, query) is None

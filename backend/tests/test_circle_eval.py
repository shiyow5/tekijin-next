"""The circle evaluation (#108): scoring, and a read-only run over the seeded circle."""

from __future__ import annotations

import dataclasses
import datetime as dt

import pytest
from sqlalchemy import text
from sqlalchemy.exc import InternalError

from tekijin.agent.stubs import KeywordIntentModel
from tekijin.config import Settings
from tekijin.data.circle import load_circle, seed_circle
from tekijin.data.db import get_sessionmaker
from tekijin.eval.circle import (
    CircleIds,
    build_circle_agent,
    circle_ids,
    circle_org_id,
    format_rows,
    format_summary,
    read_only,
    run_circle,
    score_row,
    summarize,
)
from tekijin.llm.factory import make_llm_nodes
from tekijin.retrieval.experts import FLOORS, DbExpertFinder

NOW = dt.datetime(2026, 11, 2, 12, 0, 0)
IDS = CircleIds(citation_ids={"doc:manual": "src_5", "conv:keys": "src_6"}, member_keys={41: "m04"})


# --------------------------------------------------------------------------- #
# Scoring one row
# --------------------------------------------------------------------------- #
def test_a_grounded_answer_is_scored_on_its_own_citations() -> None:
    question = {"id": "q01", "kind": "answer", "sources": ["doc:manual"], "contains": ["守衛室"]}
    values = {
        "route": "document",
        "self_answer_grounded": True,
        "answer": "鍵は守衛室で借ります。",
        "self_answer_citations": [{"source_id": "src_5", "kind": "source"}],
    }

    row = score_row(question, values, (), IDS)

    assert row.grounded and row.cite_ok and row.contains
    assert not (row.handoff or row.followup or row.additive or row.hit_at_3)


def test_a_wrong_citation_or_missing_phrase_is_not_counted() -> None:
    question = {"id": "q01", "kind": "answer", "sources": ["doc:manual"], "contains": ["守衛室"]}
    values = {
        "self_answer_grounded": True,
        "answer": "部長に聞いてください。",
        "self_answer_citations": [{"source_id": "src_9", "kind": "source"}],
    }

    row = score_row(question, values, (), IDS)

    assert row.grounded and not row.cite_ok and not row.contains


def test_a_hand_off_is_scored_on_the_additive_answer_and_the_people() -> None:
    question = {"id": "q29", "kind": "person", "members": ["m04"]}
    values = {
        "route": "person",
        "additive_answer_text": "赤道儀は岩田さんが整備しています。",
        "additive_citations": [{"source_id": "src_6", "kind": "source"}],
        "recommendations": [{"person_id": 41}, {"person_id": 42}],
    }

    row = score_row(question, values, ("send",), IDS)

    assert row.handoff and row.additive and row.hit_at_3
    assert row.top3 == ("m04", "42")
    # No expected sources or phrases: neither counts.
    assert not row.cite_ok and not row.contains


def test_a_pending_follow_up_and_a_refusal_are_recorded() -> None:
    question = {"id": "q26", "kind": "abstain"}
    assert score_row(question, {}, ("ask",), IDS).followup
    assert score_row(question, {"out_of_scope": True}, (), IDS).out_of_scope


def test_the_summary_counts_each_kind() -> None:
    rows = [
        score_row({"id": "a", "kind": "answer"}, {"route": "document"}, (), IDS),
        score_row({"id": "b", "kind": "answer"}, {"out_of_scope": True}, (), IDS),
        score_row(
            {"id": "c", "kind": "person", "members": ["m04"]},
            {"route": "person", "recommendations": [{"person_id": 41}]},
            ("send",),
            IDS,
        ),
    ]

    summary = summarize(iter(rows))

    assert summary["answer"]["n"] == 2 and summary["answer"]["out_of_scope"] == 1
    assert summary["answer"]["routes"] == {"-": 1, "document": 1}
    assert summary["abstain"]["n"] == 0
    assert summary["person"]["handoff"] == 1 and summary["person"]["hit_at_3"] == 1
    assert summary["person"]["seeks_person"] == 0
    table = format_summary(summary)
    assert table.splitlines()[0].startswith("kind | n | out_of_scope")
    assert "person | 1 |" in table
    assert format_rows(rows).splitlines()[2].startswith("c person  person    ---H----R top3=m04")


# --------------------------------------------------------------------------- #
# A run over the seeded circle, with the keyword stubs
# --------------------------------------------------------------------------- #
class _Bigrams:
    """Character bigrams, as in test_circle_integration: unspaced Japanese has no
    whitespace for the shared fake embedder to split on."""

    def __init__(self, words) -> None:
        self._words = words

    def encode(self, texts, *, kind: str = "passage") -> list[list[float]]:
        squashed = ["".join(t.split()) for t in texts]
        grams = [" ".join(t[i : i + 2] for i in range(len(t) - 1)) or t for t in squashed]
        return self._words.encode(grams, kind=kind)


class _ConfidentKeywords:
    """The keyword stub, confident enough that C2 never asks back."""

    def analyze(self, question, asker, *, context=None):
        result = KeywordIntentModel().analyze(question, asker, context=context)
        return dataclasses.replace(result, confidence=0.9)


@pytest.fixture
def bigrams(fake_embedder) -> _Bigrams:
    return _Bigrams(fake_embedder)


@pytest.fixture
def seeded(engine, bigrams):
    factory = get_sessionmaker(engine)
    with factory() as session:
        seed_circle(session, bigrams, load_circle(), now=NOW)
        session.commit()
    yield factory
    with factory() as cleanup:
        for table in (
            "self_intros",
            "chunks",
            "source_items",
            "channel_messages",
            "discord_channels",
            "discord_threads",
            "identities",
        ):
            cleanup.execute(text(f"DELETE FROM {table}"))
        cleanup.execute(text("DELETE FROM employees WHERE org_id IS NOT NULL"))
        cleanup.execute(text("DELETE FROM organizations"))
        cleanup.commit()


def test_every_circle_question_runs_read_only(seeded, bigrams) -> None:
    circle = load_circle()
    settings = Settings(llm_backend="stub")
    with seeded() as session:
        read_only(session)
        org_id = circle_org_id(session)
        ids = circle_ids(session, circle, org_id)
        finder = DbExpertFinder(session, floors=dict.fromkeys(FLOORS, 0.0))
        _, *rest = make_llm_nodes(settings)
        nodes = (_ConfidentKeywords(), *rest)
        agent = build_circle_agent(session, bigrams, settings, nodes, expert_finder=finder)
        rows = run_circle(agent, circle, org_id, ids, now=NOW)
        # The run wrote nothing: the transaction is still read only.
        with pytest.raises(InternalError):
            session.execute(text("DELETE FROM organizations"))
        session.rollback()

    assert len(ids.member_keys) == len(circle.members)
    assert set(ids.citation_ids) == {r for q in circle.questions for r in q.get("sources", [])}
    summary = summarize(rows)
    assert [summary[k]["n"] for k in ("answer", "abstain", "person")] == [22, 6, 8]
    person = [r for r in rows if r.kind == "person"]
    # The stub sees every person question as one, and with no floors somebody is
    # always found, so each one waits at the hand-off.
    assert all(r.seeks_person and r.handoff and r.route == "person" for r in person), format_rows(
        person
    )
    assert summary["person"]["hit_at_3"] >= 4

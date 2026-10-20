"""Plumbing test for ``scripts/research_fullgraph_eval.py``.

The full-graph eval harness decides whether feature flags (self_answer /
query_expansion / knowledge) may be enabled, so its state-extraction must be
correct. This test loads the script by path and checks:

* ``_terminal_route`` maps every terminal/route to the label ``metrics.decision_class``
  expects (self_answered / no_expert / person / ask / none).
* ``GraphRanker`` drives the REAL compiled graph over a person-routed query and
  extracts a non-empty expert ranking with route ``person`` — i.e. the fields the
  metrics read (``recommendation_ids`` / ``route``) come back populated.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path

from tekijin.agent import build_agent
from tekijin.agent.state import RetrievalResult
from tekijin.eval.dataset import EvalQuery
from tekijin.models.tables import Skill

_TOPIC = "ネットワーク・VPN"

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "research_fullgraph_eval.py"


def _load_harness():
    spec = importlib.util.spec_from_file_location("research_fullgraph_eval", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRetriever:
    """C4 stand-in returning a fixed retrieval dict (controls the C5 route)."""

    def __init__(self, *, people=()) -> None:
        self._payload: RetrievalResult = {
            "past_answers": [],
            "documents": [],
            "candidate_people": list(people),
            "answer_confidence": 0.0,
            "document_confidence": 0.0,
            "people_confidence": 0.9,
        }

    def search(self, query: str, *, query_vector: Sequence[float] | None = None) -> RetrievalResult:
        return self._payload


def test_terminal_route_maps_every_terminal() -> None:
    h = _load_harness()
    tr = h._terminal_route
    # grounded self-answer wins over any route already set.
    grounded = {"self_answer_grounded": True, "route": "person"}
    assert tr(grounded, (), critique_wired=True) == "self_answered"
    # critic declined -> no_expert (only when a critic is wired).
    assert tr({"answerable": False, "route": "person"}, (), critique_wired=True) == "no_expert"
    assert tr({"answerable": False, "route": "person"}, (), critique_wired=False) == "person"
    # plain person route.
    assert tr({"route": "person"}, ("send",), critique_wired=False) == "person"
    # paused at the C2 clarification.
    assert tr({}, ("ask",), critique_wired=False) == "ask"
    # off_topic / no_candidate leave route unset -> abstain-ish.
    assert tr({}, (), critique_wired=False) == "none"


def test_graph_ranker_extracts_person_ranking(seed_counts, session, fake_embedder) -> None:
    h = _load_harness()
    for emp in (1, 2, 3):
        session.add(
            Skill(id=f"sk_fg_{emp}", employee_id=emp, topic=_TOPIC, level="中級", source="self")
        )
    session.flush()
    graph = build_agent(
        fake_embedder,
        session,
        retriever=_FakeRetriever(people=[1, 2, 3]),
    )
    ranker = h.GraphRanker(graph, critique_wired=False)
    query = EvalQuery(
        id=9001,
        query="現行のVPN機器で3拠点の拠点間接続について相談したいです",
        gold_topics=[_TOPIC],
        gold_experts=[1],
        gold_route="person",
        difficulty="L1",
        expect_abstain=False,
        gold_experts_alt=[],
    )
    result = ranker(query)
    assert result.route == "person"
    assert result.ranked_experts, "person route must surface a ranked expert list"
    assert all(isinstance(pid, int) for pid in result.ranked_experts)
    # Per-row record, so a metric that moves between machines can be traced to rows.
    (row,) = ranker.rows
    assert row["id"] == 9001
    assert row["gold_route"] == "person" and row["route"] == "person"
    assert row["topics"] == result.predicted_topics
    assert row["ranked_experts"] == result.ranked_experts
    assert set(row) >= {"out_of_scope", "intent_confidence", "next", "error"}

"""Metric tests for ``scripts/research_knowledge_judge.py``.

The judge comparison (#56) decides which scorer may gate the knowledge answer, so its
metrics must be right: ROC-AUC with ties, the threshold that keeps the false-fire rate
under a cap, and cross-validated firing where every row is decided by a threshold fitted
WITHOUT that row. These run without a DB, an embedder or any model.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "research_knowledge_judge.py"


def _load():
    spec = importlib.util.spec_from_file_location("research_knowledge_judge", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_roc_auc_perfect_inverse_and_ties() -> None:
    m = _load()
    assert m.roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert m.roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    # every positive ties every negative -> 0.5
    assert m.roc_auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
    # one positive above both negatives, one tied with one negative: (1 + 1 + 1 + 0.5) / 4
    assert m.roc_auc([0.9, 0.4, 0.4, 0.1], [1, 1, 0, 0]) == 0.875


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic() -> None:
    m = _load()
    scores = [0.9, 0.7, 0.6, 0.4, 0.35, 0.3, 0.2, 0.1]
    labels = [1, 1, 0, 1, 0, 0, 0, 0]
    lo, hi = m.bootstrap_auc_ci(scores, labels, n_boot=300, seed=0)
    assert 0.0 <= lo <= m.roc_auc(scores, labels) <= hi <= 1.0
    assert (lo, hi) == m.bootstrap_auc_ci(scores, labels, n_boot=300, seed=0)


def test_threshold_at_fpr_maximises_recall_under_the_cap() -> None:
    m = _load()
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    labels = [1, 0, 1, 1, 0, 0]
    # cap 0 -> no negative may fire: lowest safe threshold is above 0.8
    t0 = m.threshold_at_fpr(scores, labels, max_fpr=0.0)
    assert t0 > 0.8
    assert [s >= t0 for s in scores] == [True, False, False, False, False, False]
    # cap 1/3 -> one negative may fire: fire down to 0.6 (3 positives, 1 negative)
    t1 = m.threshold_at_fpr(scores, labels, max_fpr=1 / 3)
    assert t1 == 0.6


def test_threshold_never_fires_when_nothing_is_safe() -> None:
    m = _load()
    t = m.threshold_at_fpr([0.9, 0.1], [0, 1], max_fpr=0.0)
    assert math.isinf(t) and t > 0


def test_stratified_folds_balance_each_class() -> None:
    m = _load()
    labels = [1] * 10 + [0] * 25
    folds = m.stratified_folds(labels, k=5, seed=0)
    assert len(folds) == len(labels)
    for f in range(5):
        pos = sum(1 for y, g in zip(labels, folds, strict=True) if g == f and y == 1)
        neg = sum(1 for y, g in zip(labels, folds, strict=True) if g == f and y == 0)
        assert pos == 2 and neg == 5


def test_cv_fire_decides_each_row_with_a_threshold_fitted_without_it() -> None:
    m = _load()
    # perfectly separable: any held-out threshold fitted at fpr 0 keeps negatives quiet
    scores = [0.9, 0.85, 0.8, 0.75, 0.7] + [0.3, 0.25, 0.2, 0.15, 0.1] * 2
    labels = [1] * 5 + [0] * 10
    fired = m.cv_fire(scores, labels, max_fpr=0.0, k=5, seed=0)
    assert len(fired) == len(scores)
    assert not any(f for f, y in zip(fired, labels, strict=True) if y == 0)
    # the threshold sits on the lowest SAFE training score, so only the overall-lowest
    # positive (0.7) may miss when it is the held-out one
    missed = [s for s, f, y in zip(scores, fired, labels, strict=True) if y == 1 and not f]
    assert set(missed) <= {0.7}


def test_yes_no_score_from_top_logprobs() -> None:
    m = _load()
    top = [
        {"token": "yes", "logprob": math.log(0.6)},
        {"token": " No", "logprob": math.log(0.2)},
        {"token": "Yes", "logprob": math.log(0.1)},
    ]
    assert math.isclose(m.yes_no_score(top), 0.7 / 0.9)
    assert m.yes_no_score([{"token": "maybe", "logprob": 0.0}]) == 0.5


def test_fit_reranker_ids_truncates_only_the_body() -> None:
    m = _load()
    prefix, body, suffix = [1, 2], list(range(100, 110)), [8, 9]
    assert m.fit_reranker_ids(prefix, body, suffix, max_length=100) == [*prefix, *body, *suffix]
    # over budget: the suffix (the yes/no decision position) must survive
    ids = m.fit_reranker_ids(prefix, body, suffix, max_length=7)
    assert ids == [1, 2, 100, 101, 102, 8, 9]


def test_run_scorer_times_scoring_apart_from_loading() -> None:
    m = _load()
    loads: list[int] = []

    def loader(_args):
        loads.append(1)
        return lambda p: p["cosine"] * 2

    pairs = [{"cosine": 0.3, "unit": {"text": "x"}}, {"cosine": 0.0, "unit": None}]
    scores, load_s, ms_per_pair = m.run_scorer(loader, pairs, None)
    assert loads == [1]
    assert scores == [0.6, 0.0]  # a pair without a unit never fires
    assert load_s >= 0.0 and ms_per_pair >= 0.0

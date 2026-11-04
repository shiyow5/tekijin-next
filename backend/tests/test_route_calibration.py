"""C5 の閾値が実データの分布と噛み合っているかを見張る（Issue #105 / #90）。

#103 は「**実装は動き、単体テストも通り、数字だけが壊れている**」形の不具合だった。
`decide_route` 自体は仕様どおり動く。壊れていたのは**閾値と実際のコサイン分布の関係**で、
e5-large では `answer_confidence` の最小値(0.816)が `PRIOR_ANSWER_SIM`(0.80) を超えるため、
全71件が `prior_answer` に倒れて層2 Recall@3 が 0.592 落ちていた。

#90 で埋め込みを Nemotron-3-Embed-1B に替え、閾値をその実測分布に較正した。
#191 で評価セット（当時 全81件 / 採点66件）基準に再較正し、DOCUMENT_SIM=0.30→0.28 とした
（当時の実測: 経路精度 0.818 > 多数決 0.742、1経路への潰れ 0.94 < 0.95。0.30 のままだと
潰れ 0.95 で制約を割り document recall も 4/10 に落ちていた）。

**較正データは #434 で現行コーパス（全87件 / 採点72件）で取り直した**（閾値は据え置き）。
現在の実測は下表。81件のまま置いていたときより、どの制約も余裕が広がっている。

| 指標 | 81件（#191 当時） | **87件（現行）** | 制約 |
|---|---|---|---|
| 1経路への潰れ | 76/81 = 0.938 | **76/87 = 0.874** | < 0.95 |
| 経路精度 | 0.818 | **0.833**（60/72） | > 多数決 |
| 多数決 | 0.742 | **0.681** | — |

経路別の内訳は person 49/49・document 11/16・prior_answer 0/7（下記のとおり意図的に無効）。
ただし `answer_confidence` は prior_answer を分離できず
（person gold が prior_answer gold より高い）、PRIOR_ANSWER_SIM=0.55 は観測最大 0.542 の
直上に置いて意図的に無効化している。**prior_answer 経路の復活は打ち止め**: コーパス集計
ルーティング(#119/#327)は実測でどの config も baseline を Pareto 改善せず、ADR-0007 で棄却された
（#119 は close 済み）。自己回答は経路でなく知識層(#357)で実現する方針。

既存の単体テストは「閾値を超えたら prior_answer を返すか」を見ている。
ここで見るのは「**その閾値は実際のデータで超えられるのか / 常に超えてしまわないか**」。

較正データ (`fixtures/synthetic/eval/route_calibration.json`) は実 DB・実埋め込みで記録した
チャネル類似度。ここでは読むだけなので **GPU も DB も要らない**。

    # 再測定（埋め込みモデルを変えたとき / コーパスを作り直したとき）
    # research_e2e.py は #89 で消した。company-v1 タグの版で次を実行する
    python scripts/research_e2e.py --task prepare
    python scripts/research_e2e.py --task route --out fixtures/synthetic/eval/route_calibration.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from tekijin.agent.route import (
    DOCUMENT_SIM,
    PERSON_WEAK_SIM,
    PRIOR_ANSWER_SIM,
    decide_route,
)
from tekijin.config import get_settings

# 一つの経路がここまで占めたら、実質的に分岐が死んでいる。
# gold の多数派（routed のうち person が 0.68）に none 15件ぶんが乗るので、健全に較正しても
# person が最大勢力になるのは正常。現行コーパス（87件）の実測は 76/87 = 0.874（#434）。
# 81件の較正データを使っていた間はこれが 76/81 = 0.938 で、しきい値まで 0.012 しか無かった
# ——つまり DOCUMENT_SIM を少し動かすだけで「古い較正データの上でだけ」CI が落ちる状態だった。
# 閾値自体は据え置き（0.95）で、余裕が 0.076 に戻っている。
_COLLAPSE_RATIO = 0.95
_REMEASURE = (
    "再測定（company-v1 タグの版で）: python scripts/research_e2e.py --task prepare && "
    "python scripts/research_e2e.py --task route "
    "--out fixtures/synthetic/eval/route_calibration.json"
)
_FIXED_HINT = (
    "ADR-0007 で prior_answer 経路の復活は打ち止めになったので、この xfail は当面そのまま。"
    "コーパス/埋め込みが変わって answer_confidence が person gold と分離できるようになったら "
    "削除すること（strict=True なので xpass すると CI が落ちて気づける）"
)


def _calibration_path() -> Path:
    return get_settings().fixtures_dir / "eval" / "route_calibration.json"


@pytest.fixture(scope="module")
def calibration() -> dict:
    path = _calibration_path()
    if not path.exists():
        pytest.skip(f"較正データが無い。{_REMEASURE}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def rows(calibration: dict) -> list[dict]:
    return calibration["rows"]


def test_calibration_matches_the_configured_embedding_model(calibration: dict) -> None:
    """コサインの絶対値はモデル依存。モデルを変えたら閾値は必ず測り直す（#63 / #90）。

    Nemotron-3-Embed-1B ではコサインの最大が 0.57 しかなく、同じ閾値が今度は
    **一度も発火しない**側に倒れる。較正データとモデルが食い違ったまま通すと、その事故を見逃す。
    """

    recorded = calibration["_meta"]["embedding_model"]
    configured = get_settings().embedding_model
    assert recorded == configured, (
        f"較正データは {recorded} で取ったものだが、設定は {configured}。{_REMEASURE}"
    )


@pytest.mark.parametrize(
    ("channel", "threshold", "name"),
    [
        pytest.param(
            "answer_confidence",
            PRIOR_ANSWER_SIM,
            "PRIOR_ANSWER_SIM",
            marks=pytest.mark.xfail(
                reason=(
                    "#119: prior_answer は Nemotron のコサインでは分離できない"
                    "（answer_confidence は person 側が prior_answer gold より高い）。"
                    "PRIOR_ANSWER_SIM は観測最大(0.542)の直上に置いて意図的に無効化している。"
                    f"{_FIXED_HINT}"
                ),
                strict=True,
            ),
        ),
        ("document_confidence", DOCUMENT_SIM, "DOCUMENT_SIM"),
        ("people_confidence", PERSON_WEAK_SIM, "PERSON_WEAK_SIM"),
    ],
)
def test_threshold_sits_inside_the_observed_distribution(
    rows: list[dict], channel: str, threshold: float, name: str
) -> None:
    """閾値が観測範囲の外にあると、その分岐は定数述語になる（常に真か常に偽）。

    #90 で DOCUMENT_SIM / PERSON_WEAK_SIM は Nemotron 分布内に較正済み。
    PRIOR_ANSWER_SIM だけは意図的に分布の外（発火しない側）に置いているので xfail のまま。
    """

    values = [r[channel] for r in rows]
    lo, hi = min(values), max(values)
    assert lo < threshold < hi, (
        f"{name}={threshold} が {channel} の観測範囲 [{lo:.3f}, {hi:.3f}] の外にある。"
        f"この分岐は常に同じ結果しか返さない。{_REMEASURE}"
    )


def test_routes_do_not_collapse_to_a_single_branch(rows: list[dict]) -> None:
    """経路が1つに潰れていないこと。潰れていると候補が固定され、推薦が成立しない。"""

    predicted = Counter(
        decide_route(
            {
                "answer_confidence": r["answer_confidence"],
                "document_confidence": r["document_confidence"],
                "people_confidence": r["people_confidence"],
                "candidate_people": [1] * r["n_candidates"],
                "past_answers": [{"responder_id": 1, "score": 1.0}],
            }
        ).route
        for r in rows
    )
    top_route, count = predicted.most_common(1)[0]
    ratio = count / len(rows)
    assert ratio < _COLLAPSE_RATIO, (
        f"予測経路の {ratio:.0%} が {top_route} に偏っている（{dict(predicted)}）。"
    )


def test_route_accuracy_beats_the_majority_baseline(rows: list[dict]) -> None:
    """多数決（常に person）を下回るなら、経路判定は害にしかなっていない。"""

    routed = [r for r in rows if r["gold_route"] != "none"]
    assert routed, "経路つきの較正データが無い"
    majority = max(Counter(r["gold_route"] for r in routed).values()) / len(routed)
    hits = sum(
        1
        for r in routed
        if decide_route(
            {
                "answer_confidence": r["answer_confidence"],
                "document_confidence": r["document_confidence"],
                "people_confidence": r["people_confidence"],
                "candidate_people": [1] * r["n_candidates"],
                "past_answers": [{"responder_id": 1, "score": 1.0}],
            }
        ).route
        == r["gold_route"]
    )
    accuracy = hits / len(routed)
    assert accuracy > majority, f"経路精度 {accuracy:.3f} が多数決 {majority:.3f} 以下。"

"""research_latency.py — 各 LLM ノード呼び出しの実レイテンシを直接測る（#472）。

AI 回答の速度の律速は vLLM 呼び出し。person 経路は C1→C2→(additive compose)→C7 draft と
直列に LLM を呼ぶ。どのノードが支配的かを、**DB も埋め込みも通さず**各アダプタを直接叩いて
測る（最短で律速を isolate する）。self_answer/additive は同じ composer なので compose を1回で代表。

各呼び出しを N 回まわし、mean / p50 / p95 を出す。person 経路の推定合計も出す
（C1 + C2 + additive_compose + draft。C2 は #377 fast-path でスキップされうる点は注記）。

使い方（DGX・本番 vLLM 稼働前提・DB不要）:
    PYTHONPATH=backend/src TEKIJIN_APP_ENV=development \
      TEKIJIN_LLM_BACKEND=vllm TEKIJIN_LLM_BASE_URL=http://localhost:18080/v1 \
      TEKIJIN_LLM_MODEL=Qwen3.6-35B-A3B-NVFP4 \
      .venv/bin/python scripts/research_latency.py --repeat 5 --out latency.json

使い方（evo-x2・llama-server 稼働前提。docs/benchmarks/2026-10-21-evo-x2-llamacpp.md）:
    PYTHONPATH=backend/src TEKIJIN_APP_ENV=development \
      TEKIJIN_LLM_BACKEND=vllm TEKIJIN_LLM_BASE_URL=http://127.0.0.1:8080/v1 \
      TEKIJIN_LLM_MODEL=qwen3.6-35b-a3b TEKIJIN_LLM_THINKING_CONTROL=chat_template \
      TEKIJIN_LLM_STRUCTURED_OUTPUT_METHOD=json_schema \
      python scripts/research_latency.py --repeat 3 --out latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "backend", "src")

# 代表質問（person 経路に流れる知識質問。実 eval の傾向に合わせた手書き）。
_QUESTIONS = [
    "お客様がUTMの入れ替えを検討中。他社製品からの移行で注意点は?",
    "契約書管理が適当な客への提案で、費用対効果をどう説明するのがいいですか",
    "拠点間のVPNが不安定という相談。切り分けの初手は?",
    "CRMを導入したが定着しない客への打ち手は?",
]

# self_answer/additive の compose に渡す最小の根拠（実経路と同型・CitedEvidence は下で構築）。
_EVIDENCE_RAW = [
    ("qa_sample", "qa", "過去に同様の移行提案でSFA/CRMを提案し受注した事例がある。"),
    ("doc_sample", "document", "移行時は現行製品の設定を棚卸ししてから切替える。"),
]


def _stats(samples: list[float]) -> dict[str, float]:
    s = sorted(samples)
    n = len(s)
    return {
        "n": n,
        "mean_ms": round(statistics.mean(s) * 1000, 1),
        "p50_ms": round(s[n // 2] * 1000, 1),
        "p95_ms": round(s[min(n - 1, int(n * 0.95))] * 1000, 1),
        "min_ms": round(s[0] * 1000, 1),
        "max_ms": round(s[-1] * 1000, 1),
    }


class _Timer:
    """Context manager appending the elapsed seconds to ``bucket`` on exit."""

    def __init__(self, bucket: list[float]) -> None:
        self._bucket = bucket

    def __enter__(self) -> None:
        self._t0 = time.perf_counter()

    def __exit__(self, *exc: object) -> None:
        self._bucket.append(time.perf_counter() - self._t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5, help="各質問×各ノードの反復回数")
    ap.add_argument("--out", default="latency.json")
    args = ap.parse_args()

    sys.path.insert(0, SRC)
    from tekijin.config import get_settings
    from tekijin.llm.vllm import (
        VllmDraftModel,
        VllmIntentModel,
        VllmSelfAnswerModel,
        VllmSufficiencyModel,
    )
    from tekijin.retrieval.fragments import CitedEvidence

    settings = get_settings()
    intent = VllmIntentModel(settings=settings)
    sufficiency = VllmSufficiencyModel(settings=settings)
    draft = VllmDraftModel(settings=settings)
    self_answer = VllmSelfAnswerModel(settings=settings)
    evidence = [
        CitedEvidence(source_id=sid, kind=kind, text=text)
        for sid, kind, text in _EVIDENCE_RAW
    ]

    timings: dict[str, list[float]] = {
        "c1_intent": [],
        "c2_sufficiency": [],
        "c7_draft": [],
        "compose": [],
    }

    # warm-up（初回は接続確立/キャッシュで遅い。計測から除外）。
    try:
        intent.analyze(_QUESTIONS[0], {"id": 0})
    except Exception as exc:  # noqa: BLE001
        print(f"warm-up failed: {exc}", file=sys.stderr)

    for q in _QUESTIONS:
        for _ in range(args.repeat):
            try:
                with _Timer(timings["c1_intent"]):
                    intent_out = intent.analyze(q, {"id": 0})
                # C2 は intent を受けて充足判定（followup_count=0）。
                with _Timer(timings["c2_sufficiency"]):
                    sufficiency.check(q, intent_out, 0)
                # C7 draft（相手・不足項目のダミーで文面生成）。
                with _Timer(timings["c7_draft"]):
                    draft.draft(
                        q,
                        {"name": "高梨", "dept": "技術部"},
                        {"id": 0},
                        ["現行製品", "拠点数"],
                        known_values={},
                    )
                # self_answer/additive の compose（同 composer）。
                with _Timer(timings["compose"]):
                    self_answer.compose(q, evidence)
            except Exception as exc:  # noqa: BLE001
                print(f"call failed for {q[:20]!r}: {exc}", file=sys.stderr)

    report = {k: _stats(v) for k, v in timings.items() if v}
    # person 経路の推定（直列・p50 合算）。additive は floor 超のときだけ。
    p50 = {k: report[k]["p50_ms"] for k in report}
    est = {
        "person_no_additive_p50_ms": round(
            p50.get("c1_intent", 0)
            + p50.get("c2_sufficiency", 0)
            + p50.get("c7_draft", 0),
            1,
        ),
        "person_with_additive_p50_ms": round(
            p50.get("c1_intent", 0)
            + p50.get("c2_sufficiency", 0)
            + p50.get("compose", 0)
            + p50.get("c7_draft", 0),
            1,
        ),
        "note": "C2 は #377 fast-path で routable ならスキップ。additive は floor(0.20)超のときのみ。",
    }

    out = {"per_node": report, "person_route_estimate": est, "repeat": args.repeat}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

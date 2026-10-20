"""research_fullgraph_eval.py — 本番グラフ(build_agent)を eval 全行にE2Eで流し、
任意のフラグ構成(self_answer / query_expansion / knowledge / answerability)を
既存の指標(Hit@3 / Recall@3 / source recall / decision recall)で一括測定する。

これまでの計測は PipelineRanker(検索+採点のみ・C5/self_answer/knowledge ノードを通らない)
や retrieval-recall 近似で、フラグ有効化の是非をE2Eで判定できなかった。本ハーネスは
`Ranker` プロトコル(EvalQuery -> RankResult)を build_agent の実グラフで実装するので、
`run_eval` を通せば**既存の全メトリクスがそのまま**出る(再発明ゼロ)。

測るもの(フラグごとに1回):
- **Hit@3 / Recall@3 / Top1**(系統②取次ぎ・top3に有効専門家が居るか)
- **decision recall**(self_answer / route / abstain の C5 振り分け・#297)
- **source recall / precision**(self_answer が gold_source を引用したか・ハルシ検知・#297)
- **person 非回帰**(query_expansion / self_answer を ON にして person 経路の recall が落ちないか)

使い方（ローカル配線確認・stub バックエンド・vLLM 不要）:
    PYTHONPATH=backend/src TEKIJIN_APP_ENV=development \
      python scripts/research_fullgraph_eval.py --backend stub \
      --db-url postgresql+psycopg://... --out fg_stub.json

使い方（DGX・throwaway prepare 済み・本番 vLLM 稼働・埋め込みCPU）:
    PYTHONPATH=backend/src CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 \
      TEKIJIN_EMBEDDING_MODEL=/home/team_a/models/Nemotron-3-Embed-1B-BF16 \
      TEKIJIN_APP_ENV=development \
      TEKIJIN_LLM_BACKEND=vllm TEKIJIN_LLM_BASE_URL=http://localhost:18080/v1 \
      TEKIJIN_LLM_MODEL=Qwen3.6-35B-A3B-NVFP4 \
      .venv/bin/python scripts/research_fullgraph_eval.py --backend vllm \
      --self-answer --query-expansion \
      --db-url postgresql+psycopg://postgres:calibpw@localhost:15442/calib \
      --out fg_self_answer_qexp.json

各フラグは既定 OFF。--self-answer / --query-expansion / --answerability / --knowledge-floor
で個別に立て、baseline(何も立てない)と差分を見る。DGX は共有機なので **1構成ずつ逐次**。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "backend", "src")

# 本番の eval 基準時刻（tekijin.eval.__main__ の EVAL_NOW と一致させる。採点の
# recency / 7日ロード窓が run 間で再現するよう固定）。
EVAL_NOW = dt.datetime(2026, 8, 22, 0, 0, 0)


def _build_models(backend: str, settings, *, self_answer: bool, answerability: bool):
    """(intent, sufficiency, draft, self_answer_model, answerability_model) を返す。

    stub: すべて None（build_agent が決定的 stub を使う）。ただし self_answer /
    answerability は「モデルを渡したときだけノードが配線される」設計なので、stub でも
    ノードを試したい場合に備えて None のまま（有効化は vLLM 実測が前提）。
    vllm: 本番 LLM を配線。self_answer / answerability はフラグが立ったときだけ渡す。
    """

    if backend == "stub":
        # 決定的 stub。self_answer/answerability ノードは vLLM 実測が前提なので
        # stub では配線しない（compose/critic の LLM が要る）。CLI で立てても無視
        # されるので、黙って握り潰さず警告する（起動ログの True と実態の乖離防止）。
        if self_answer or answerability:
            print(
                "  !! --backend stub では self_answer/answerability ノードは配線されません"
                "（vLLM 実測が前提）。フラグは無視されます。",
                file=sys.stderr,
            )
        return None, None, None, None, None

    from tekijin.llm.vllm import (
        VllmAnswerabilityModel,
        VllmDraftModel,
        VllmIntentModel,
        VllmSelfAnswerModel,
        VllmSufficiencyModel,
    )

    intent = VllmIntentModel(settings=settings)
    sufficiency = VllmSufficiencyModel(settings=settings)
    draft = VllmDraftModel(settings=settings)
    sa = VllmSelfAnswerModel(settings=settings) if self_answer else None
    ans = VllmAnswerabilityModel(settings=settings) if answerability else None
    return intent, sufficiency, draft, sa, ans


def _row_record(
    query,
    values: dict,
    next_nodes: tuple[str, ...],
    route: str,
    ranked: list,
    error: str | None,
) -> dict:
    """1行分の診断記録(出力 JSON の ``rows``)。"""

    return {
        "id": query.id,
        "gold_route": query.gold_route,
        "route": route,
        "topics": list(values.get("topics") or []),
        "out_of_scope": values.get("out_of_scope"),
        "intent_confidence": values.get("intent_confidence"),
        "ranked_experts": list(ranked),
        "next": list(next_nodes),
        "error": error,
    }


def _terminal_route(
    values: dict, next_nodes: tuple[str, ...], *, critique_wired: bool
) -> str:
    """最終 state から「予測ルート/終端」を1つに定める(metrics.decision_class 互換)。

    self_answer が grounded なら self_answered（#291/#357 の自己回答終端）。critic が
    否決したら no_expert。どちらでもなければ C5 の route。ask で中断したなら followup。
    それ以外(off_topic/no_candidate/unresolved)は route 未設定なので abstain 相当に倒す。
    """

    if values.get("self_answer_grounded"):
        return "self_answered"
    if critique_wired and values.get("answerable") is False:
        return "no_expert"
    route = values.get("route")
    if route:
        return route
    if "ask" in next_nodes:
        return "ask"
    return "none"


class GraphRanker:
    """build_agent の実グラフを Ranker(EvalQuery -> RankResult) として実装。"""

    def __init__(self, graph, *, critique_wired: bool) -> None:
        self._graph = graph
        self._critique_wired = critique_wired
        self.errors = 0  # rows whose invoke crashed (counted for the output audit)
        # #413: rows where the additive cited answer fired on a person route.
        self.additive_fired_person = 0
        self.person_rows = 0
        # One record per row (written to the output JSON as ``rows``), so a metric
        # that moves between machines or models can be traced back to the rows.
        self.rows: list[dict] = []

    def _record(self, query, values, next_nodes, route, ranked, *, error=None) -> None:
        self.rows.append(_row_record(query, values, next_nodes, route, ranked, error))

    def __call__(self, query):
        from tekijin.eval.runner import RankResult

        config = {"configurable": {"thread_id": f"eval-{query.id}"}}
        initial = {
            # asker は offline eval では不明。存在しない id にして誰もフィルタしない
            # （PipelineRanker が asker=None を scorer に渡すのと同じ意図）。
            "question": query.query,
            "asker": {"id": 0},
            "now": EVAL_NOW,
            "question_id": f"eval_{query.id}",
        }
        try:
            self._graph.invoke(initial, config)
            values = self._graph.get_state(config).values or {}
            next_nodes = tuple(self._graph.get_state(config).next)
        except Exception as exc:  # noqa: BLE001 — 1行の失敗で全体を止めない
            self.errors += 1
            print(f"  !! id={query.id} invoke失敗: {exc}", file=sys.stderr)
            self._record(query, {}, (), "none", [], error=str(exc))
            return RankResult(ranked_experts=[], route="none", predicted_topics=[])

        route = _terminal_route(values, next_nodes, critique_wired=self._critique_wired)
        # C6 writes ``recommendations`` (rank-ordered dicts); ``recommendation_ids``
        # is a service-layer persistence field, absent in the raw graph state.
        recs = values.get("recommendations") or []
        ranked = [
            r["person_id"] for r in recs if isinstance(r, dict) and "person_id" in r
        ]
        citations = values.get("self_answer_citations") or []
        cited = [
            c["source_id"]
            for c in citations
            if isinstance(c, dict) and c.get("source_id")
        ]
        # #413: additive citation firing on the person route (does NOT change route).
        if route == "person":
            self.person_rows += 1
            if values.get("additive_answer_text"):
                self.additive_fired_person += 1
        topics = list(values.get("topics") or [])
        self._record(query, values, next_nodes, route, ranked)
        return RankResult(
            ranked_experts=ranked,
            route=route,
            predicted_topics=topics,
            cited_source_ids=cited,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", required=True)
    ap.add_argument("--out", default="fullgraph_eval.json")
    ap.add_argument("--backend", choices=["stub", "vllm"], default="vllm")
    ap.add_argument(
        "--self-answer", action="store_true", help="#291 self_answer を配線"
    )
    ap.add_argument(
        "--query-expansion", action="store_true", help="#371 クエリ拡張を ON"
    )
    ap.add_argument(
        "--question-fit",
        action="store_true",
        help="#405 C6 に質問↔過去回答の意味一致(qsim)項を足す(routing不変)",
    )
    ap.add_argument(
        "--additive",
        action="store_true",
        help="#413 person経路でも引用付き回答を併記(self_answer composer 配線・routing不変)",
    )
    ap.add_argument(
        "--daily-knowledge",
        action="store_true",
        help="#433 日報を System1 の知識源に(C4 に daily dense channel・routing不変)",
    )
    ap.add_argument(
        "--additive-floor",
        type=float,
        default=None,
        help="#413 additive の cosine フロア（未指定=config 既定）。有効化前の floor 実測用",
    )
    ap.add_argument(
        "--score-all-employees",
        action="store_true",
        help="#87 C6 の候補を C4 の集合でなく全社員にする(経路シグナルは不変)",
    )
    ap.add_argument(
        "--branch-constraint",
        action="store_true",
        help="#83 明示された拠点を C6 で条件として扱う",
    )
    ap.add_argument(
        "--answerability", action="store_true", help="#70 棄却クリティックを配線"
    )
    ap.add_argument(
        "--knowledge-floor",
        type=float,
        default=None,
        help="#357 knowledge_answer の cosine フロア（未指定=ノード無し）",
    )
    ap.add_argument("--limit", type=int, default=0, help="先頭N件だけ（配線確認用）")
    args = ap.parse_args()

    url = args.db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    os.environ["TEKIJIN_DATABASE_URL"] = url
    sys.path.insert(0, SRC)

    from tekijin.agent.graph import build_agent
    from tekijin.config import get_settings
    from tekijin.data.db import get_engine, get_sessionmaker
    from tekijin.eval.dataset import load_eval_queries
    from tekijin.eval.runner import format_report, run_eval
    from tekijin.retrieval.embedding import SentenceTransformerEmbedder

    settings = get_settings()
    session = get_sessionmaker(get_engine(url))()
    embedder = SentenceTransformerEmbedder(
        use_e5_prefix=settings.embedding_use_e5_prefix,
        query_prefix=settings.embedding_query_prefix,
        passage_prefix=settings.embedding_passage_prefix,
        trust_remote_code=settings.embedding_trust_remote_code,
        revision=settings.embedding_model_revision,
        app_env=settings.app_env,
    )
    # #413 additive needs the self_answer composer wired (it shares it).
    wire_self_answer = args.self_answer or args.additive
    intent, sufficiency, draft, sa_model, ans_model = _build_models(
        args.backend,
        settings,
        self_answer=wire_self_answer,
        answerability=args.answerability,
    )
    # Fall back to the actual production default, not a duplicated literal, so an
    # "unspecified" eval run always matches the config the app ships with.
    additive_floor = (
        args.additive_floor
        if args.additive_floor is not None
        else settings.additive_self_answer_floor
    )

    graph = build_agent(
        embedder,
        session,
        intent_model=intent,
        sufficiency_model=sufficiency,
        draft_model=draft,
        self_answer_model=sa_model,
        answerability_model=ans_model,
        knowledge_answer_min_similarity=args.knowledge_floor,
        query_expansion_enabled=args.query_expansion,
        question_fit_enabled=args.question_fit,
        branch_constraint_enabled=args.branch_constraint,
        additive_self_answer_enabled=args.additive,
        additive_self_answer_floor=additive_floor,
        daily_knowledge_enabled=args.daily_knowledge,
        score_all_employees=args.score_all_employees,
    )
    ranker = GraphRanker(graph, critique_wired=ans_model is not None)

    queries = load_eval_queries()
    if args.limit:
        queries = queries[: args.limit]
    print(
        f"backend={args.backend} self_answer={args.self_answer} "
        f"query_expansion={args.query_expansion} question_fit={args.question_fit} "
        f"score_all_employees={args.score_all_employees} "
        f"branch_constraint={args.branch_constraint} "
        f"answerability={args.answerability} "
        f"knowledge_floor={args.knowledge_floor} rows={len(queries)}"
    )

    report = run_eval(queries, ranker)
    print(format_report(report))
    if ranker.errors:
        print(
            f"  ※ invoke 失敗行: {ranker.errors}/{len(queries)}（miss/abstain として計上）"
        )
    if args.additive:
        print(
            f"  #413 additive: person経路 {ranker.person_rows}行中 "
            f"{ranker.additive_fired_person}行で引用回答が併記(取次ぎは維持)"
        )

    out = {
        "config": {
            "backend": args.backend,
            "self_answer": args.self_answer,
            "query_expansion": args.query_expansion,
            "question_fit": args.question_fit,
            "additive": args.additive,
            "additive_floor": additive_floor,
            "daily_knowledge": args.daily_knowledge,
            "score_all_employees": args.score_all_employees,
            "branch_constraint": args.branch_constraint,
            "answerability": args.answerability,
            "knowledge_floor": args.knowledge_floor,
            "rows": len(queries),
            "errors": ranker.errors,
            "additive_fired_person": ranker.additive_fired_person,
        },
        "metrics": report.metrics.as_dict(),
        "by_difficulty": {k: v.as_dict() for k, v in report.by_difficulty.items()},
        "decision_recall": report.decision_recall.as_dict(),
        "source_recall": report.source_recall.as_dict(),
        "rows": ranker.rows,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")
    session.close()
    session.get_bind().dispose()


if __name__ == "__main__":
    main()

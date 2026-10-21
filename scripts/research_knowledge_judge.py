"""research_knowledge_judge.py — 知識回答を出すかの判定を、判断器ごとに比べる(#56)。

research_knowledge_falsefire の結果: cosine の閾値では、人に回すべき質問への誤発火を抑えると
知識回答がまったく発火しなくなる。そこで「この知識単位で質問に直接答えられるか」に点数を付ける
判断器を並べ、同じ87行・同じ知識単位で比べる。

段階は3つに分かれている。DB と埋め込みが要るのは最初の段階だけ。

1. pairs: 評価行ごとに最上位の知識単位を取り、質問と知識単位の組を JSON に書く
2. score: 組ごとに判断器で点数を付ける(cosine / llm / reranker / sokudan)
3. report: データ系の行(document / prior_answer)を正例、人に回す行と答えない行を負例として、
   ROC-AUC(bootstrap 95%信頼区間)と、5分割交差検証で誤発火率の上限から閾値を決めたときの
   発火率を出す。各行の発火は、その行を含まない4分割で決めた閾値で判定する

使い方(evo-x2・抽出+埋込+approve 済み):
    export PYTHONPATH=backend/src TEKIJIN_APP_ENV=development HF_HUB_OFFLINE=1
    DB=postgresql+psycopg://tekijin:tekijin@localhost:5432/tekijin
    S=scripts/research_knowledge_judge.py
    python $S pairs --db-url $DB --out pairs.json
    python $S score --pairs pairs.json --judge cosine --out s_cos.json
    python $S score --pairs pairs.json --judge llm \
      --llm-base-url http://127.0.0.1:8080/v1 --llm-model qwen3.6-35b-a3b --out s_llm.json
    python $S score --pairs pairs.json --judge reranker \
      --reranker-model Qwen/Qwen3-Reranker-4B --name reranker-4b --out s_rr4b.json
    python $S score --pairs pairs.json --judge sokudan --out s_sok.json
    python $S report --pairs pairs.json --scores s_*.json --out judge.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections.abc import Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "backend", "src")
DATA_ROUTES = ("document", "prior_answer")
MAX_FPRS = (0.05, 0.10)

# 判断器に渡す判定の定義。research_knowledge_llm_gate.py と同じ基準にそろえる。
CRITERION_JA = (
    "知識ケース(問題→打ち手→結果)が、ユーザーの質問が求めている具体的な回答を実際に含んでいる"
    "ときだけ yes とする。単に同じ分野・同じトピックというだけ、あるいは関連する人を紹介すべき"
    "相談は no とする。確信が持てなければ no。"
)
CRITERION_EN = (
    "Given a question posted to a company's internal help desk, judge whether the knowledge "
    "case (problem, action, result) actually contains the specific answer the question asks "
    "for. Answer no when it merely shares the topic, or when the question should be handed "
    "to a person who knows the matter."
)


# ---------------------------------------------------------------------------
# metrics (pure; tested in backend/tests/test_research_knowledge_judge.py)


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """P(score of a random positive > score of a random negative); ties count half."""

    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def bootstrap_auc_ci(
    scores: Sequence[float], labels: Sequence[int], *, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Percentile 95% interval of the AUC over row resamples (resamples lacking a class skip)."""

    rng = random.Random(seed)
    n = len(scores)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        ys = [labels[i] for i in idx]
        if 0 < sum(ys) < n:
            aucs.append(roc_auc([scores[i] for i in idx], ys))
    aucs.sort()
    lo = aucs[int(0.025 * (len(aucs) - 1))]
    hi = aucs[math.ceil(0.975 * (len(aucs) - 1))]
    return lo, hi


def threshold_at_fpr(scores: Sequence[float], labels: Sequence[int], *, max_fpr: float) -> float:
    """Lowest threshold (fire when ``score >= t``) whose false-fire rate stays within ``max_fpr``.

    The lowest such threshold fires the most positives. Returns ``+inf`` (never fire) when even
    the top score already breaks the cap.
    """

    n_neg = sum(1 for y in labels if y == 0)
    best = math.inf
    for t in sorted(set(scores), reverse=True):
        fp = sum(1 for s, y in zip(scores, labels, strict=True) if y == 0 and s >= t)
        if (fp / n_neg if n_neg else 0.0) > max_fpr:
            break
        best = t
    return best


def stratified_folds(labels: Sequence[int], *, k: int = 5, seed: int = 0) -> list[int]:
    """Fold id per row, dealing each class round-robin after a seeded shuffle."""

    rng = random.Random(seed)
    folds = [0] * len(labels)
    for cls in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == cls]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[i] = j % k
    return folds


def cv_fire(
    scores: Sequence[float], labels: Sequence[int], *, max_fpr: float, k: int = 5, seed: int = 0
) -> list[bool]:
    """Fire decision per row, using a threshold fitted on the other ``k - 1`` folds."""

    folds = stratified_folds(labels, k=k, seed=seed)
    fired = [False] * len(scores)
    for f in range(k):
        train = [i for i, g in enumerate(folds) if g != f]
        t = threshold_at_fpr(
            [scores[i] for i in train], [labels[i] for i in train], max_fpr=max_fpr
        )
        for i, g in enumerate(folds):
            if g == f:
                fired[i] = scores[i] >= t
    return fired


def yes_no_score(top_logprobs: Sequence[dict]) -> float:
    """P(yes) / (P(yes) + P(no)) from one position's top log-probs; 0.5 when neither appears."""

    p = {"yes": 0.0, "no": 0.0}
    for item in top_logprobs:
        tok = str(item.get("token", "")).strip().lower()
        if tok in p:
            p[tok] += math.exp(float(item["logprob"]))
    total = p["yes"] + p["no"]
    return p["yes"] / total if total > 0 else 0.5


# ---------------------------------------------------------------------------
# stage 1: pairs


def _unit_text(unit) -> str:
    return f"問題: {unit.problem or ''}\n打ち手: {unit.action or ''}\n結果: {unit.result or ''}"


def build_pairs(db_url: str) -> list[dict]:
    url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    os.environ["TEKIJIN_DATABASE_URL"] = url
    sys.path.insert(0, SRC)

    from tekijin.config import get_settings
    from tekijin.data.db import get_engine, get_sessionmaker
    from tekijin.data.knowledge import search_knowledge_units
    from tekijin.eval.dataset import load_eval_queries
    from tekijin.retrieval.embedding import QUERY, SentenceTransformerEmbedder

    s = get_settings()
    session = get_sessionmaker(get_engine(url))()
    embedder = SentenceTransformerEmbedder(
        use_e5_prefix=s.embedding_use_e5_prefix,
        query_prefix=s.embedding_query_prefix,
        passage_prefix=s.embedding_passage_prefix,
        trust_remote_code=s.embedding_trust_remote_code,
        revision=s.embedding_model_revision,
        app_env=s.app_env,
    )
    pairs = []
    try:
        for q in load_eval_queries():
            qvec = embedder.encode([q.query], kind=QUERY)[0]
            hits = search_knowledge_units(session, qvec, top_k=1, review_status="approved")
            unit, sim = (hits[0][0], float(hits[0][1])) if hits else (None, 0.0)
            gold = set(q.gold_topics or ())
            pairs.append(
                {
                    "id": q.id,
                    "query": q.query,
                    "gold_route": q.gold_route,
                    "gold_topics": sorted(gold),
                    "label": 1 if q.gold_route in DATA_ROUTES else 0,
                    "cosine": sim,
                    "topic_match": bool(unit and set(unit.topics) & gold),
                    "unit": (
                        {"id": unit.id, "topics": list(unit.topics), "text": _unit_text(unit)}
                        if unit
                        else None
                    ),
                }
            )
    finally:
        session.close()
        session.get_bind().dispose()
    return pairs


# ---------------------------------------------------------------------------
# stage 2: scorers. Each returns one float per pair (higher = more answerable).


def score_cosine(pairs: list[dict], _args) -> list[float]:
    return [p["cosine"] for p in pairs]


def score_llm(pairs: list[dict], args) -> list[float]:
    """Qwen3.6 behind an OpenAI-compatible server: P(yes) of the first answer token."""

    import httpx

    system = (
        "あなたは社内の質問に、蓄積された知識ケースで直接答えられるかを判定する審査員です。"
        + CRITERION_JA
        + "答えは yes か no の一語だけで返してください。"
    )
    out = []
    with httpx.Client(base_url=args.llm_base_url, timeout=120) as client:
        for p in pairs:
            if p["unit"] is None:
                out.append(0.0)
                continue
            user = (
                f"ユーザーの質問:\n{p['query']}\n\n知識ケース:\n{p['unit']['text']}"
                "\n\n直接答えられますか。yes か no で答えてください。"
            )
            body = {
                "model": args.llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            r = client.post("/chat/completions", json=body)
            r.raise_for_status()
            content = r.json()["choices"][0]["logprobs"]["content"]
            out.append(yes_no_score(content[0]["top_logprobs"]) if content else 0.5)
    return out


def score_reranker(pairs: list[dict], args) -> list[float]:
    """Qwen3-Reranker: P(yes) over {yes, no} at the last position (the model card's recipe)."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.reranker_model, padding_side="left")
    dtype = torch.bfloat16 if args.reranker_dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.reranker_model, torch_dtype=dtype).eval()
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    prefix = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
        'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    out = []
    with torch.no_grad():
        for p in pairs:
            if p["unit"] is None:
                out.append(0.0)
                continue
            text = (
                f"{prefix}<Instruct>: {CRITERION_EN}\n<Query>: {p['query']}\n"
                f"<Document>: {p['unit']['text']}{suffix}"
            )
            enc = tok(text, return_tensors="pt", truncation=True, max_length=8192)
            logits = model(**enc).logits[0, -1, [no_id, yes_id]].float()
            out.append(float(torch.softmax(logits, dim=-1)[1]))
    return out


def score_sokudan(pairs: list[dict], _args) -> list[float]:
    """sokudan-ja-310m bool (temperature-calibrated P(true); monotone, so AUC is unaffected)."""

    import sokudan

    agent = sokudan.load("GeneLab/sokudan-ja-310m", backend="torch")
    question = {"answerable": {"type": "bool", "instructions": CRITERION_JA}}
    out = []
    for p in pairs:
        if p["unit"] is None:
            out.append(0.0)
            continue
        state = f"質問: {p['query']}\n\n知識ケース:\n{p['unit']['text']}"
        res = agent.predict(state, question)
        out.append(float(res["answers"]["answerable"]["noul"]))
    return out


SCORERS = {
    "cosine": score_cosine,
    "llm": score_llm,
    "reranker": score_reranker,
    "sokudan": score_sokudan,
}


# ---------------------------------------------------------------------------
# stage 3: report


def _rate(flags: Sequence[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def summarize(pairs: list[dict], scores: Sequence[float], *, k: int = 5, seed: int = 0) -> dict:
    labels = [p["label"] for p in pairs]
    lo, hi = bootstrap_auc_ci(scores, labels, seed=seed)
    result = {
        "auc": round(roc_auc(scores, labels), 4),
        "auc_ci95": [round(lo, 4), round(hi, 4)],
        "at_fpr": {},
    }
    for cap in MAX_FPRS:
        fired = cv_fire(scores, labels, max_fpr=cap, k=k, seed=seed)
        rows = list(zip(fired, pairs, strict=True))
        data = [(f, p) for f, p in rows if p["gold_route"] in DATA_ROUTES]
        result["at_fpr"][str(cap)] = {
            "data_fire": _rate([f for f, _ in data]),
            "data_fire_topic_match": _rate([f and p["topic_match"] for f, p in data]),
            "person_false_fire": _rate([f for f, p in rows if p["gold_route"] == "person"]),
            "abstain_false_fire": _rate([f for f, p in rows if p["gold_route"] == "none"]),
        }
    return result


def report(pairs: list[dict], score_files: Sequence[str]) -> dict:
    by_id = {p["id"]: p for p in pairs}
    out = {
        "n": len(pairs),
        "n_pos": sum(p["label"] for p in pairs),
        "n_person": sum(1 for p in pairs if p["gold_route"] == "person"),
        "n_abstain": sum(1 for p in pairs if p["gold_route"] == "none"),
        "judges": {},
    }
    for path in score_files:
        with open(path, encoding="utf-8") as f:
            sf = json.load(f)
        ordered = [by_id[int(i)] for i in sf["scores"]]
        summary = summarize(ordered, list(sf["scores"].values()))
        summary["ms_per_pair"] = sf.get("ms_per_pair")
        out["judges"][sf["name"]] = summary
    return out


def _print_table(rep: dict) -> None:
    print(
        f"n={rep['n']} 正例(data)={rep['n_pos']} "
        f"person={rep['n_person']} abstain={rep['n_abstain']}"
    )
    print(
        " judge          | AUC (95%CI)         | 上限 | data発火 | topic一致 | person誤発火"
        " | abstain誤発火 | ms/組"
    )
    for name, j in rep["judges"].items():
        ci = f"{j['auc']:.3f} ({j['auc_ci95'][0]:.3f}-{j['auc_ci95'][1]:.3f})"
        for cap, r in j["at_fpr"].items():
            print(
                f" {name:14s} | {ci:19s} | {cap:4s} | {r['data_fire']:8.3f}"
                f" | {r['data_fire_topic_match']:9.3f} | {r['person_false_fire']:12.3f}"
                f" | {r['abstain_false_fire']:13.3f} | {j['ms_per_pair']}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_pairs = sub.add_parser("pairs")
    p_pairs.add_argument("--db-url", required=True)
    p_pairs.add_argument("--out", default="kb_judge_pairs.json")
    p_score = sub.add_parser("score")
    p_score.add_argument("--pairs", required=True)
    p_score.add_argument("--judge", required=True, choices=sorted(SCORERS))
    p_score.add_argument("--name", default=None, help="結果の表に出す名前(既定は --judge)")
    p_score.add_argument("--llm-base-url", default="http://127.0.0.1:8080/v1")
    p_score.add_argument("--llm-model", default="qwen3.6-35b-a3b")
    p_score.add_argument("--reranker-model", default="Qwen/Qwen3-Reranker-0.6B")
    p_score.add_argument("--reranker-dtype", choices=("fp32", "bf16"), default="fp32")
    p_score.add_argument("--out", required=True)
    p_rep = sub.add_parser("report")
    p_rep.add_argument("--pairs", required=True)
    p_rep.add_argument("--scores", nargs="+", required=True)
    p_rep.add_argument("--out", default="kb_judge.json")
    args = ap.parse_args()

    if args.cmd == "pairs":
        pairs = build_pairs(args.db_url)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.out} ({len(pairs)} pairs, positives={sum(p['label'] for p in pairs)})")
        return

    with open(args.pairs, encoding="utf-8") as f:
        pairs = json.load(f)
    if args.cmd == "score":
        start = time.perf_counter()
        scores = SCORERS[args.judge](pairs, args)
        elapsed = time.perf_counter() - start
        name = args.name or args.judge
        payload = {
            "name": name,
            "judge": args.judge,
            "ms_per_pair": round(1000 * elapsed / max(len(pairs), 1), 1),
            "scores": {str(p["id"]): s for p, s in zip(pairs, scores, strict=True)},
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.out} ({name}, {payload['ms_per_pair']} ms/pair)")
        return

    rep = report(pairs, args.scores)
    _print_table(rep)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

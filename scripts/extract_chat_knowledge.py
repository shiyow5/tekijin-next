"""#448: 社内チャット → ケース知識単位の抽出バッチ（オフライン）。

生チャットはノイズ（挨拶・連絡・雑談）が大半で、そのまま知識源にしても System1 の
grounded 率はほぼ動かない（research_knowledge_source.py 実測）。
価値は「やり取り」＝課題に対して具体的な回答/打ち手が付いた会話にある。そこで抽出単位を
**会話**（同一チャンネル・時間窓でまとめた連続メッセージ）とし、本番 vLLM(:18080) の
structured output で `問題→打ち手/回答→結果` に蒸留、extractable=false でチャットの大半を捨てる。
チャットにはタグが無いので topics は LLM の topic_hints を正規語彙へスナップ（off-vocab は破棄）。
graph からは呼ばれない純オフライン。人手レビュー（#354）は review_status=unreviewed のまま。

抽出品質を測れるよう、格納後に件数と
サンプル数件（source_id/topics/problem/action/result/confidence）を出力する。

使い方（DGX・throwaway pgvector を prepare 済み・本番 vLLM 稼働前提）:
    PYTHONPATH=backend/src CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 \
      TEKIJIN_EMBEDDING_MODEL=/home/team_a/models/Nemotron-3-Embed-1B-BF16 \
      TEKIJIN_APP_ENV=development \
      TEKIJIN_LLM_BACKEND=vllm TEKIJIN_LLM_BASE_URL=http://localhost:18080/v1 \
      TEKIJIN_LLM_MODEL=Qwen3.6-35B-A3B-NVFP4 \
      .venv/bin/python scripts/extract_chat_knowledge.py \
      --db-url postgresql+psycopg://postgres:calibpw@localhost:15433/calib \
      --limit 200 --out chat_knowledge_extract.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "backend", "src")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-url", required=True)
    ap.add_argument("--limit", type=int, default=None, help="抽出対象の会話数上限（PoC 用）")
    ap.add_argument("--gap-minutes", type=int, default=30, help="会話を区切る無音時間（分）")
    ap.add_argument("--min-messages", type=int, default=2, help="会話とみなす最小メッセージ数")
    ap.add_argument("--out", default="chat_knowledge_extract.json")
    args = ap.parse_args()

    url = args.db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    os.environ["TEKIJIN_DATABASE_URL"] = url
    sys.path.insert(0, SRC)

    from tekijin.config import get_settings
    from tekijin.data.db import get_engine, get_sessionmaker, session_scope
    from tekijin.data.knowledge import list_knowledge_units
    from tekijin.knowledge.chat import chat_conversation_sources
    from tekijin.knowledge.extract import CaseExtractor, extract_and_store

    settings = get_settings()
    factory = get_sessionmaker(get_engine(url))
    extractor = CaseExtractor(settings=settings)

    with session_scope(factory) as session:
        sources = chat_conversation_sources(
            session,
            gap_minutes=args.gap_minutes,
            min_messages=args.min_messages,
            limit=args.limit,
        )
        print(f"抽出対象の会話: {len(sources)} 件（生 2000 行から grouping）")
        counts = extract_and_store(session, sources, extractor, infer_topics_from_hints=True)
        print(f"抽出結果: {counts}")
        stored = [u for u in list_knowledge_units(session) if u.source_type == "chat"]

    samples = [
        {
            "source_id": u.source_id,
            "topics": list(u.topics),
            "industry": u.industry,
            "problem": u.problem,
            "action": u.action,
            "result": u.result,
            "confidence": u.confidence,
            "review_status": u.review_status,
        }
        for u in stored[:10]
    ]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "counts": counts,
                "chat_units": len(stored),
                "samples": samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"wrote {args.out} (chat_units={len(stored)})")


if __name__ == "__main__":
    main()

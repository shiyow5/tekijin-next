#!/usr/bin/env python3
"""Embed knowledge articles into their pgvector column (#112).

Runs after drafts were added to ``knowledge_articles`` (their embedding starts
NULL). The app embeds an article when it is edited or verified on the web, so this
is for drafts written in bulk. Uses the REAL local embedding model, so it needs the
ML deps — never CI. By default only fills NULL embeddings; ``--all`` re-embeds
every article that is not retired.

    PYTHONPATH=backend/src CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 \
      TEKIJIN_EMBEDDING_MODEL=/home/team_a/models/Nemotron-3-Embed-1B-BF16 \
      TEKIJIN_APP_ENV=development \
      .venv/bin/python scripts/embed_knowledge.py \
      --db-url postgresql+psycopg://postgres:calibpw@localhost:15441/calib
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "backend", "src")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db-url", required=True)
    ap.add_argument(
        "--all",
        dest="only_missing",
        action="store_false",
        help="re-embed every article, not just those with a NULL embedding",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    url = args.db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    os.environ["TEKIJIN_DATABASE_URL"] = url
    sys.path.insert(0, SRC)

    from tekijin.config import get_settings
    from tekijin.data.db import get_engine, get_sessionmaker, session_scope
    from tekijin.knowledge.index import embed_articles
    from tekijin.retrieval.embedding import SentenceTransformerEmbedder

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(
        use_e5_prefix=settings.embedding_use_e5_prefix,
        query_prefix=settings.embedding_query_prefix,
        passage_prefix=settings.embedding_passage_prefix,
        trust_remote_code=settings.embedding_trust_remote_code,
        revision=settings.embedding_model_revision,
        app_env=settings.app_env,
    )
    # Fail fast on a width mismatch before opening a DB connection.
    probe = embedder.encode(["テスト"], kind="passage")[0]
    if len(probe) != settings.embedding_dim:
        raise ValueError(
            f"embedding model width {len(probe)} != embedding_dim {settings.embedding_dim}"
        )

    factory = get_sessionmaker(get_engine(url))
    with session_scope(factory) as session:
        embedded = embed_articles(
            session, embedder, only_missing=args.only_missing, batch_size=args.batch_size
        )
    print(f"embedded knowledge articles: {embedded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

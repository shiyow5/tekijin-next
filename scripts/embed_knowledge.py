#!/usr/bin/env python3
"""Embed knowledge units into their pgvector column (#357 slice 3).

Runs AFTER an extraction (e.g. ``scripts/extract_chat_knowledge.py``) has populated
``knowledge_units`` (embedding starts NULL). Uses the REAL local embedding model, so it needs the ML
deps and the DGX box — never CI. By default only fills NULL embeddings; ``--all``
re-embeds every unit (e.g. after a re-extraction changed the text).

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
        help="re-embed every unit, not just those with a NULL embedding",
    )
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    url = args.db_url.replace("postgresql://", "postgresql+psycopg://", 1)
    os.environ["TEKIJIN_DATABASE_URL"] = url
    sys.path.insert(0, SRC)

    from tekijin.config import get_settings
    from tekijin.data.db import get_engine, get_sessionmaker, session_scope
    from tekijin.knowledge.index import embed_knowledge_units
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
        embedded = embed_knowledge_units(
            session, embedder, only_missing=args.only_missing, batch_size=args.batch_size
        )
    print(f"embedded knowledge units: {embedded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

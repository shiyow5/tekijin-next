#!/usr/bin/env python3
"""Run every question of the fictional circle through the real graph (#108).

Builds the graph the API builds for a member's question (the configured LLM nodes,
``ChunkSearch``, ``DbExpertFinder``, a ``MemorySaver``) over the circle the database
holds (the organization with the circle's ``GUILD_ID``), asks the questions one by
one, and prints per-kind counts: refused, follow-up pending, hand-off pending,
grounded answer, additive answer, correct citation, expected phrases, Hit@3 of the
correct members, and the routes. ``--rows`` also prints one line per question.

Read only: the session's transaction is ``READ ONLY`` and is rolled back at the
end, so a write anywhere in the run fails instead of landing in the database.

The LLM backend comes from the environment. Without ``TEKIJIN_LLM_BACKEND=vllm`` the
keyword stubs answer, and the first line of the output says which one ran.
Example (evo-x2, llama-server)::

    PYTHONPATH=backend/src HF_HUB_OFFLINE=1 \\
      TEKIJIN_LLM_BACKEND=vllm TEKIJIN_LLM_BASE_URL=http://127.0.0.1:8080/v1 \\
      TEKIJIN_LLM_MODEL=qwen3.6-35b-a3b TEKIJIN_LLM_STRUCTURED_OUTPUT_METHOD=json_schema \\
      TEKIJIN_LLM_THINKING_CONTROL=chat_template \\
      python scripts/circle_eval.py --rows --json out.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys

from tekijin.config import get_settings
from tekijin.data.circle import load_circle
from tekijin.data.db import get_engine, get_sessionmaker
from tekijin.eval.circle import (
    build_circle_agent,
    circle_ids,
    circle_org_id,
    format_rows,
    format_summary,
    read_only,
    run_circle,
    summarize,
)
from tekijin.llm.factory import make_llm_nodes
from tekijin.retrieval.embedding import SentenceTransformerEmbedder

# The day the questions are asked. The circle's data is dated autumn 2026.
NOW = dt.datetime(2026, 11, 2, 12, 0, 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", action="store_true", help="print one line per question")
    parser.add_argument("--json", metavar="PATH", help="write every row as JSON")
    args = parser.parse_args(argv)

    settings = get_settings()
    nodes = make_llm_nodes(settings)
    print(f"llm_backend={settings.llm_backend} nodes={[type(n).__name__ for n in nodes]}")
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        use_e5_prefix=settings.embedding_use_e5_prefix,
        query_prefix=settings.embedding_query_prefix,
        passage_prefix=settings.embedding_passage_prefix,
        trust_remote_code=settings.embedding_trust_remote_code,
        revision=settings.embedding_model_revision,
        app_env=settings.app_env,
    )
    circle = load_circle()
    with get_sessionmaker(get_engine(settings.database_url))() as session:
        read_only(session)
        try:
            org_id = circle_org_id(session)
            ids = circle_ids(session, circle, org_id)
            agent = build_circle_agent(session, embedder, settings, nodes)
            rows = run_circle(agent, circle, org_id, ids, now=NOW)
        finally:
            session.rollback()

    print(format_summary(summarize(rows)))
    if args.rows:
        print(format_rows(rows))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as out:
            json.dump([dataclasses.asdict(r) for r in rows], out, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

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

Knowledge (#113): ``--knowledge on`` wires the knowledge answer (verified knowledge,
floor ``--knowledge-floor``, default the configured one) and ``--knowledge both``
runs the questions twice, off then on. ``--floor-sweep`` prints, for each floor,
how many questions the nearest verified knowledge would take (no graph run).
Building that knowledge is the one step that writes, and only to the circle's
organization: ``--build-knowledge`` deletes the circle's knowledge, drafts it again
from the circle's documents and solved conversations with the configured LLM,
marks the drafts verified (an admin's step in real use) and embeds them. It needs
a real LLM backend and runs before the questions, in its own transaction::

    python scripts/circle_eval.py --build-knowledge --floor-sweep --knowledge both --rows

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
    RowResult,
    build_circle_agent,
    circle_ids,
    circle_org_id,
    format_rows,
    format_summary,
    read_only,
    run_circle,
    summarize,
)
from tekijin.eval.circle_knowledge import (
    as_dict,
    build_circle_knowledge,
    floor_sweep,
    format_build,
    format_sweep,
    reset_circle_knowledge,
)
from tekijin.knowledge.documents import DocumentDraftExtractor
from tekijin.knowledge.extract import DraftExtractor
from tekijin.llm.factory import make_llm_nodes
from tekijin.retrieval.embedding import SentenceTransformerEmbedder

# The day the questions are asked. The circle's data is dated autumn 2026.
NOW = dt.datetime(2026, 11, 2, 12, 0, 0)


def _build(session_factory, embedder, settings) -> dict:
    if settings.llm_backend == "stub":
        raise SystemExit("--build-knowledge needs a real LLM backend (TEKIJIN_LLM_BACKEND=vllm)")
    with session_factory() as session:
        removed = reset_circle_knowledge(session)
        report = build_circle_knowledge(
            session,
            embedder,
            case_extractor=DraftExtractor(settings=settings),
            document_extractor=DocumentDraftExtractor(settings=settings),
            now=NOW,
        )
        session.commit()
    print(f"knowledge rebuilt (removed {removed}): {format_build(report)}")
    return as_dict(report)


def _run(session_factory, embedder, settings, nodes, circle, floor) -> list[RowResult]:
    with session_factory() as session:
        read_only(session)
        try:
            org_id = circle_org_id(session)
            ids = circle_ids(session, circle, org_id)
            agent = build_circle_agent(session, embedder, settings, nodes, knowledge_floor=floor)
            return run_circle(agent, circle, org_id, ids, now=NOW)
        finally:
            session.rollback()


def _sweep(session_factory, embedder, circle) -> list:
    with session_factory() as session:
        read_only(session)
        try:
            ids = circle_ids(session, circle, circle_org_id(session))
            return floor_sweep(session, embedder, circle, ids)
        finally:
            session.rollback()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", action="store_true", help="print one line per question")
    parser.add_argument("--json", metavar="PATH", help="write every row as JSON")
    parser.add_argument(
        "--knowledge",
        choices=("off", "on", "both"),
        default="off",
        help="answer from verified knowledge (#113)",
    )
    parser.add_argument(
        "--knowledge-floor", type=float, default=None, help="the knowledge answer's floor"
    )
    parser.add_argument(
        "--build-knowledge",
        action="store_true",
        help="WRITES: rebuild and verify the circle's knowledge with the LLM first",
    )
    parser.add_argument(
        "--floor-sweep", action="store_true", help="print what each floor would take"
    )
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
    session_factory = get_sessionmaker(get_engine(settings.database_url))
    out: dict = {}
    if args.build_knowledge:
        out["build"] = _build(session_factory, embedder, settings)
    if args.floor_sweep:
        sweep = _sweep(session_factory, embedder, circle)
        print(format_sweep(sweep))
        out["sweep"] = [dataclasses.asdict(r) for r in sweep]

    floor = (
        args.knowledge_floor
        if args.knowledge_floor is not None
        else settings.knowledge_answer_min_similarity
    )
    modes = {"off": [None], "on": [floor], "both": [None, floor]}[args.knowledge]
    for mode_floor in modes:
        label = "knowledge=off" if mode_floor is None else f"knowledge=on floor={mode_floor}"
        rows = _run(session_factory, embedder, settings, nodes, circle, mode_floor)
        print(label)
        print(format_summary(summarize(rows)))
        if args.rows:
            print(format_rows(rows))
        out[label] = [dataclasses.asdict(r) for r in rows]
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            payload = (
                out
                if (args.build_knowledge or args.floor_sweep or len(modes) > 1)
                else (out[next(iter(k for k in out if k.startswith("knowledge=")))])
            )
            json.dump(payload, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

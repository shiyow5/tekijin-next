"""Seed the database from the synthetic JSON fixtures.

Idempotent: every table is truncated (``RESTART IDENTITY CASCADE``) before rows
are re-inserted, so running the seed repeatedly converges on the same state.
Embeddings are left ``NULL`` (a later ingestion component fills them in).

CLI::

    python -m tekijin.data.seed

Prints per-table insert counts to stdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from tekijin.auth.passwords import hash_password
from tekijin.config import get_settings
from tekijin.data.db import (
    Base,
    create_all,
    ensure_pgvector,
    get_engine,
    get_sessionmaker,
    session_scope,
)
from tekijin.data.mappers import build_all

# Order in which to TRUNCATE — children before parents. CASCADE makes the exact
# order forgiving, but listing children first keeps intent clear.
#
# SECURITY: this tuple is the ONLY source of table names interpolated into the
# TRUNCATE statement. It is a hard-coded allow-list — never build it from user
# input, request data, or any external source, since the names are spliced into
# raw SQL (identifiers cannot be passed as bind parameters).
_TRUNCATE_ORDER: tuple[str, ...] = (
    "knowledge_units",
    "evidence",
    "person_topic_edges",
    "events",
    "eval_runs",
    # #247: runtime-only (no fixture writes it), FKs questions + employees.
    "offline_consults",
    "messages",
    "recommendations",
    "answers",
    "questions",
    "documents",
    "daily_reports",
    "ai_chat_history",
    "employee_chat_history",
    "project_members",
    "skills",
    "certifications",
    "employee_profiles",
    "projects",
    "employees",
)


def truncate_all(session: Session) -> None:
    """Empty every table and reset identity sequences."""

    tables = ", ".join(_TRUNCATE_ORDER)
    session.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


def seed_session(session: Session, fixtures_dir: Path) -> dict[str, int]:
    """Truncate and re-insert every fixture group. Returns per-table counts."""

    truncate_all(session)
    counts: dict[str, int] = {}
    for logical_name, rows in build_all(fixtures_dir):
        session.add_all(rows)
        # Flush per group so FK-dependent inserts see their parents.
        session.flush()
        counts[logical_name] = len(rows)
    _seed_passwords(session)
    return counts


def _seed_passwords(session: Session) -> None:
    """Give every seeded employee the demo login password (#241).

    All employees share one demo password (the data is synthetic and the password
    is documented), so a single PBKDF2 hash is computed once and applied to the
    whole roster. The admin is NOT an employee, so it is untouched here — it
    authenticates against ``settings.admin_password`` directly.
    """

    encoded = hash_password(get_settings().demo_user_password)
    session.execute(text("UPDATE employees SET password_hash = :hash"), {"hash": encoded})


def apply_migrations(
    database_url: str | None = None,
    *,
    engine: Engine | None = None,
) -> None:
    """Bring the schema up to date WITHOUT touching data (non-destructive).

    Ensures the ``vector`` extension, creates any missing tables (``create_all``
    never drops or truncates), and runs the idempotent additive DDL in
    :func:`_apply_schema_upgrades`. Existing rows are retained. NOTE: an embedding
    column widened by a dimension change is reset to ``NULL`` (a model change
    invalidates old vectors); recompute with ``make embed``.

    This is the safe deploy-time path when a database already holds data — unlike
    :func:`run_seed`, which truncates and reloads fixtures. Exposed as a CLI via
    ``python -m tekijin.data.migrate``.
    """

    eng = engine if engine is not None else get_engine(database_url)
    ensure_pgvector(eng)
    create_all(eng)
    _apply_schema_upgrades(eng)
    # Repairs a database seeded before the realignment above existed.
    realign_identity_sequences(eng)


def run_seed(
    database_url: str | None = None,
    fixtures_dir: Path | None = None,
    *,
    engine: Engine | None = None,
) -> dict[str, int]:
    """Full seed pipeline: migrate schema, then TRUNCATE and load fixtures.

    DESTRUCTIVE: :func:`seed_session` truncates every table before re-inserting.
    For a data-preserving schema update use :func:`apply_migrations` instead.

    Pass ``engine`` to reuse an existing engine (preferred by callers that
    already hold one — a live ``Engine`` carries the real password, whereas
    ``str(engine.url)`` masks it and would produce a broken URL). Otherwise an
    engine is built from ``database_url`` (or the configured default).
    """

    settings = get_settings()
    eng = engine if engine is not None else get_engine(database_url)
    fixtures = fixtures_dir or settings.fixtures_dir

    apply_migrations(engine=eng)

    factory = get_sessionmaker(eng)
    with session_scope(factory) as session:
        counts = seed_session(session, fixtures)
    # AFTER the fixtures, not before: the rows above are inserted with explicit
    # ids, which does not advance the identity sequence. Leaving it behind means
    # the next insert that omits an id is handed one that already exists (on the
    # live database the sequence sat at 1 against a roster of 40 — so the first
    # auto-assigned employee would have collided with employee 1).
    realign_identity_sequences(eng)
    return counts


def _apply_schema_upgrades(engine: Engine) -> None:
    """Additive, idempotent DDL for columns ``create_all`` cannot add or change.

    The repo has no migration tool — ``create_all`` creates missing TABLES but
    never ALTERs an existing one. So a database seeded before a column was added
    (e.g. the Docker Compose persistent volume) would be missing it and break at
    runtime. ``ADD COLUMN IF NOT EXISTS`` makes re-seeding pick up new columns;
    new tables (e.g. ``eval_runs``) are handled by ``create_all`` itself.

    Column TYPE changes are likewise invisible to ``create_all``. When the
    embedding model changed (#63: e5-large 1024-d → Nemotron-3-Embed-1B 2048-d),
    an existing DB keeps its ``vector(1024)`` columns, which would reject the new
    2048-d vectors at ``make embed`` time. The guarded block below widens each
    embedding column to ``vector(2048)`` only when it is not already that width,
    dropping any stale (wrong-model) embeddings with ``USING NULL`` — they are
    recomputed by ``make embed``. Idempotent: a fresh ``create_all`` already
    builds ``vector(2048)`` so the block is a no-op there (no table rewrite).
    """

    # The target width is derived from the single source of truth
    # (``settings.embedding_dim``, which also drives ``Vector(EMBEDDING_DIM)`` in
    # models/tables.py) so a future dim change updates ``create_all`` and this
    # migration together. It is an internal ``int`` (never external input), so
    # interpolating it into the SQL is safe.
    dim = get_settings().embedding_dim
    with engine.begin() as conn:
        # Login password hash (#241). Older DBs (persistent volume) get it here;
        # a fresh create_all already builds it. NULL until _seed_passwords runs.
        conn.execute(
            text("ALTER TABLE employees ADD COLUMN IF NOT EXISTS password_hash VARCHAR(255)")
        )
        # #506: existing rows must come out ACTIVE, so the default is on the
        # column itself rather than applied only to new inserts.
        conn.execute(
            text(
                "ALTER TABLE employees ADD COLUMN IF NOT EXISTS "
                "is_active BOOLEAN NOT NULL DEFAULT true"
            )
        )
        # #66: members created from Discord have no email, and carry their
        # organization and their join / opt-out state.
        conn.execute(text("ALTER TABLE employees ALTER COLUMN email DROP NOT NULL"))
        conn.execute(
            text(
                "ALTER TABLE employees ADD COLUMN IF NOT EXISTS "
                "org_id INTEGER REFERENCES organizations(id)"
            )
        )
        conn.execute(text("ALTER TABLE employees ADD COLUMN IF NOT EXISTS joined_at TIMESTAMP"))
        conn.execute(text("ALTER TABLE employees ADD COLUMN IF NOT EXISTS opted_out_at TIMESTAMP"))
        conn.execute(text("ALTER TABLE questions ADD COLUMN IF NOT EXISTS route VARCHAR(32)"))
        # Self-resolution UX signal (#159): the asker marked the question solved
        # without a live hand-off. Older DBs get it here; fresh ones via create_all.
        conn.execute(
            text("ALTER TABLE questions ADD COLUMN IF NOT EXISTS resolution_kind VARCHAR(16)")
        )
        # The graph thread_id for an API-created question (responder inbox → handoff
        # deep link). Added for #123; older DBs get it here, fresh ones via create_all.
        conn.execute(text("ALTER TABLE questions ADD COLUMN IF NOT EXISTS session_id VARCHAR(64)"))
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_questions_session_id ON questions (session_id)")
        )
        # Runtime resolution timestamp (#97): so avg resolution time counts live
        # accepts / self-resolutions, not only seeded answers rows.
        conn.execute(text("ALTER TABLE questions ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMP"))
        # When the asker acknowledged a decline notification (#E7). ``IF EXISTS``
        # on the table (not just the column) because a from-scratch schema
        # upgrade test exercises this function against a minimal table set that
        # does not include ``recommendations``.
        conn.execute(
            text(
                "ALTER TABLE IF EXISTS recommendations "
                "ADD COLUMN IF NOT EXISTS declined_seen_at TIMESTAMP"
            )
        )
        # #509: when the asker/responder acknowledged an accepted/incoming-request
        # notification. Same "IF EXISTS on the table" guard as declined_seen_at above.
        conn.execute(
            text(
                "ALTER TABLE IF EXISTS recommendations "
                "ADD COLUMN IF NOT EXISTS accepted_seen_at TIMESTAMP"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE IF EXISTS recommendations "
                "ADD COLUMN IF NOT EXISTS request_seen_at TIMESTAMP"
            )
        )
        # #518: the employee a responder named on a 'referred' outcome. Nullable;
        # older DBs (persistent volume) get it here, fresh ones via create_all.
        conn.execute(
            text(
                "ALTER TABLE IF EXISTS recommendations "
                "ADD COLUMN IF NOT EXISTS referred_to_employee_id INTEGER "
                "REFERENCES employees(id)"
            )
        )
        # Indexed to match the table's other FK-to-employees columns (create_all builds
        # it on a fresh DB; a live DB gets it here). Guarded on the table existing —
        # `CREATE INDEX IF NOT EXISTS` only guards the INDEX name, not the table, so a
        # minimal DB without `recommendations` (the schema-upgrade test) would error.
        conn.execute(
            text(
                "DO $$ BEGIN\n"
                "  IF to_regclass('recommendations') IS NOT NULL THEN\n"
                "    CREATE INDEX IF NOT EXISTS ix_recommendations_referred_to_employee_id\n"
                "      ON recommendations (referred_to_employee_id);\n"
                "  END IF;\n"
                "END $$;"
            )
        )
        # Asker's chosen consultation method ("direct" | "chat"); NULL treated
        # as "chat" everywhere.
        conn.execute(
            text("ALTER TABLE questions ADD COLUMN IF NOT EXISTS consult_method VARCHAR(32)")
        )
        # #355: precomputed daily-report topics for the C6 scorer. daily_reports is
        # a pre-existing table, so create_all() never adds this column to a live DB
        # — reseeding would then insert DailyReport(topics=...) and 500. ADD IF NOT
        # EXISTS makes an old DGX DB pick it up on the next deploy/reseed.
        conn.execute(text("ALTER TABLE daily_reports ADD COLUMN IF NOT EXISTS topics TEXT[]"))
        # #433: daily-report embedding so a report can be a searchable knowledge
        # source for System 1. daily_reports pre-exists, so create_all never adds
        # this to a live DB — ADD IF NOT EXISTS gives an old DGX DB the column on
        # the next deploy/reseed; ``make embed`` then fills it. The width is fixed
        # up by the widen block below if an older narrower vector already exists.
        conn.execute(
            text(f"ALTER TABLE daily_reports ADD COLUMN IF NOT EXISTS embedding vector({dim})")
        )
        # #451: the chat-extraction batch (knowledge/chat.py) reads the ENTIRE
        # history ordered by (channel, sent_at, id), filtered to rows that actually
        # carry a message and a timestamp. Without a matching index that is a Seq
        # Scan plus an external merge sort on every run, growing with the real
        # corpus. Partial, so it stays proportional to the rows the batch can use
        # rather than to the whole log. `create_all` builds the table but not this
        # index, so a live DB only ever gets it here.
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chat_history_channel_sent_at_id "
                "ON employee_chat_history (channel, sent_at, id) "
                "WHERE message IS NOT NULL AND sent_at IS NOT NULL"
            )
        )
        # Widen embedding columns to the current dim when an older DB is narrower.
        # Table/column names are a hard-coded allow-list spliced via format() —
        # never build them from external input (identifiers can't be bound).
        conn.execute(
            text(
                "DO $$\n"
                "DECLARE\n"
                "  tbl text;\n"
                "  cur text;\n"
                "BEGIN\n"
                "  FOREACH tbl IN ARRAY ARRAY['employee_profiles','questions',"
                "'answers','documents','daily_reports'] LOOP\n"
                "    SELECT format_type(atttypid, atttypmod) INTO cur\n"
                "      FROM pg_attribute\n"
                "      WHERE attrelid = tbl::regclass AND attname = 'embedding'\n"
                "        AND attnum > 0 AND NOT attisdropped;\n"
                f"    IF cur IS DISTINCT FROM 'vector({dim})' THEN\n"
                "      EXECUTE format(\n"
                f"        'ALTER TABLE %I ALTER COLUMN embedding TYPE vector({dim}) USING NULL',\n"
                "        tbl);\n"
                "    END IF;\n"
                "  END LOOP;\n"
                "END $$;"
            )
        )


def _format_counts(counts: dict[str, int]) -> str:
    lines = [f"  {name:<18} {count:>6}" for name, count in counts.items()]
    total = sum(counts.values())
    lines.append(f"  {'TOTAL':<18} {total:>6}")
    return "\n".join(lines)


def main() -> int:
    """CLI entry point. Seeds the configured database and prints counts."""

    # ``Base`` is referenced so the metadata is registered even if models were
    # not imported elsewhere in this process. Use an explicit raise (not
    # ``assert``) so the guard survives ``python -O``.
    if not Base.metadata.tables:
        raise RuntimeError("no tables registered on Base.metadata")
    counts = run_seed()
    print("Seeded TEKIJIN fixtures:")
    print(_format_counts(counts))
    return 0


def realign_identity_sequences(engine: Engine) -> None:
    """Move each identity sequence past the largest id its table already holds.

    Rows loaded from fixtures carry explicit ids, and an explicit id does not
    advance the sequence — so straight after a seed the sequence still points at
    1 while the table holds 40 rows. The next insert that lets the database pick
    an id then collides with an existing row.

    Doing this ONCE, here, is what lets every insert afterwards simply use the
    sequence. Realigning inside each insert instead is not concurrency-safe: two
    overlapping transactions read the same ``max(id)``, both ``setval`` to it,
    and both receive the same value — reproduced, both were handed 41. Left
    alone, ``nextval`` is atomic and two callers cannot collide.

    Idempotent, and a no-op on an empty table.
    """

    with engine.begin() as conn:
        for table in ("employees",):
            conn.execute(
                text(
                    "SELECT setval("
                    "  pg_get_serial_sequence(:table, 'id'),"
                    "  COALESCE((SELECT MAX(id) FROM employees), 1),"
                    "  (SELECT COUNT(*) FROM employees) > 0"
                    ")"
                ),
                {"table": table},
            )


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    sys.exit(main())

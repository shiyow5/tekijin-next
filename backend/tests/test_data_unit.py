"""Database-free unit tests for the data layer.

Covers value parsing, fixture loading, dict->ORM mappers, and DTO converters.
Building ORM instances needs no database connection, so these run everywhere.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import pytest
import sqlalchemy
from sqlalchemy.orm import sessionmaker

from tekijin.config import Settings, get_settings
from tekijin.data import db as dbmod
from tekijin.data.dto import (
    AnswerDTO,
    DocumentDTO,
    EmployeeDTO,
    ProfileDTO,
    QuestionDTO,
)
from tekijin.data.loaders import (
    FIXTURE_FILES,
    load_fixture,
    load_json,
    parse_date,
    parse_datetime,
)
from tekijin.data.mappers import (
    build_all,
    map_answer,
    map_document,
    map_employee,
    map_profile,
    map_question,
)
from tekijin.data.seed import _format_counts
from tekijin.models import tables as t

# Expected fixture row counts (from the synthetic dataset).
# NOTE: #51/#52 で合成データにトピック差・部署横断メンバーを入れた際、
# skills だけ 58 -> 61 に変わった（他は不変）。fixtures を再生成したらここも更新すること。
EXPECTED_COUNTS = {
    "employees": 40,
    "projects": 120,
    "profiles": 40,
    "certifications": 98,
    "skills": 61,
    "project_members": 237,
    "employee_chat": 2000,
    "daily_reports": 3070,
    "questions": 150,
    "answers": 150,
    "documents": 36,  # #296: 型番/製品文書6件を追加（30 -> 36）
}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_embedding_dim_default(monkeypatch) -> None:
    # Isolate from ambient TEKIJIN_* env and any local .env so this asserts the
    # real code default (2048, Nemotron-3-Embed-1B), not the developer's
    # TEKIJIN_EMBEDDING_DIM.
    for key in list(os.environ):
        if key.startswith("TEKIJIN_"):
            monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.embedding_dim == 2048


def test_embedding_dim_wired_into_tables() -> None:
    # The pgvector column width is derived from settings, so the module constant
    # must track whatever value was in effect at import time.
    assert get_settings().embedding_dim == t.EMBEDDING_DIM


# --------------------------------------------------------------------------- #
# loaders / parsers
# --------------------------------------------------------------------------- #
def test_parse_date() -> None:
    assert parse_date("2026-04-01") == dt.date(2026, 4, 1)
    assert parse_date(None) is None
    assert parse_date("") is None


def test_parse_datetime_variants() -> None:
    assert parse_datetime("2026-05-22T09:00:00") == dt.datetime(2026, 5, 22, 9, 0, 0)
    # date-only strings (documents.updated_at) become midnight datetimes.
    assert parse_datetime("2026-03-30") == dt.datetime(2026, 3, 30, 0, 0, 0)
    assert parse_datetime(None) is None
    assert parse_datetime("") is None


def test_load_json_rejects_non_list(tmp_path) -> None:
    p = tmp_path / "obj.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected a JSON array"):
        load_json(p)


def test_load_fixture_unknown_name() -> None:
    with pytest.raises(KeyError):
        load_fixture(get_settings().fixtures_dir, "nope")


def test_all_fixtures_load() -> None:
    fixtures_dir = get_settings().fixtures_dir
    for name in FIXTURE_FILES:
        rows = load_fixture(fixtures_dir, name)
        assert isinstance(rows, list) and rows


# --------------------------------------------------------------------------- #
# #326: 営業部日報の CRS 訪問日報フォーマット
# --------------------------------------------------------------------------- #
def _sales_and_other_reports() -> tuple[list[dict], list[dict]]:
    fixtures_dir = get_settings().fixtures_dir
    employees = load_fixture(fixtures_dir, "employees")
    daily = load_fixture(fixtures_dir, "daily_reports")
    sales_ids = {e["id"] for e in employees if e["department"] == "営業部"}
    assert sales_ids, "営業部の社員が存在する前提"
    sales = [d for d in daily if d["employee_id"] in sales_ids]
    other = [d for d in daily if d["employee_id"] not in sales_ids]
    assert sales and other
    return sales, other


def test_sales_daily_reports_use_crs_visit_format() -> None:
    """営業部の日報はすべて CRS 訪問日報フォーマット、他部署は不変（#326）。"""
    sales, other = _sales_and_other_reports()
    for d in sales:
        c = d["content"]
        assert c.startswith("【訪問】"), c
        assert "要件:" in c and "所要" in c and "当社:" in c, c
        # 粒度は 1〜2 文（実態）。見出し＋詳細で過剰に長くならない。
        assert len(c) <= 160, c
        assert d["issue"], "営業日報は課題タグを持つ"
    # 他部署の日報は CRS 見出しを持たない（不変性の担保）。
    assert not any(d["content"].startswith("【訪問】") for d in other)


def test_daily_reports_have_precomputed_topics() -> None:
    """#355: daily reports carry a precomputed ``topics`` list for scorer evidence."""
    daily = load_fixture(get_settings().fixtures_dir, "daily_reports")
    assert all("topics" in d and isinstance(d["topics"], list) for d in daily)
    # Some (not all) reports are on-topic; sales reports (#326) always are.
    assert any(d["topics"] for d in daily)


def test_sales_daily_reports_carry_topic_keywords() -> None:
    """CRS 日報がトピック語を含み、証拠被覆に寄与する（#326・ADR-0006）。"""
    sales, _ = _sales_and_other_reports()
    # build_fixtures の TOPICS 語彙（SALES_TOPIC_SCRIPT の課題句/商材句が含む）。
    keys = (
        "CRM",
        "顧客情報",
        "営業活動",
        "業務効率化",
        "手作業",
        "業務フロー",
        "契約",
        "基幹システム",
        "システム間連携",
        "老朽化",
        "ネットワーク",
        "VPN",
        "接続トラブル",
        "セキュリティ",
        "UTM",
        "脆弱性",
        "クラウド",
        "データ基盤",
    )
    assert all(any(k in d["content"] for k in keys) for d in sales)


# --------------------------------------------------------------------------- #
# mappers
# --------------------------------------------------------------------------- #
def test_map_employee_full() -> None:
    row = map_employee(
        {
            "id": 1,
            "name": "田中 太郎",
            "email": "t@example.com",
            "department": "営業部",
            "section": "第1課",
            "position": "部長",
            "branch": "大阪",
            "role": "営業",
            "hire_date": "2011-04-01",
            "department_history": [{"department": "営業部"}],
        }
    )
    assert row.id == 1
    assert row.hire_date == dt.date(2011, 4, 1)
    assert row.department_history == [{"department": "営業部"}]


def test_mappers_leave_embeddings_none() -> None:
    prof = map_profile({"employee_id": 1, "description": "x", "updated_at": None})
    q = map_question({"id": "q1", "asker_id": 1, "body": "b", "topics": ["a"]})
    ans = map_answer({"id": "a1", "question_id": "q1", "responder_id": 2, "body": "b"})
    doc = map_document({"id": "d1", "title": "t", "body": "b", "source": "s"})
    assert prof.embedding is None
    assert q.embedding is None and q.topics == ["a"]
    assert ans.embedding is None
    assert doc.embedding is None


def test_build_all_counts_and_order() -> None:
    groups = build_all(get_settings().fixtures_dir)
    names = [name for name, _ in groups]
    # employees and projects must come before their FK dependants.
    assert names.index("employees") < names.index("profiles")
    assert names.index("projects") < names.index("project_members")
    assert names.index("questions") < names.index("answers")
    counts = {name: len(rows) for name, rows in groups}
    assert counts == EXPECTED_COUNTS


# --------------------------------------------------------------------------- #
# DTO converters (build ORM instances directly, no DB)
# --------------------------------------------------------------------------- #
def test_employee_dto_from_row() -> None:
    row = t.Employee(
        id=7,
        name="A",
        email="a@x",
        department="d",
        section="s",
        position="p",
        branch="b",
        role="r",
        hire_date=dt.date(2020, 1, 1),
    )
    dto = EmployeeDTO.from_row(row)
    assert dto.id == 7 and dto.branch == "b" and dto.hire_date == dt.date(2020, 1, 1)


def test_profile_and_question_has_embedding_flag() -> None:
    prof = ProfileDTO.from_row(t.EmployeeProfile(employee_id=1, description="d", embedding=None))
    assert prof.has_embedding is False
    q = QuestionDTO.from_row(
        t.Question(id="q1", asker_id=1, body="b", topics=["x"], embedding=[0.1] * 3)
    )
    assert q.has_embedding is True and q.topics == ("x",)
    # topics=None -> empty tuple
    q2 = QuestionDTO.from_row(t.Question(id="q2", asker_id=1, body="b", topics=None))
    assert q2.topics == ()


def test_answer_and_document_dtos() -> None:
    ans = AnswerDTO.from_row(
        t.Answer(
            id="a1",
            question_id="q1",
            responder_id=3,
            body="b",
            topic="T",
            reuse_count=5,
            was_helpful=True,
            embedding=None,
        )
    )
    assert ans.topic == "T" and ans.reuse_count == 5 and ans.has_embedding is False
    doc = DocumentDTO.from_row(
        t.Document(id="d1", title="t", body="b", source="s", embedding=[0.0] * 3)
    )
    assert doc.has_embedding is True


def test_knowledge_unit_dto_from_row() -> None:
    from tekijin.data.dto import KnowledgeUnitDTO

    row = t.KnowledgeUnit(
        id=3,
        kind="case",
        problem="CRM 導入が進まない",
        action="SFA/CRM を提案",
        result="受注",
        topics=["CRM・営業支援"],
        industry="製造業",
        source_type="slack_thread",
        source_id="1234",
        confidence=0.8,
        review_status="approved",
        embedding=None,
    )
    dto = KnowledgeUnitDTO.from_row(row)
    assert dto.id == 3 and dto.kind == "case"
    assert dto.topics == ("CRM・営業支援",)
    assert dto.source_type == "slack_thread" and dto.source_id == "1234"
    assert dto.review_status == "approved" and dto.has_embedding is False
    # topics=None -> empty tuple (mirrors the other DTOs)
    bare = KnowledgeUnitDTO.from_row(
        t.KnowledgeUnit(
            id=4,
            kind="case",
            source_type="answer",
            source_id="a_1",
            review_status="unreviewed",
            topics=None,
            embedding=[0.0] * 3,
        )
    )
    assert bare.topics == () and bare.has_embedding is True


def test_knowledge_write_validation_is_database_free() -> None:
    """Boundary validation raises before any SQL, so it needs no DB (#357)."""
    from tekijin.data import knowledge as kb

    # A bad kind / missing provenance is a clean ValueError, not a CheckViolation.
    with pytest.raises(ValueError, match="unknown knowledge-unit kind"):
        kb.upsert_knowledge_unit(
            session=None, kind="bogus", problem="p", action="a", source_type="x", source_id="1"
        )
    with pytest.raises(ValueError, match="provenance"):
        kb.upsert_knowledge_unit(
            session=None, kind="case", problem="p", action="a", source_type="", source_id="1"
        )
    with pytest.raises(ValueError, match="unknown review status"):
        kb.set_review_status(None, 1, "bogus")
    # The read helpers validate review_status before touching the session too.
    with pytest.raises(ValueError, match="unknown review status"):
        kb.list_knowledge_units(None, review_status="bogus")
    with pytest.raises(ValueError, match="unknown review status"):
        kb.knowledge_units_by_topics(None, ["CRM・営業支援"], review_status="bogus")
    assert set(kb.VALID_KINDS) == {"case", "procedure", "decision"}
    assert set(kb.VALID_REVIEW_STATUSES) == {"unreviewed", "approved", "rejected"}


# --------------------------------------------------------------------------- #
# db wiring (no connection opened)
# --------------------------------------------------------------------------- #
def test_get_engine_and_sessionmaker_no_connect() -> None:
    eng = dbmod.get_engine("postgresql+psycopg://u:p@localhost:5432/db")
    assert isinstance(eng, sqlalchemy.Engine)
    factory = dbmod.get_sessionmaker(eng)
    assert isinstance(factory, sessionmaker)


def test_format_counts() -> None:
    out = _format_counts({"employees": 40, "projects": 120})
    assert "employees" in out and "TOTAL" in out and "160" in out


def test_main_prints_counts(monkeypatch, capsys) -> None:
    import tekijin.data.seed as seed_mod

    monkeypatch.setattr(seed_mod, "run_seed", lambda: {"employees": 40})
    rc = seed_mod.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "Seeded TEKIJIN fixtures" in out and "employees" in out


def test_main_raises_when_no_tables_registered(monkeypatch) -> None:
    from sqlalchemy import MetaData
    from sqlalchemy.orm import DeclarativeBase

    import tekijin.data.seed as seed_mod

    class _EmptyBase(DeclarativeBase):
        metadata = MetaData()  # no tables registered

    monkeypatch.setattr(seed_mod, "Base", _EmptyBase)
    with pytest.raises(RuntimeError, match="no tables registered"):
        seed_mod.main()

"""Pure fixture-dict -> ORM-instance mappers.

Each ``map_*`` function turns one raw fixture record into an unsaved ORM
instance. Building an ORM object requires no database connection, so these are
unit-tested directly. Embeddings are always left ``None`` (computed later at
ingestion time). ``build_all`` assembles every table's rows in FK-safe order.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import DeclarativeBase

from tekijin.data.loaders import load_fixture, parse_date, parse_datetime
from tekijin.models import tables as t

# A mapper turns one raw fixture record into an unsaved ORM instance.
Mapper = Callable[[dict[str, Any]], DeclarativeBase]


def map_employee(r: dict[str, Any]) -> t.Employee:
    return t.Employee(
        id=r["id"],
        name=r["name"],
        email=r["email"],
        department=r.get("department"),
        section=r.get("section"),
        position=r.get("position"),
        role=r.get("role"),
        hire_date=parse_date(r.get("hire_date")),
        department_history=r.get("department_history"),
    )


def map_profile(r: dict[str, Any]) -> t.EmployeeProfile:
    return t.EmployeeProfile(
        employee_id=r["employee_id"],
        description=r.get("description"),
        embedding=None,
        updated_at=parse_datetime(r.get("updated_at")),
    )


def map_employee_chat(r: dict[str, Any]) -> t.EmployeeChatHistory:
    return t.EmployeeChatHistory(
        id=r["id"],
        sender_employee_id=r["sender_employee_id"],
        receiver_employee_id=r.get("receiver_employee_id"),
        channel=r.get("channel"),
        message=r.get("message"),
        sent_at=parse_datetime(r.get("sent_at")),
    )


def map_question(r: dict[str, Any]) -> t.Question:
    return t.Question(
        id=r["id"],
        asker_id=r["asker_id"],
        body=r.get("body"),
        topics=r.get("topics"),
        status=r.get("status"),
        created_at=parse_datetime(r.get("created_at")),
        embedding=None,
    )


def map_answer(r: dict[str, Any]) -> t.Answer:
    return t.Answer(
        id=r["id"],
        question_id=r["question_id"],
        responder_id=r["responder_id"],
        body=r.get("body"),
        created_at=parse_datetime(r.get("created_at")),
        embedding=None,
        reuse_count=r.get("reuse_count"),
        was_helpful=r.get("was_helpful"),
        topic=r.get("topic"),
    )


def map_document(r: dict[str, Any]) -> t.Document:
    return t.Document(
        id=r["id"],
        title=r.get("title"),
        body=r.get("body"),
        source=r.get("source"),
        updated_at=parse_datetime(r.get("updated_at")),
        embedding=None,
    )


# Ordered so that parents are inserted before their FK dependants.
_PLAN: tuple[tuple[str, str, Mapper], ...] = (
    ("employees", "employees", map_employee),
    ("profiles", "profiles", map_profile),
    ("employee_chat", "employee_chat", map_employee_chat),
    ("questions", "questions", map_question),
    ("answers", "answers", map_answer),
    ("documents", "documents", map_document),
)


def build_all(fixtures_dir: Path) -> list[tuple[str, list[DeclarativeBase]]]:
    """Load every fixture and map it to ORM rows, in FK-safe insert order.

    Returns a list of ``(logical_name, orm_rows)`` pairs so the seed can insert
    each group in order and report per-table counts.
    """

    result: list[tuple[str, list[DeclarativeBase]]] = []
    for logical_name, fixture_name, mapper in _PLAN:
        rows = [mapper(rec) for rec in load_fixture(fixtures_dir, fixture_name)]
        result.append((logical_name, rows))
    return result

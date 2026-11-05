"""Read-only data-transfer objects returned by the repository layer.

The repository never leaks live ORM instances to callers (search, scorers,
API). Instead it maps each row to one of these frozen dataclasses, so downstream
code gets immutable snapshots that are safe to pass around and cache. Every
``from_row`` builder is a pure function of an ORM instance and therefore
unit-testable without a database.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tekijin.models.tables import (
        Answer,
        Document,
        Employee,
        EmployeeProfile,
        OfflineConsult,
        Question,
    )


@dataclass(frozen=True, slots=True)
class EmployeeDTO:
    id: int
    name: str
    email: str | None
    department: str | None
    section: str | None
    position: str | None
    role: str | None
    hire_date: dt.date | None

    @classmethod
    def from_row(cls, row: Employee) -> EmployeeDTO:
        return cls(
            id=row.id,
            name=row.name,
            email=row.email,
            department=row.department,
            section=row.section,
            position=row.position,
            role=row.role,
            hire_date=row.hire_date,
        )


@dataclass(frozen=True, slots=True)
class ProfileDTO:
    employee_id: int
    description: str | None
    has_embedding: bool
    updated_at: dt.datetime | None

    @classmethod
    def from_row(cls, row: EmployeeProfile) -> ProfileDTO:
        return cls(
            employee_id=row.employee_id,
            description=row.description,
            has_embedding=row.embedding is not None,
            updated_at=row.updated_at,
        )


@dataclass(frozen=True, slots=True)
class OfflineConsultDTO:
    """A 直接相談 retrospective as topic evidence (#247).

    ``topics`` are validated against ``TOPIC_VOCABULARY`` at the API boundary, so
    the scorer joins on them directly.
    ``resolution`` decides whether this counts at all — see
    ``collect_topic_evidence``.
    """

    id: int
    responder_id: int
    topics: tuple[str, ...]
    resolution: str
    created_at: dt.datetime | None

    @classmethod
    def from_row(cls, row: OfflineConsult) -> OfflineConsultDTO:
        return cls(
            id=row.id,
            responder_id=row.responder_id,
            topics=tuple(row.topics or ()),
            resolution=row.resolution,
            created_at=row.created_at,
        )


@dataclass(frozen=True, slots=True)
class QuestionDTO:
    id: str
    asker_id: int
    body: str | None
    topics: tuple[str, ...]
    status: str | None
    created_at: dt.datetime | None
    has_embedding: bool

    @classmethod
    def from_row(cls, row: Question) -> QuestionDTO:
        return cls(
            id=row.id,
            asker_id=row.asker_id,
            body=row.body,
            topics=tuple(row.topics or ()),
            status=row.status,
            created_at=row.created_at,
            has_embedding=row.embedding is not None,
        )


@dataclass(frozen=True, slots=True)
class AnswerDTO:
    id: str
    question_id: str
    responder_id: int
    body: str | None
    topic: str | None
    reuse_count: int | None
    was_helpful: bool | None
    created_at: dt.datetime | None
    has_embedding: bool

    @classmethod
    def from_row(cls, row: Answer) -> AnswerDTO:
        return cls(
            id=row.id,
            question_id=row.question_id,
            responder_id=row.responder_id,
            body=row.body,
            topic=row.topic,
            reuse_count=row.reuse_count,
            was_helpful=row.was_helpful,
            created_at=row.created_at,
            has_embedding=row.embedding is not None,
        )


@dataclass(frozen=True, slots=True)
class DocumentDTO:
    id: str
    title: str | None
    body: str | None
    source: str | None
    updated_at: dt.datetime | None
    has_embedding: bool

    @classmethod
    def from_row(cls, row: Document) -> DocumentDTO:
        return cls(
            id=row.id,
            title=row.title,
            body=row.body,
            source=row.source,
            updated_at=row.updated_at,
            has_embedding=row.embedding is not None,
        )

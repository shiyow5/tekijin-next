"""Tests for the knowledge-unit extraction pipeline (#357 slice 2).

Unit tests drive :class:`CaseExtractor` with a deterministic fake model (no
network); integration tests exercise the DB reads and idempotent storage against
the seeded PostgreSQL fixtures.
"""

from __future__ import annotations

import pytest

from tekijin.data.knowledge import get_knowledge_unit_by_source, list_knowledge_units
from tekijin.knowledge.extract import (
    CaseExtractor,
    ExtractionSource,
    extract_and_store,
)
from tekijin.llm.schemas import CaseExtractionSchema


class _FakeModel:
    """Stands in for a ``with_structured_output``-bound runnable in tests."""

    def __init__(self, schema: CaseExtractionSchema | None) -> None:
        self._schema = schema
        self.calls: list[list[tuple[str, str]]] = []

    def invoke(self, prompt: list[tuple[str, str]]) -> CaseExtractionSchema | None:
        self.calls.append(prompt)
        return self._schema


def _src(source_id: str = "1", text: str = "CRM 導入相談") -> ExtractionSource:
    return ExtractionSource(
        source_type="slack_thread",
        source_id=source_id,
        text=text,
        topics=("CRM・営業支援",),
    )


# --------------------------------------------------------------------------- #
# schema validation (no DB, no network)
# --------------------------------------------------------------------------- #
def test_case_schema_extractable_requires_problem_and_action() -> None:
    # A pass (extractable=false) needs nothing.
    ok = CaseExtractionSchema(extractable=False)
    assert ok.extractable is False and ok.result is None
    # extractable=true with both fields is valid.
    full = CaseExtractionSchema(
        extractable=True, problem="CRM が根付かない", action="SFA/CRM を提案", result="受注"
    )
    assert full.action == "SFA/CRM を提案"
    # extractable=true missing problem or action raises.
    with pytest.raises(ValueError, match="problem is required"):
        CaseExtractionSchema(extractable=True, problem="  ", action="a")
    with pytest.raises(ValueError, match="action is required"):
        CaseExtractionSchema(extractable=True, problem="p", action="")


# --------------------------------------------------------------------------- #
# extractor (fake model)
# --------------------------------------------------------------------------- #
def test_extractor_returns_none_for_non_case() -> None:
    fake = _FakeModel(CaseExtractionSchema(extractable=False))
    extractor = CaseExtractor(model=fake)
    assert extractor.extract(_src()) is None
    # The prompt carries the system instruction + the record text.
    assert fake.calls and fake.calls[0][0][0] == "system"
    assert fake.calls[0][1] == ("human", "CRM 導入相談")


def test_extractor_raises_on_empty_tool_call() -> None:
    # An empty structured output (no tool call) is a hard failure, NOT a benign skip
    # — it matches the other vLLM adapters and is distinguishable from extractable=false.
    extractor = CaseExtractor(model=_FakeModel(None))
    with pytest.raises(ValueError, match="structured output was empty"):
        extractor.extract(_src())


def test_extractor_returns_case_when_extractable() -> None:
    schema = CaseExtractionSchema(
        extractable=True,
        problem="CRM が定着しない",
        action="SFA/CRM を提案",
        result="受注",
        industry="製造業",
        confidence=0.9,
    )
    extractor = CaseExtractor(model=_FakeModel(schema))
    out = extractor.extract(_src())
    assert out is not None and out.problem == "CRM が定着しない" and out.confidence == 0.9


# --------------------------------------------------------------------------- #
# storage (DB) — provenance-faithful topics, skip, idempotent
# --------------------------------------------------------------------------- #
def test_extract_and_store_counts_and_topics(seed_counts, session) -> None:
    schema = CaseExtractionSchema(
        extractable=True,
        problem="p",
        action="a",
        result="受注",
        industry="小売業",  # the model's industry is kept...
        confidence=0.8,
    )
    extractor = CaseExtractor(model=_FakeModel(schema))
    sources = [
        ExtractionSource("slack_thread", "80001", "text1", ("CRM・営業支援",)),
        ExtractionSource("slack_thread", "80002", "text2", ("ネットワーク・VPN",)),
    ]
    counts = extract_and_store(session, sources, extractor)
    assert counts == {"seen": 2, "stored": 2, "skipped": 0, "errored": 0}

    unit = get_knowledge_unit_by_source(session, "slack_thread", "80001")
    assert unit is not None
    assert unit.kind == "case" and unit.result == "受注" and unit.industry == "小売業"
    # ...but topics come from the SOURCE record, never the model (vocabulary lock).
    assert unit.topics == ("CRM・営業支援",)
    assert unit.review_status == "unreviewed"  # awaits human review (#354)


def test_extract_and_store_skips_non_cases(seed_counts, session) -> None:
    extractor = CaseExtractor(model=_FakeModel(CaseExtractionSchema(extractable=False)))
    counts = extract_and_store(session, [_src("80101"), _src("80102")], extractor)
    assert counts == {"seen": 2, "stored": 0, "skipped": 2, "errored": 0}
    assert get_knowledge_unit_by_source(session, "slack_thread", "80101") is None


class _RaisingModel:
    """A model whose invoke raises, to exercise per-source error isolation."""

    def invoke(self, prompt: list[tuple[str, str]]) -> CaseExtractionSchema:
        raise RuntimeError("simulated LLM/client failure")


def test_extract_and_store_isolates_per_source_errors(seed_counts, session) -> None:
    # A failing source is counted as errored and skipped; the batch continues and the
    # good source is still stored (no whole-run rollback). Empty tool call also errors.
    good = CaseExtractor(
        model=_FakeModel(CaseExtractionSchema(extractable=True, problem="p", action="a"))
    )
    bad = CaseExtractor(model=_RaisingModel())
    empty = CaseExtractor(model=_FakeModel(None))

    # Mixed batch via a tiny dispatcher extractor.
    class _Dispatch:
        def extract(self, source: ExtractionSource):
            if source.source_id == "80301":
                return good.extract(source)
            if source.source_id == "80302":
                return bad.extract(source)
            return empty.extract(source)  # 80303 → raises (empty tool call)

    counts = extract_and_store(session, [_src("80301"), _src("80302"), _src("80303")], _Dispatch())
    assert counts == {"seen": 3, "stored": 1, "skipped": 0, "errored": 2}
    assert get_knowledge_unit_by_source(session, "slack_thread", "80301") is not None
    assert get_knowledge_unit_by_source(session, "slack_thread", "80302") is None


def test_extract_and_store_is_idempotent(seed_counts, session) -> None:
    schema = CaseExtractionSchema(extractable=True, problem="p", action="a", confidence=0.5)
    extractor = CaseExtractor(model=_FakeModel(schema))
    sources = [_src("80201"), _src("80202")]
    before = len(list_knowledge_units(session))
    extract_and_store(session, sources, extractor)
    session.flush()
    after_first = len(list_knowledge_units(session))
    extract_and_store(session, sources, extractor)  # re-run same sources
    session.flush()
    after_second = len(list_knowledge_units(session))
    assert after_first - before == 2
    assert after_second == after_first  # upsert, not duplicate


# --------------------------------------------------------------------------- #
# chat extraction (#448): topics inferred from LLM hints, snapped to vocabulary
# --------------------------------------------------------------------------- #
def _chat_src(source_id: str = "chat_IT問い合わせ_1") -> ExtractionSource:
    # A tag-less chat conversation: topics come from the model's hints, not source.
    return ExtractionSource(
        source_type="chat",
        source_id=source_id,
        text="社員5: VPNが繋がりません\n社員9: クライアント証明書を入れ直してください",
        topics=(),
    )


def test_chat_prompt_uses_the_chat_system_instruction() -> None:
    # A chat source gets the conversation prompt (asks for topic_hints).
    fake = _FakeModel(CaseExtractionSchema(extractable=False))
    extractor = CaseExtractor(model=fake)
    extractor.extract(_chat_src())
    system_text = fake.calls[0][0][1]
    assert "会話" in system_text and "topic_hints" in system_text


def test_chat_extract_infers_and_normalizes_topics(seed_counts, session) -> None:
    # The model proposes free-text hints; the caller snaps them to the canonical
    # vocabulary ("VPN" -> "ネットワーク・VPN") and never lets the model mint a topic.
    schema = CaseExtractionSchema(
        extractable=True,
        problem="VPNが繋がらない",
        action="クライアント証明書を再インストール",
        result="解決",
        topic_hints=["VPN", "存在しないトピック"],
        confidence=0.7,
    )
    extractor = CaseExtractor(model=_FakeModel(schema))
    counts = extract_and_store(
        session, [_chat_src("chat_IT問い合わせ_9001")], extractor, infer_topics_from_hints=True
    )
    assert counts == {"seen": 1, "stored": 1, "skipped": 0, "errored": 0}
    unit = get_knowledge_unit_by_source(session, "chat", "chat_IT問い合わせ_9001")
    assert unit is not None
    assert unit.topics == ("ネットワーク・VPN",)  # normalized; off-vocab hint dropped


def test_chat_extract_skips_case_with_no_routable_topic(seed_counts, session) -> None:
    # A real case whose hints all fall off the vocabulary is not storable (untopiced
    # units match no evidence). It is skipped, not stored with empty topics.
    schema = CaseExtractionSchema(
        extractable=True,
        problem="p",
        action="a",
        topic_hints=["まったく無関係な語"],
    )
    extractor = CaseExtractor(model=_FakeModel(schema))
    counts = extract_and_store(
        session, [_chat_src("chat_x_9002")], extractor, infer_topics_from_hints=True
    )
    assert counts == {"seen": 1, "stored": 0, "skipped": 1, "errored": 0}
    assert get_knowledge_unit_by_source(session, "chat", "chat_x_9002") is None


def test_tagged_extract_ignores_topic_hints(seed_counts, session) -> None:
    # Even if the model response carried hints, a tagged source keeps its own
    # tags — the vocabulary lock for tag-bearing sources is unchanged (#357).
    schema = CaseExtractionSchema(
        extractable=True, problem="p", action="a", topic_hints=["セキュリティ"]
    )
    extractor = CaseExtractor(model=_FakeModel(schema))
    src = ExtractionSource("slack_thread", "80777", "text", ("CRM・営業支援",))
    extract_and_store(session, [src], extractor, infer_topics_from_hints=True)
    unit = get_knowledge_unit_by_source(session, "slack_thread", "80777")
    assert unit is not None and unit.topics == ("CRM・営業支援",)

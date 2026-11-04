"""Unit tests for #69 fragment-text retrieval (retrieve-then-classify context).

``collect_context_fragments`` re-hydrates the text of C4's top hits (which the
``RetrievalResult`` carries only as ids + scores) so C1 can classify a question's
topic with the retrieved evidence in front of it. Pure read path over a small
repository protocol — tested here with an in-memory fake, no database.
"""

from __future__ import annotations

from tekijin.agent.state import empty_retrieval
from tekijin.data.dto import AnswerDTO, DocumentDTO, QuestionDTO
from tekijin.retrieval.fragments import collect_cited_evidence, collect_context_fragments


def _answer(answer_id: str, question_id: str, body: str) -> AnswerDTO:
    return AnswerDTO(
        id=answer_id,
        question_id=question_id,
        responder_id=1,
        body=body,
        topic=None,
        reuse_count=None,
        was_helpful=None,
        created_at=None,
        has_embedding=False,
    )


def _question(question_id: str, body: str) -> QuestionDTO:
    return QuestionDTO(
        id=question_id,
        asker_id=1,
        body=body,
        topics=(),
        status=None,
        created_at=None,
        has_embedding=False,
    )


def _document(doc_id: str, title: str, body: str) -> DocumentDTO:
    return DocumentDTO(
        id=doc_id,
        title=title,
        body=body,
        source=None,
        updated_at=None,
        has_embedding=False,
    )


class _FakeSource:
    """Minimal stand-in for the repository's id-keyed batch lookups."""

    def __init__(self, *, answers=(), questions=(), documents=()) -> None:
        self._answers = {a.id: a for a in answers}
        self._questions = {q.id: q for q in questions}
        self._documents = {d.id: d for d in documents}

    def answers_by_ids(self, ids):
        return {i: self._answers[i] for i in ids if i in self._answers}

    def questions_by_ids(self, ids):
        return {i: self._questions[i] for i in ids if i in self._questions}

    def documents_by_ids(self, ids):
        return {i: self._documents[i] for i in ids if i in self._documents}


def test_empty_retrieval_yields_no_fragments() -> None:
    assert collect_context_fragments(_FakeSource(), empty_retrieval()) == []


def test_combines_past_answer_and_document_text() -> None:
    source = _FakeSource(
        answers=[_answer("a1", "q1", "UTMのファーム更新は保守時間内に実施します")],
        questions=[_question("q1", "UTMのファームウェア更新手順を教えて")],
        documents=[_document("d1", "セキュリティ運用手順", "ファイアウォールの設定変更フロー")],
    )
    retrieval = empty_retrieval()
    retrieval["past_answers"] = [{"qa_id": "a1", "score": 0.9, "responder_id": 1}]
    retrieval["documents"] = [{"doc_id": "d1", "score": 0.7}]

    fragments = collect_context_fragments(source, retrieval)

    joined = "\n".join(fragments)
    # The QA fragment carries BOTH the linked question and the answer body (the
    # question is where the topic-bearing vocabulary usually lives).
    assert "UTMのファームウェア更新手順を教えて" in joined
    assert "UTMのファーム更新は保守時間内に実施します" in joined
    # The document fragment carries title + body.
    assert "セキュリティ運用手順" in joined
    assert "ファイアウォールの設定変更フロー" in joined


def test_skips_missing_ids() -> None:
    source = _FakeSource(answers=[_answer("a1", "q1", "本文")], questions=[_question("q1", "質問")])
    retrieval = empty_retrieval()
    retrieval["past_answers"] = [
        {"qa_id": "a1", "score": 0.9, "responder_id": 1},
        {"qa_id": "missing", "score": 0.8, "responder_id": 2},
    ]
    retrieval["documents"] = [{"doc_id": "nope", "score": 0.5}]

    fragments = collect_context_fragments(source, retrieval)

    assert len(fragments) == 1
    assert "質問" in fragments[0]


def test_clips_long_fragments() -> None:
    long_body = "あ" * 500
    source = _FakeSource(
        answers=[_answer("a1", "q1", long_body)], questions=[_question("q1", "短い質問")]
    )
    retrieval = empty_retrieval()
    retrieval["past_answers"] = [{"qa_id": "a1", "score": 0.9, "responder_id": 1}]

    fragments = collect_context_fragments(source, retrieval, max_chars=80)

    assert len(fragments[0]) <= 80
    assert fragments[0].endswith("…")


def test_caps_fragment_count() -> None:
    answers = [_answer(f"a{i}", f"q{i}", f"回答{i}") for i in range(10)]
    questions = [_question(f"q{i}", f"質問{i}") for i in range(10)]
    source = _FakeSource(answers=answers, questions=questions)
    retrieval = empty_retrieval()
    retrieval["past_answers"] = [
        {"qa_id": f"a{i}", "score": 1.0 - i * 0.01, "responder_id": 1} for i in range(10)
    ]

    fragments = collect_context_fragments(source, retrieval, max_fragments=3)

    assert len(fragments) == 3
    # Highest-scoring answers survive the cap (a0/a1/a2, in channel order).
    assert "質問0" in fragments[0]


def test_answer_without_linked_question_uses_answer_body_only() -> None:
    # The QA's question id resolves to nothing -> the answer body still stands in.
    source = _FakeSource(answers=[_answer("a1", "missing-q", "回答本文のみ")])
    retrieval = empty_retrieval()
    retrieval["past_answers"] = [{"qa_id": "a1", "score": 0.9, "responder_id": 1}]

    fragments = collect_context_fragments(source, retrieval)

    assert len(fragments) == 1
    assert "回答本文のみ" in fragments[0]


def test_document_with_only_a_title_is_kept() -> None:
    source = _FakeSource(documents=[_document("d1", "タイトルのみ", "")])
    retrieval = empty_retrieval()
    retrieval["documents"] = [{"doc_id": "d1", "score": 0.5}]

    fragments = collect_context_fragments(source, retrieval)

    assert len(fragments) == 1
    assert "タイトルのみ" in fragments[0]


def test_empty_bodied_rows_are_skipped() -> None:
    # A resolved row whose text is entirely empty contributes no fragment.
    source = _FakeSource(
        answers=[_answer("a1", "q1", "")],
        questions=[_question("q1", "")],
        documents=[_document("d1", "", "")],
    )
    retrieval = empty_retrieval()
    retrieval["past_answers"] = [{"qa_id": "a1", "score": 0.9, "responder_id": 1}]
    retrieval["documents"] = [{"doc_id": "d1", "score": 0.5}]

    assert collect_context_fragments(source, retrieval) == []


def test_interleaves_channels_so_both_are_represented() -> None:
    answers = [_answer(f"a{i}", f"q{i}", f"回答{i}") for i in range(5)]
    questions = [_question(f"q{i}", f"質問{i}") for i in range(5)]
    documents = [_document(f"d{i}", f"文書{i}", f"文書本文{i}") for i in range(5)]
    source = _FakeSource(answers=answers, questions=questions, documents=documents)
    retrieval = empty_retrieval()
    retrieval["past_answers"] = [
        {"qa_id": f"a{i}", "score": 1.0 - i * 0.01, "responder_id": 1} for i in range(5)
    ]
    retrieval["documents"] = [{"doc_id": f"d{i}", "score": 1.0 - i * 0.01} for i in range(5)]

    fragments = collect_context_fragments(source, retrieval, max_fragments=4)

    joined = "\n".join(fragments)
    # Round-robin (answer, doc, answer, doc) — a long answer list must not crowd
    # out the documents, mirroring _aggregate_people's interleave.
    assert any("質問" in f for f in fragments)
    assert any("文書" in f for f in fragments)
    assert "文書0" in joined


# --------------------------------------------------------------------------- #
# #291: collect_cited_evidence — id-paired evidence for a cited self-answer
# --------------------------------------------------------------------------- #
def test_cited_evidence_empty_retrieval() -> None:
    assert collect_cited_evidence(_FakeSource(), empty_retrieval()) == []


def test_cited_evidence_keeps_source_id_and_kind() -> None:
    source = _FakeSource(
        answers=[_answer("a1", "q1", "UTMのファーム更新は保守時間内に実施します")],
        questions=[_question("q1", "UTMのファームウェア更新手順を教えて")],
        documents=[_document("d1", "セキュリティ運用手順", "ファイアウォールの設定変更フロー")],
    )
    retrieval = empty_retrieval()
    retrieval["past_answers"] = [{"qa_id": "a1", "score": 0.9, "responder_id": 1}]
    retrieval["documents"] = [{"doc_id": "d1", "score": 0.7}]

    items = collect_cited_evidence(source, retrieval)

    # Round-robin: answer first, then document; each keeps its own id + kind.
    assert [(e.source_id, e.kind) for e in items] == [("a1", "qa"), ("d1", "document")]
    qa = items[0]
    assert "UTMのファームウェア更新手順を教えて" in qa.text  # question body
    assert "UTMのファーム更新は保守時間内に実施します" in qa.text  # answer body
    assert "セキュリティ運用手順" in items[1].text and "ファイアウォール" in items[1].text


def test_cited_evidence_skips_missing_and_empty() -> None:
    source = _FakeSource(
        answers=[_answer("a1", "q1", "本文"), _answer("a2", "q2", "   ")],  # a2 empty body
        questions=[_question("q1", "質問")],
    )
    retrieval = empty_retrieval()
    retrieval["past_answers"] = [
        {"qa_id": "a1", "score": 0.9, "responder_id": 1},
        {"qa_id": "a2", "score": 0.8, "responder_id": 2},  # empty -> skipped
        {"qa_id": "missing", "score": 0.7, "responder_id": 3},  # absent -> skipped
    ]
    items = collect_cited_evidence(source, retrieval)
    assert [e.source_id for e in items] == ["a1"]


def test_cited_evidence_clips_to_max_chars() -> None:
    source = _FakeSource(documents=[_document("d1", "T", "あ" * 1000)])
    retrieval = empty_retrieval()
    retrieval["documents"] = [{"doc_id": "d1", "score": 0.5}]
    (item,) = collect_cited_evidence(source, retrieval, max_chars=50)
    assert len(item.text) <= 50 and item.text.endswith("…")

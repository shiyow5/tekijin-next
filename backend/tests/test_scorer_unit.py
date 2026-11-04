"""Database-free unit tests for the C6 scorer's pure logic.

Covers evidence stacking -> edge weight, the time-decayed recency (with an
injected ``now``), the load window saturation, answer quality,
confidence derivation, and topic matching. Every value is deterministic.
"""

from __future__ import annotations

import datetime as dt

import pytest

from tekijin.data.dto import (
    AnswerDTO,
    CertificationDTO,
    OfflineConsultDTO,
    ProjectMembershipDTO,
    SkillDTO,
)
from tekijin.scorer import features
from tekijin.scorer.evidence import Evidence, collect_topic_evidence, edge_weight
from tekijin.scorer.scorer import ExpertiseScorer
from tekijin.scorer.topics import (
    PRODUCT_TOPIC_MAP,
    TOPIC_VOCABULARY,
    canonicalize_topic,
    cert_matches_topic,
    normalize_topics,
    product_matches_topic,
)
from tekijin.scorer.weights import (
    BASE_SCORE_CERTIFICATION,
    BASE_SCORE_HELPFUL_ANSWER,
    BASE_SCORE_OFFLINE_CONSULT,
    BASE_SCORE_PROJECT_LEAD,
    BASE_SCORE_PROJECT_MEMBER,
    BASE_SCORE_SKILL,
    OFFLINE_CONSULT_EVIDENCE_CAP,
)

NOW = dt.datetime(2026, 8, 21, 12, 0, 0)
TOPIC = "ネットワーク・VPN"


# --------------------------------------------------------------------------- #
# helpers to build DTOs
# --------------------------------------------------------------------------- #
def _cert(name: str, acquired: dt.date | None = None) -> CertificationDTO:
    return CertificationDTO(id="c", employee_id=1, name=name, acquired_at=acquired)


def _skill(topic: str, level: str | None = "中級", source: str | None = "self") -> SkillDTO:
    return SkillDTO(id="s", employee_id=1, topic=topic, level=level, source=source)


def _membership(product: str, role: str, end: dt.date | None = None) -> ProjectMembershipDTO:
    return ProjectMembershipDTO(
        project_id=1,
        employee_id=1,
        role=role,
        product=product,
        industry=None,
        subject=None,
        status="受注",
        start_date=dt.date(2026, 1, 1),
        end_date=end,
    )


def _answer(helpful: bool | None, reuse: int | None, created: dt.datetime) -> AnswerDTO:
    return AnswerDTO(
        id="a",
        question_id="q",
        responder_id=1,
        body="b",
        topic=TOPIC,
        reuse_count=reuse,
        was_helpful=helpful,
        created_at=created,
        has_embedding=False,
    )


def _consult(
    topics: tuple[str, ...],
    resolution: str = "resolved",
    created: dt.datetime | None = None,
) -> OfflineConsultDTO:
    return OfflineConsultDTO(
        id=1,
        responder_id=1,
        topics=topics,
        resolution=resolution,
        created_at=created or dt.datetime(2026, 1, 1),
    )


# --------------------------------------------------------------------------- #
# #247: offline (直接相談) retrospectives as evidence
# --------------------------------------------------------------------------- #
def test_collect_topic_evidence_offline_consult_default_is_empty() -> None:
    """No offline_consults arg -> no such evidence (pre-#247 behaviour)."""

    ev = collect_topic_evidence(TOPIC, [], [], [], [])
    assert not any(e.source_type == "offline_consult" for e in ev)


def test_collect_topic_evidence_includes_on_topic_offline_consult() -> None:
    ev = collect_topic_evidence(
        TOPIC, [], [], [], [], offline_consults=[_consult((TOPIC,)), _consult(("経理・決算",))]
    )
    consults = [e for e in ev if e.source_type == "offline_consult"]
    assert len(consults) == 1  # off-topic retrospective excluded
    assert consults[0].base_score == BASE_SCORE_OFFLINE_CONSULT


def test_offline_consult_is_weaker_than_a_self_declared_skill() -> None:
    """It is HEARSAY: the asker summarising what the responder said (#247).

    A self-declared skill is at least first-hand about oneself; a retrospective is
    one person's paraphrase of another's words. So it must sit below
    ``BASE_SCORE_SKILL``.
    """

    assert 0 < BASE_SCORE_OFFLINE_CONSULT < BASE_SCORE_SKILL


def test_unresolved_offline_consult_is_not_evidence() -> None:
    """「解決しなかった」は専門性の証拠にならない — but it is not NEGATIVE evidence.

    Same rule as a decline (db-schema.md: 断り≠非専門): a consultation that did not
    resolve simply contributes nothing. It must never subtract, or asking the wrong
    person once would damage their standing on a topic they do know.
    """

    ev = collect_topic_evidence(
        TOPIC, [], [], [], [], offline_consults=[_consult((TOPIC,), resolution="unresolved")]
    )
    assert not any(e.source_type == "offline_consult" for e in ev)
    assert edge_weight(ev) == 0.0


def test_partially_resolved_offline_consult_counts() -> None:
    ev = collect_topic_evidence(
        TOPIC, [], [], [], [], offline_consults=[_consult((TOPIC,), resolution="partial")]
    )
    assert sum(1 for e in ev if e.source_type == "offline_consult") == 1


def test_collect_topic_evidence_offline_consult_capped() -> None:
    """Volume must not saturate topic_fit."""

    consults = [_consult((TOPIC,)) for _ in range(OFFLINE_CONSULT_EVIDENCE_CAP + 4)]
    ev = collect_topic_evidence(TOPIC, [], [], [], [], offline_consults=consults)
    assert sum(1 for e in ev if e.source_type == "offline_consult") == OFFLINE_CONSULT_EVIDENCE_CAP


def test_collect_topic_evidence_offline_consult_unions_topics() -> None:
    ev = collect_topic_evidence(
        [TOPIC, "セキュリティ"], [], [], [], [], offline_consults=[_consult(("セキュリティ",))]
    )
    assert sum(1 for e in ev if e.source_type == "offline_consult") == 1


# --------------------------------------------------------------------------- #
# topic matching
# --------------------------------------------------------------------------- #
def test_cert_matches_topic_by_substring() -> None:
    assert cert_matches_topic("情報処理安全確保支援士", "セキュリティ") is True
    assert cert_matches_topic("ネットワークスペシャリスト", "ネットワーク・VPN") is True
    assert cert_matches_topic("日商簿記1級", "セキュリティ") is False
    assert cert_matches_topic(None, "セキュリティ") is False


def test_product_matches_topic() -> None:
    assert product_matches_topic("CRM導入支援", "CRM・営業支援") is True
    assert product_matches_topic("広告運用代行", "Webマーケティング・広告") is True
    assert product_matches_topic("CRM導入支援", "セキュリティ") is False
    assert product_matches_topic(None, "CRM・営業支援") is False


def test_topic_vocabulary_is_the_scorer_join_vocabulary() -> None:
    # 22 canonical topics, unique. Every CERT/PRODUCT map resolves INTO this set —
    # otherwise that evidence source could never match a C1 topic (#116).
    assert len(TOPIC_VOCABULARY) == 22
    assert len(set(TOPIC_VOCABULARY)) == 22
    vocab = set(TOPIC_VOCABULARY)
    assert set(PRODUCT_TOPIC_MAP.values()) <= vocab


def test_topic_vocabulary_matches_eval_topic_keywords() -> None:
    # The eval gold vocabulary (tests/eval_topic_keywords, formerly
    # scripts/build_eval_v2.TOPICS) and the runtime vocabulary MUST be the same
    # set, or the eval measures a different vocabulary than production routes on.
    from eval_topic_keywords import TOPIC_KEYWORDS

    assert set(TOPIC_KEYWORDS) == set(TOPIC_VOCABULARY)


def test_canonicalize_topic_exact_alias_and_substring() -> None:
    assert canonicalize_topic("セキュリティ") == "セキュリティ"  # exact
    assert canonicalize_topic("VPN") == "ネットワーク・VPN"  # alias
    assert canonicalize_topic("運用保守") == "サーバー・インフラ運用"  # alias (non-substring)
    assert canonicalize_topic("サーバー") == "サーバー・インフラ運用"  # unambiguous substring
    assert canonicalize_topic("値段交渉") is None  # unmappable -> dropped
    assert canonicalize_topic("  ") is None


def test_canonicalize_topic_drops_ambiguous_fragments() -> None:
    # "運用" is a substring of TWO canonical topics -> ambiguous -> dropped, not a
    # wrong guess. Same for "システム" (システム開発・API vs 基幹システム).
    assert canonicalize_topic("運用") is None
    assert canonicalize_topic("システム") is None
    # "IT" is a substring of ONLY 社内IT・ヘルプデスク, so the unambiguous-substring
    # rule would wrongly snap it there — too generic, so it is explicitly dropped.
    assert canonicalize_topic("IT") is None


def test_normalize_topics_merges_split_compounds_and_dedups() -> None:
    # The #116 failure mode: the model splits "購買・仕入れ" into words.
    assert normalize_topics(["購買", "仕入れ", "値段交渉", "取引先"]) == ["購買・仕入れ"]
    # Split "サーバー・インフラ運用" + an unmappable extra.
    assert normalize_topics(["サーバー", "インフラ", "老朽化"]) == ["サーバー・インフラ運用"]
    # Already-canonical topics pass through, order preserved, deduped.
    assert normalize_topics(["セキュリティ", "セキュリティ", "クラウド"]) == [
        "セキュリティ",
        "クラウド移行",
    ]
    assert normalize_topics([]) == []


# --------------------------------------------------------------------------- #
# evidence -> edge weight
# --------------------------------------------------------------------------- #
def test_collect_topic_evidence_assigns_base_scores() -> None:
    evidence = collect_topic_evidence(
        TOPIC,
        [_cert("ネットワークスペシャリスト"), _cert("日商簿記1級")],  # only the first matches
        [_skill(TOPIC), _skill("経理・決算")],  # only the first matches
        [_membership("保守運用サポート", "lead")],  # product -> サーバー…, NOT this topic
        [_answer(True, 3, NOW), _answer(False, 0, NOW)],
    )
    kinds = [(e.source_type, e.base_score) for e in evidence]
    assert ("cert", BASE_SCORE_CERTIFICATION) in kinds
    assert ("self", BASE_SCORE_SKILL) in kinds
    assert ("answer", BASE_SCORE_HELPFUL_ANSWER) in kinds
    # The non-matching cert / skill / project produced no evidence.
    assert sum(1 for e in evidence if e.source_type == "cert") == 1
    assert sum(1 for e in evidence if e.source_type == "self") == 1
    assert sum(1 for e in evidence if e.source_type == "project") == 0


def test_collect_topic_evidence_skill_provenance() -> None:
    # Inferred skills keep provenance ("inferred"); self-declared and null-source
    # skills both default to "self". base_score is 0.3 either way.
    inferred = collect_topic_evidence(TOPIC, [], [_skill(TOPIC, source="inferred")], [], [])
    declared = collect_topic_evidence(TOPIC, [], [_skill(TOPIC, source="self")], [], [])
    unknown = collect_topic_evidence(TOPIC, [], [_skill(TOPIC, source=None)], [], [])
    assert inferred[0].source_type == "inferred"
    assert declared[0].source_type == "self"
    assert unknown[0].source_type == "self"  # null source defaults to self-declared
    assert inferred[0].base_score == declared[0].base_score == BASE_SCORE_SKILL


def test_collect_topic_evidence_unions_multiple_topics() -> None:
    # Skills on either topic are both evidence; a skill on neither is excluded.
    topics = ["ネットワーク・VPN", "セキュリティ"]
    skills = [
        _skill("ネットワーク・VPN"),
        _skill("セキュリティ"),
        _skill("経理・決算"),  # neither topic
    ]
    evidence = collect_topic_evidence(topics, [], skills, [], [])
    self_topics = sorted(e.detail for e in evidence if e.source_type == "self")
    assert self_topics == ["セキュリティ", "ネットワーク・VPN"]  # detail carries skill topic


def test_collect_topic_evidence_single_and_multi_are_consistent() -> None:
    # A single-topic string and a one-element sequence behave identically.
    one = collect_topic_evidence(TOPIC, [], [_skill(TOPIC)], [], [])
    seq = collect_topic_evidence([TOPIC], [], [_skill(TOPIC)], [], [])
    assert [e.source_type for e in one] == [e.source_type for e in seq]


def test_collect_topic_evidence_project_role_base_scores() -> None:
    lead = collect_topic_evidence("CRM・営業支援", [], [], [_membership("CRM導入支援", "lead")], [])
    member = collect_topic_evidence(
        "CRM・営業支援", [], [], [_membership("CRM導入支援", "member")], []
    )
    assert lead[0].base_score == BASE_SCORE_PROJECT_LEAD
    assert member[0].base_score == BASE_SCORE_PROJECT_MEMBER


def test_edge_weight_saturates_and_is_monotonic() -> None:
    small = edge_weight([Evidence("self", BASE_SCORE_SKILL, None, "x")])
    big = edge_weight(
        [
            Evidence("answer", BASE_SCORE_HELPFUL_ANSWER, None, "x"),
            Evidence("cert", BASE_SCORE_CERTIFICATION, None, "x"),
            Evidence("project", BASE_SCORE_PROJECT_LEAD, None, "x"),
        ]
    )
    assert 0.0 < small < big < 1.0
    assert edge_weight([]) == 0.0


# --------------------------------------------------------------------------- #
# recency (now injected)
# --------------------------------------------------------------------------- #
def test_recency_half_life() -> None:
    six_months_ago = NOW - dt.timedelta(days=182)  # ~ half-life
    assert features.recency(NOW, [six_months_ago]) == pytest.approx(0.5, abs=0.02)


def test_recency_picks_most_recent_and_handles_empty() -> None:
    old = NOW - dt.timedelta(days=365)
    fresh = NOW - dt.timedelta(days=1)
    assert features.recency(NOW, [old, fresh]) > features.recency(NOW, [old])
    assert features.recency(NOW, []) == 0.0


def test_recency_accepts_date_and_clamps_future() -> None:
    # A plain date is treated as midnight; a future date does not exceed 1.0.
    assert features.recency(NOW, [dt.date(2026, 8, 21)]) == pytest.approx(1.0, abs=0.01)
    future = NOW + dt.timedelta(days=30)
    assert features.recency(NOW, [future]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# load / answer_quality / confidence
# --------------------------------------------------------------------------- #
def test_load_monotonic_and_half_saturation() -> None:
    assert features.load(0) == 0.0
    assert features.load(5) == pytest.approx(0.5)
    assert features.load(20) > features.load(5)
    # Realistic weekly counts stay below the cap (extreme counts float-saturate).
    assert features.load(15) < 1.0


def test_answer_quality_monotonic() -> None:
    none = features.answer_quality(0, 0, 0)
    some = features.answer_quality(0, 0, 2)
    helpful = features.answer_quality(2, 0, 2)
    reused = features.answer_quality(2, 10, 2)
    assert none == 0.0
    assert 0.0 < some < helpful < reused < 1.0


def test_confidence_label_by_evidence_kind() -> None:
    # 高: answered on the topic and not merely project-adjacent (#110).
    assert features.confidence_label(["answer"]) == "高"
    assert features.confidence_label(["cert", "self", "answer"]) == "高"
    # 中: answered, but also carries project membership (diluting signal, #110).
    assert features.confidence_label(["answer", "project"]) == "中"
    assert features.confidence_label(["answer", "answer", "project"]) == "中"
    # 低: no past answer — certs/skills/projects alone don't prove answering here.
    assert features.confidence_label(["cert", "project", "self"]) == "低"
    assert features.confidence_label([]) == "低"


# --------------------------------------------------------------------------- #
# reason-detail helpers (static, DB-free)
# --------------------------------------------------------------------------- #
def test_recency_detail_empty_and_months() -> None:
    assert ExpertiseScorer._recency_detail([], NOW) == ""
    # Skills carry no timestamp -> still empty.
    assert ExpertiseScorer._recency_detail([Evidence("self", 0.3, None, "x")], NOW) == ""
    recent = ExpertiseScorer._recency_detail(
        [Evidence("answer", 0.7, NOW - dt.timedelta(days=5), "x")], NOW
    )
    assert recent == "直近1か月以内の関連実績あり"
    older = ExpertiseScorer._recency_detail(
        [Evidence("project", 0.8, NOW - dt.timedelta(days=95), "x")], NOW
    )
    assert older == "直近の関連実績: 約3か月前"


def test_load_label_bands() -> None:
    assert ExpertiseScorer._load_label(0) == "少なめ"
    assert ExpertiseScorer._load_label(3) == "やや多め"
    assert ExpertiseScorer._load_label(9) == "多め"


def test_recency_moments_excludes_certifications() -> None:
    # A cert carries a date but must not feed recency (it does not age out).
    evidence = [
        Evidence("cert", 0.6, dt.date(2020, 1, 1), "old cert"),
        Evidence("answer", 0.7, NOW, "fresh answer"),
        Evidence("self", 0.3, None, "skill"),
    ]
    moments = ExpertiseScorer._recency_moments(evidence, NOW)
    assert moments == [NOW]  # only the answer's timestamp


def test_recency_moments_ongoing_project_uses_now() -> None:
    # A finished project uses its end_date; an ongoing one (timestamp None) is
    # treated as current work -> ``now``, so it stays fresh.
    ended = Evidence("project", 0.8, dt.date(2026, 1, 1), "ended")
    ongoing = Evidence("project", 0.8, None, "ongoing")
    assert ExpertiseScorer._recency_moments([ongoing], NOW) == [NOW]
    moments = ExpertiseScorer._recency_moments([ended, ongoing], NOW)
    assert moments == [dt.date(2026, 1, 1), NOW]
    # An ongoing project scores as fresh as an answer submitted right now.
    assert features.recency(NOW, ExpertiseScorer._recency_moments([ongoing], NOW)) == 1.0


# --------------------------------------------------------------------------- #
# rank input validation (DB-free: raised before any DB access)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("top_k", [0, -1, -5])
def test_rank_rejects_nonpositive_top_k(top_k: int) -> None:
    from tekijin.data.repository import Repository

    scorer = ExpertiseScorer(Repository(None))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top_k must be positive"):
        scorer.rank(TOPIC, [1], asker_id=None, now=NOW, top_k=top_k)


def test_rank_rejects_aware_now() -> None:
    from tekijin.data.repository import Repository

    scorer = ExpertiseScorer(Repository(None))  # type: ignore[arg-type]
    aware = dt.datetime(2026, 8, 21, 12, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))
    with pytest.raises(ValueError, match="now must be naive"):
        scorer.rank(TOPIC, [1], asker_id=None, now=aware, top_k=3)


def test_positive_weight_sum_matches_frontend_fit_ceiling() -> None:
    """The frontend fit gauge (#240) normalises the composite score against the
    sum of the POSITIVE scorer weights, hardcoded as BASE_POSITIVE_WEIGHTS = 0.8 in
    `frontend/src/lib/fit.ts`. There is no runtime cross-language link, so guard it
    here: if the weights change, this fails and flags that fit.ts must be updated
    too (else the gauge silently mis-normalises)."""
    from tekijin.scorer.weights import DEFAULT_WEIGHTS as w

    positive_sum = w.topic_fit + w.recency + w.answer_quality
    assert positive_sum == pytest.approx(0.8), (
        f"positive weight sum {positive_sum} != 0.8; update BASE_POSITIVE_WEIGHTS in "
        "frontend/src/lib/fit.ts to match."
    )

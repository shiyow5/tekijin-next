"""Topic matching for evidence sources that lack a clean topic field.

``skills.topic`` and ``answers.topic`` already equal a question topic verbatim,
so they need no mapping. Certifications (free-text names) and projects (a
``product`` label) do. These lookup tables make that correspondence explicit,
deterministic, and tunable — like the weights, they are data, not logic, and can
be extended or overridden without changing the scorer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

# The canonical topic vocabulary: the 22 topics the WHOLE system agrees on. The
# scorer joins on these verbatim (``skills.topic`` / ``answers.topic`` /
# ``questions.topics`` all use them, and the CERT/PRODUCT maps below resolve to
# them), so a topic that is not one of these matches NO evidence and makes the
# recommendation random (#116). This is the single source of truth C1 must emit;
# ``tests/eval_topic_keywords.TOPIC_KEYWORDS`` (formerly
# ``scripts/build_eval_v2.TOPICS``) mirrors it and a test cross-checks both
# against the DB fixtures so the list cannot drift.
TOPIC_VOCABULARY: tuple[str, ...] = (
    "CRM・営業支援",
    "ECサイト構築",
    "SNS運用",
    "Webマーケティング・広告",
    "クラウド移行",
    "サーバー・インフラ運用",
    "システム開発・API",
    "セキュリティ",
    "データ基盤・分析",
    "ネットワーク・VPN",
    "パフォーマンスチューニング",
    "モバイルアプリ開発",
    "人事・採用",
    "問い合わせ・ヘルプデスク運用",
    "基幹システム",
    "契約管理",
    "広報・PR",
    "業務効率化コンサル",
    "社内IT・ヘルプデスク",
    "経理・決算",
    "総務・法務",
    "購買・仕入れ",
)

_VOCAB_SET = frozenset(TOPIC_VOCABULARY)

# Aliases for free-text C1 outputs that substring matching cannot resolve on its
# own — either because the word is not a substring of the canonical name
# ("運用保守"), or because it is a bare synonym ("VPN"). Deliberately EXCLUDES
# ambiguous fragments (e.g. bare "ヘルプデスク" / "運用" / "システム" match two
# canonical topics) so they fall through to "drop" rather than a wrong guess.
_TOPIC_ALIASES: dict[str, str] = {
    "運用保守": "サーバー・インフラ運用",
    "インフラ運用": "サーバー・インフラ運用",
    "インフラ": "サーバー・インフラ運用",
    "ネットワーク": "ネットワーク・VPN",
    "VPN": "ネットワーク・VPN",
    "API": "システム開発・API",
    "システム開発": "システム開発・API",
    "EC": "ECサイト構築",
    "ECサイト": "ECサイト構築",
    "CRM": "CRM・営業支援",
    "営業支援": "CRM・営業支援",
    "SNS": "SNS運用",
    "広告": "Webマーケティング・広告",
    "Webマーケティング": "Webマーケティング・広告",
    "マーケティング": "Webマーケティング・広告",
    "クラウド": "クラウド移行",
    "データ基盤": "データ基盤・分析",
    "データ分析": "データ基盤・分析",
    "モバイルアプリ": "モバイルアプリ開発",
    "人事": "人事・採用",
    "採用": "人事・採用",
    "問い合わせ": "問い合わせ・ヘルプデスク運用",
    "社内IT": "社内IT・ヘルプデスク",
    "契約": "契約管理",
    "広報": "広報・PR",
    "PR": "広報・PR",
    "業務効率化": "業務効率化コンサル",
    "経理": "経理・決算",
    "決算": "経理・決算",
    "総務": "総務・法務",
    "法務": "総務・法務",
    "購買": "購買・仕入れ",
    "仕入れ": "購買・仕入れ",
}


# Generic fragments that HAPPEN to be a substring of exactly one canonical name
# but are far too broad to route on it. "IT" is only in "社内IT・ヘルプデスク", so the
# unambiguous-substring rule would wrongly snap a bare "IT" there (#116 review);
# drop it instead so only an explicit alias/exact hit resolves such tokens.
_SUBSTRING_DROP: frozenset[str] = frozenset({"IT"})


def canonicalize_topic(raw: str) -> str | None:
    """Snap one (possibly free-text) topic to a canonical topic, or ``None``.

    Order: exact vocabulary hit → explicit alias → UNAMBIGUOUS substring match
    (the raw is contained in exactly one canonical name, or vice-versa). Anything
    that maps to zero or multiple canonical topics returns ``None`` (dropped) — a
    wrong topic is worse than a missing one, since it routes to the wrong experts.
    """

    r = raw.strip()
    if not r:
        return None
    if r in _VOCAB_SET:
        return r
    if r in _TOPIC_ALIASES:
        return _TOPIC_ALIASES[r]
    if r in _SUBSTRING_DROP:
        return None
    matches = {c for c in TOPIC_VOCABULARY if r in c or c in r}
    if len(matches) == 1:
        return next(iter(matches))
    return None


def normalize_topics(raw_topics: Iterable[str]) -> list[str]:
    """Map C1's (free-text) topics onto the canonical vocabulary, de-duplicated.

    C1 tends to split compound topic names into words ("購買", "仕入れ" for
    "購買・仕入れ") or use synonyms, but the scorer matches topics by exact
    equality — so un-normalized topics make every evidence source miss (#116).
    This collapses variants to canonical names and drops the unmappable, keeping
    first-seen order.
    """

    out: list[str] = []
    for raw in raw_topics:
        canonical = canonicalize_topic(raw)
        if canonical is not None and canonical not in out:
            out.append(canonical)
    return out


# Certification name -> topics it is evidence for, matched by substring so name
# variants ("...スペシャリスト", suffixes) still resolve. Topics absent here simply
# receive no certification evidence (skills / answers / projects still apply).
CERT_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ネットワーク・VPN": ("ネットワーク",),
    "セキュリティ": ("セキュリティ", "安全確保支援士"),
    "データ基盤・分析": ("データベース", "E資格", "G検定", "データ"),
    "クラウド移行": ("AWS", "クラウド"),
    "サーバー・インフラ運用": ("LPIC", "Linux", "インフラ"),
    "システム開発・API": ("基本情報", "応用情報"),
    "契約管理": ("法務",),
    "人事・採用": ("社会保険労務士", "給与計算", "衛生管理"),
    "経理・決算": ("簿記",),
    "CRM・営業支援": ("販売士", "中小企業診断士"),
}

# Project ``product`` -> the single question topic it evidences.
PRODUCT_TOPIC_MAP: dict[str, str] = {
    "CRM導入支援": "CRM・営業支援",
    "ECサイト構築": "ECサイト構築",
    "SNS運用代行": "SNS運用",
    "Webマーケティング支援": "Webマーケティング・広告",
    "広告運用代行": "Webマーケティング・広告",
    "クラウド移行支援": "クラウド移行",
    "データ基盤構築": "データ基盤・分析",
    "パフォーマンスチューニング": "パフォーマンスチューニング",
    "ヘルプデスク運用代行": "問い合わせ・ヘルプデスク運用",
    "問い合わせ対応システム導入": "問い合わせ・ヘルプデスク運用",
    "モバイルアプリ開発": "モバイルアプリ開発",
    "保守運用サポート": "サーバー・インフラ運用",
    "基幹システム導入": "基幹システム",
    "契約管理システム導入": "契約管理",
    "業務効率化コンサルティング": "業務効率化コンサル",
}


def cert_matches_topic(
    cert_name: str | None,
    topic: str,
    keyword_map: Mapping[str, Sequence[str]] = CERT_TOPIC_KEYWORDS,
) -> bool:
    """True when ``cert_name`` contains any keyword mapped to ``topic``."""

    if not cert_name:
        return False
    return any(keyword in cert_name for keyword in keyword_map.get(topic, ()))


def product_matches_topic(
    product: str | None,
    topic: str,
    product_map: Mapping[str, str] = PRODUCT_TOPIC_MAP,
) -> bool:
    """True when a project's ``product`` maps to ``topic``."""

    if not product:
        return False
    return product_map.get(product) == topic

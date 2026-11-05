"""Deterministic, LLM-free stand-ins for the C1/C2/C7 LLM nodes.

These satisfy the protocols in :mod:`tekijin.agent.protocols` with rule-based
logic — keyword tables and templates — so the whole graph runs (and is tested)
without a model or the network. They are the DEFAULT implementations; a real
vLLM-backed version is injected later without changing the graph. Every method
is a pure function of its inputs, so runs are reproducible.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tekijin.agent.protocols import (
    AnswerabilityResult,
    IntentResult,
    QuestionStructureResult,
    SelfAnswerResult,
    SufficiencyResult,
)

if TYPE_CHECKING:  # pragma: no cover - typing only (keeps retrieval out of the stub path)
    from tekijin.retrieval.fragments import CitedEvidence

# At most one clarifying round (model-definition C2: "逆質問はまとめて1回").
MAX_FOLLOWUPS = 1

# Below this C1 confidence — or with no topic extracted at all — the intent is
# too unclear to search on, so C2 asks the user to clarify (model-definition C1:
# "confidence<閾値 → C2 の逆質問へ").
INTENT_CONFIDENCE_THRESHOLD = 0.5

# Question keyword -> canonical topic (matched case-insensitively). Keys are the
# 22 canonical topics and MUST match ``answers.topic`` verbatim (the scorer joins on
# exact topic equality). Extend freely; this is data, not logic.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "CRM・営業支援": ("crm", "営業支援", "顧客管理", "sfa"),
    "ECサイト構築": ("ec", "ecサイト", "ネットショップ", "通販サイト", "カート"),
    "SNS運用": ("sns", "インスタ", "twitter", "x運用", "tiktok"),
    "Webマーケティング・広告": ("広告", "webマーケ", "seo", "リスティング", "集客"),
    "クラウド移行": ("クラウド", "aws", "azure", "gcp", "オンプレ", "移行"),
    "サーバー・インフラ運用": ("サーバー", "インフラ", "linux", "運用監視", "オンプレミス"),
    "システム開発・API": ("api", "システム開発", "連携", "スクラッチ開発"),
    "セキュリティ": ("セキュリティ", "utm", "ファイアウォール", "脆弱性", "不正アクセス"),
    "データ基盤・分析": ("データ基盤", "分析", "bi", "データ活用", "dwh", "etl"),
    "ネットワーク・VPN": ("vpn", "ネットワーク", "回線", "拠点間"),
    "パフォーマンスチューニング": (
        "パフォーマンス",
        "チューニング",
        "遅い",
        "高速化",
        "レスポンス改善",
    ),
    "モバイルアプリ開発": ("モバイルアプリ", "アプリ開発", "ios", "android", "スマホアプリ"),
    "人事・採用": ("採用", "人事", "労務", "求人", "面接"),
    "問い合わせ・ヘルプデスク運用": ("問い合わせ対応", "コールセンター", "faq", "チケット"),
    "基幹システム": ("基幹システム", "erp", "在庫管理", "受発注"),
    "契約管理": ("契約", "契約書", "電子契約"),
    "広報・PR": ("広報", "pr", "プレスリリース", "ブランディング"),
    "業務効率化コンサル": ("業務効率化", "業務改善", "rpa", "自動化", "コンサル"),
    "社内IT・ヘルプデスク": ("社内it", "社内ヘルプデスク", "pcトラブル", "情シス", "キッティング"),
    "経理・決算": ("経理", "決算", "簿記", "仕訳", "請求書"),
    "総務・法務": ("総務", "法務", "コンプライアンス", "規程", "契約審査"),
    "購買・仕入れ": ("購買", "仕入れ", "調達", "発注", "サプライヤー"),
}

# Product-like tokens to surface verbatim when present (case-insensitive match).
PRODUCT_KEYWORDS: tuple[str, ...] = ("UTM", "VPN", "CRM", "API", "BI", "EC", "SNS")

# Anything clearly not work-related -> out_of_scope.
OUT_OF_SCOPE_KEYWORDS: tuple[str, ...] = (
    "天気",
    "ランチ",
    "昼飯",
    "株価",
    "競馬",
    "恋愛",
    "芸能",
    "プライベート",
    "旅行",
)

# question_type keyword tables (checked in order after out_of_scope).
_QUOTE_KEYWORDS = ("見積", "価格", "費用", "料金")
_ADMIN_KEYWORDS = ("申請", "手続き", "経費精算", "精算")
_CHITCHAT_KEYWORDS = ("こんにちは", "雑談", "おはよう", "よろしく")
# #108: wording that asks to be put in touch with someone rather than only for an
# answer (「見てくれる人」「誰に聞けばいい」「相談したい」).
SEEKS_PERSON_KEYWORDS: tuple[str, ...] = (
    "くれる人",
    "くれそうな人",
    "できる人",
    "詳しい人",
    "くわしい人",
    "わかる人",
    "分かる人",
    "人を探",
    "誰に聞",
    "誰に相談",
    "相談したい",
)

# Required information slots per question type, and how to detect each in text.
_REQUIRED_SLOTS: dict[str, tuple[str, ...]] = {
    "見積": ("現行製品", "対象拠点数"),
    "技術相談": ("現行製品", "対象拠点数"),
}


# Short pure-ASCII abbreviations (≤ this many chars) must match on a word
# boundary, so "ec"/"pr"/"bi" do not fire inside "security"/"project"/"ability".
# Longer keys and Japanese keys keep plain substring matching.
_SHORT_ABBR_MAX = 3


def _keyword_matches(keyword: str, text: str) -> bool:
    """True if ``keyword`` occurs in ``text`` (case-insensitive).

    A short pure-ASCII alphanumeric abbreviation matches only at ASCII word
    boundaries (``re.ASCII`` so Japanese chars count as boundaries), so it never
    fires inside a longer English word. Everything else is substring matching.
    """

    k = keyword.lower()
    lowered = text.lower()
    if len(k) <= _SHORT_ABBR_MAX and k.isascii() and k.isalnum():
        return re.search(rf"\b{re.escape(k)}\b", lowered, re.ASCII) is not None
    return k in lowered


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(_keyword_matches(k, text) for k in keywords)


# Site-count slot values look like "5拠点" / "3店舗": a number glued to a unit.
_SITE_COUNT_RE = re.compile(r"\d+\s*(?:拠点|箇所|店舗|事業所)")


def missing_required_slots(question: str, intent: IntentResult) -> list[str]:
    """Deterministically list the required slots not yet filled for this intent.

    The same slot logic :meth:`RuleSufficiencyModel.check` uses (``_REQUIRED_SLOTS``
    minus ``_slot_present``), factored out so the C2 fast path (#376) can surface
    ``missing`` for the hand-off draft WITHOUT the (LLM) sufficiency call. LLM-free
    and prompt-independent, so a confident, on-topic query still tells the responder
    which estimate slots (現行製品 / 対象拠点数) remain open.
    """

    required = _REQUIRED_SLOTS.get(intent.question_type, ())
    return [
        slot for slot in required if not RuleSufficiencyModel._slot_present(slot, question, intent)
    ]


def collect_known_values(question: str, question_type: str, products: list[str]) -> dict[str, str]:
    """Concrete values for the required slots that are actually filled (C7 input).

    Mirrors :meth:`RuleSufficiencyModel._slot_present` so a slot appears here iff
    it would count as present there. C7 feeds these to the draft so the hand-off
    shows the *filled* value (model-definition C7: ``known_values`` input) rather
    than re-asking it. The ``現行製品`` branch also re-scans ``question`` when
    ``products`` is empty — a defensive fallback for a caller/model that leaves
    ``products`` unset while the name is in the text (the stub's C1 already
    populates it, so this is belt-and-suspenders, not the primary path).
    """

    values: dict[str, str] = {}
    for slot in _REQUIRED_SLOTS.get(question_type, ()):
        if slot == "現行製品":
            matched = products or [p for p in PRODUCT_KEYWORDS if _keyword_matches(p, question)]
            if matched:
                # Prefer the product mentioned earliest in the text (not the fixed
                # keyword-table order): this value is shown to the responder as a
                # confirmed premise, so "CRMとVPN" should surface CRM, not VPN.
                lowered = question.lower()
                values[slot] = min(matched, key=lambda p: _mention_index(lowered, p))
        elif slot == "対象拠点数":  # pragma: no branch - only two slots defined
            match = _SITE_COUNT_RE.search(question)
            if match:
                values[slot] = match.group(0)
    return values


def _mention_index(lowered_question: str, product: str) -> int:
    """Position of ``product`` in the (already lower-cased) question, or +inf-ish.

    A product not literally found (only possible for an externally supplied
    ``products`` value that is not a substring) sorts last rather than winning at
    index -1.
    """

    index = lowered_question.find(product.lower())
    return index if index >= 0 else len(lowered_question)


class KeywordIntentModel:
    """C1 stub: extract topics/products and classify by keyword tables."""

    def analyze(
        self,
        question: str,
        asker: dict[str, Any] | None,
        *,
        context: Sequence[str] | None = None,
    ) -> IntentResult:
        out_of_scope = _contains_any(question, OUT_OF_SCOPE_KEYWORDS)
        question_topics = [t for t, kws in TOPIC_KEYWORDS.items() if _contains_any(question, kws)]
        # #69 topic mediation: a retrieved fragment that names a canonical topic
        # keyword surfaces that topic even when the QUESTION worded it differently
        # (the #116 vocabulary-mismatch bridge). Context ONLY adds to the emitted
        # `topics` (what feeds C6 scoring); it never pulls an off-topic question
        # back in-scope, and — deliberately — does NOT move question_type or
        # confidence, which stay driven by the user's actual question. Otherwise a
        # merely topic-adjacent retrieval hit could force a needless C2 follow-up
        # (question_type→技術相談 flips required slots) or lift confidence past the
        # clarify threshold for the wrong reason (code-review #275).
        context_topics: list[str] = []
        if context and not out_of_scope:
            context_text = " ".join(context)
            context_topics = [
                t for t, kws in TOPIC_KEYWORDS.items() if _contains_any(context_text, kws)
            ]
        products = [p for p in PRODUCT_KEYWORDS if _keyword_matches(p, question)]
        # question_type / confidence: question-derived topics only (see above).
        question_type = self._classify(question, question_topics, products, out_of_scope)

        if out_of_scope:
            confidence = 0.9  # confident it is off-topic
        else:
            confidence = 0.4
            if question_topics:
                confidence += 0.25
            if products:
                confidence += 0.15
            if question_type in ("見積", "技術相談", "事務手続き"):
                confidence += 0.1
        confidence = round(min(confidence, 1.0), 2)

        # Emit the mediated union in canonical vocabulary order (source-agnostic).
        merged = set(question_topics) | set(context_topics)
        topics = [t for t in TOPIC_KEYWORDS if t in merged]

        return IntentResult(
            topics=topics,
            products=products,
            situation=None,
            question_type=question_type,
            out_of_scope=out_of_scope,
            confidence=confidence,
            seeks_person=not out_of_scope and _contains_any(question, SEEKS_PERSON_KEYWORDS),
        )

    @staticmethod
    def _classify(question: str, topics: list[str], products: list[str], out_of_scope: bool) -> str:
        if out_of_scope:
            return "業務外"
        if _contains_any(question, _QUOTE_KEYWORDS):
            return "見積"
        if _contains_any(question, _ADMIN_KEYWORDS):
            return "事務手続き"
        # Substantive intent (a topic or product) wins over a greeting: a message
        # like "こんにちは、ネットワークの技術相談です" is a technical consult, not
        # chitchat. Only classify 雑談 when nothing actionable was extracted.
        if topics or products:
            return "技術相談"
        if _contains_any(question, _CHITCHAT_KEYWORDS):
            return "雑談"
        return "製品QA"


class RuleSufficiencyModel:
    """C2 stub: flag missing required slots; ask back at most once."""

    def check(self, question: str, intent: IntentResult, followup_count: int) -> SufficiencyResult:
        missing = missing_required_slots(question, intent)

        # Already asked our one clarification -> proceed regardless, but KEEP any
        # still-unresolved slots so C7 can flag them (do not silently clear them).
        if followup_count >= MAX_FOLLOWUPS:
            return SufficiencyResult(sufficient=True, missing=missing, followup_question=None)

        # Intent itself too weak to search on -> ask to clarify the intent. This
        # deliberately routes low-signal chitchat ("こんにちは") through a friendly
        # clarification rather than off_topic: a greeting is in-scope for a work
        # helpdesk, so we ask what they need instead of deflecting. Genuinely
        # off-topic input (天気/ランチ …) is caught earlier by C1's out_of_scope.
        if intent.confidence < INTENT_CONFIDENCE_THRESHOLD or not intent.topics:
            return SufficiencyResult(
                sufficient=False,
                missing=["相談内容"],
                followup_question="ご相談の内容やトピックを、もう少し具体的に教えてください。",
            )
        if not missing:
            return SufficiencyResult(sufficient=True, missing=[], followup_question=None)
        followup = "次の点を教えてください: " + "、".join(missing)
        return SufficiencyResult(sufficient=False, missing=missing, followup_question=followup)

    @staticmethod
    def _slot_present(slot: str, question: str, intent: IntentResult) -> bool:
        # A slot is satisfied only by an actual VALUE, not a bare label: "現行環境"
        # or "拠点間" mention the topic without answering "which product?" / "how
        # many sites?", so they must still prompt a clarification.
        if slot == "現行製品":
            return bool(intent.products)  # a concrete, known product name
        if slot == "対象拠点数":  # pragma: no branch - only two slots defined
            return _SITE_COUNT_RE.search(question) is not None
        return True  # pragma: no cover - defensive; unknown slots count as present


class TemplateDraftModel:
    """C7 stub: fill a polite Japanese hand-off template (敬体, 簡潔)."""

    def draft(
        self,
        question: str,
        responder: dict[str, Any],
        asker: dict[str, Any] | None,
        missing: list[str],
        *,
        situation: str | None = None,
        topics: list[str] | None = None,
        known_values: dict[str, str] | None = None,
    ) -> str:
        name = responder.get("name") or "ご担当者"
        dept = responder.get("dept") or responder.get("department") or ""
        header = f"{name}さん（{dept}）" if dept else f"{name}さん"
        lines = [
            header,
            "お世話になっております。下記の件でご相談させてください。",
        ]
        # C1 の状況理解を背景として先頭に添える（依頼文が質問の意味とずれない）。
        if situation:
            lines.append(f"【背景】\n{situation}")
        lines.append(f"【相談内容】\n{question}")
        # 確定済みスロット値は「埋まった前提」として明示し、回答者が確認しやすくする。
        if known_values:
            filled = "、".join(f"{slot}：{value}" for slot, value in known_values.items())
            lines.append(f"【確認済みの前提】\n{filled}")
        if topics:
            lines.append("【関連トピック】\n" + "、".join(topics))
        if missing:
            lines.append("【補足いただきたい点】\n" + "、".join(missing))
        lines.append("お手数ですが、ご確認いただけますと幸いです。")
        return "\n".join(lines)


class RuleAnswerabilityModel:
    """Evidence-sufficiency critic stub (#70): confidence from candidate evidence.

    Deterministic scaffolding — the real judgement is the vLLM critic. With no
    candidate (nobody to answer) the confidence is 0 (reject); otherwise it rises
    with how many candidates carry a track-record line, capped at 100. The graph
    compares this to an externalised threshold, so the stub never hard-codes the
    accept/reject decision here.
    """

    def assess(self, question: str, candidate_evidence: Sequence[str]) -> AnswerabilityResult:
        evidence = [line for line in candidate_evidence if line and line.strip()]
        if not evidence:
            return AnswerabilityResult(
                confidence=0, reason="回答できる実績が社内に見つかりません。"
            )
        confidence = min(100, 40 + 20 * len(evidence))
        return AnswerabilityResult(confidence=confidence, reason="候補者に関連実績があります。")


class TemplateSelfAnswerModel:
    """Self-answer composer stub (#291): deterministic, network-free scaffolding.

    The real composition is the vLLM model's job; this stub just proves the
    contract — no evidence means ``grounded=False`` (fall back to a person), and
    with evidence it echoes the top source and cites every supplied source id so
    the chat can link back. It never fabricates content beyond the evidence text.
    """

    def compose(self, question: str, evidence: Sequence[CitedEvidence]) -> SelfAnswerResult:
        items = [e for e in evidence if e.text and e.text.strip()]
        if not items:
            return SelfAnswerResult(answer="", cited_source_ids=[], grounded=False)
        answer = f"社内の記録によると、{items[0].text}"
        return SelfAnswerResult(
            answer=answer,
            cited_source_ids=[e.source_id for e in items],
            grounded=True,
        )


class TemplateQuestionStructurer:
    """Question re-drafter stub (#475): deterministic, network-free scaffolding.

    The real reshaping is the vLLM model's job; this stub proves the contract so the
    on-demand endpoint runs LLM-free in tests. It seeds ``summary`` from C1's
    ``situation`` (falling back to the raw question) and ``blocker`` from the question
    itself, and leaves ``environment``/``tried`` EMPTY — exactly the "don't invent a
    field the asker never gave" rule the real model must follow. It never fabricates
    an environment or a tried-step it has no evidence for.
    """

    def structure(
        self,
        question: str,
        *,
        situation: str | None = None,
        topics: list[str] | None = None,
    ) -> QuestionStructureResult:
        text = (question or "").strip()
        summary = (situation or "").strip() or text
        return QuestionStructureResult(
            summary=summary,
            environment="",
            tried="",
            blocker=text,
        )

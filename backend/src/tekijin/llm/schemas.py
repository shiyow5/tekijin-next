"""Pydantic schemas the LLM (C1/C2) is forced to emit via structured output.

These mirror the dataclasses in :mod:`tekijin.agent.protocols` but carry field
descriptions to steer the model, and are what ``with_structured_output`` binds.
The vLLM adapters convert an instance of these back into the protocol dataclasses.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class IntentSchema(BaseModel):
    """C1 structured output (model-definition §2 C1)."""

    products: list[str] = Field(default_factory=list, description="言及された製品名")
    situation: str | None = Field(default=None, description="状況の一言要約")
    question_type: str = Field(
        default="製品QA", description="製品QA/見積/技術相談/事務手続き/雑談/業務外 のいずれか"
    )
    out_of_scope: bool = Field(
        default=False, description="組織と結びつかない雑談・悪意ある入力なら true"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="意図理解の確信度")
    seeks_person: bool = Field(
        default=False,
        description=(
            "答えそのものより、頼める人・詳しい人を探している、または繋いでほしいなら true"
            "（「〜できる人」「誰に聞けば」「相談したい」「見てくれる人」など）"
        ),
    )


class SufficiencySchema(BaseModel):
    """C2 structured output (model-definition §2 C2)."""

    sufficient: bool = Field(default=True, description="取り次ぎ・検索に十分な情報があるか")
    missing: list[str] = Field(default_factory=list, description="不足している必須項目")
    followup_question: str | None = Field(
        default=None, description="不足時に返すまとめて1つの逆質問"
    )

    @model_validator(mode="after")
    def _followup_required_when_insufficient(self) -> SufficiencySchema:
        # An insufficient result MUST carry a non-empty follow-up, or the graph
        # would pause on an empty clarification (the LLM must not omit it).
        if not self.sufficient and not (self.followup_question or "").strip():
            raise ValueError("followup_question is required when sufficient is false")
        return self


class AnswerabilitySchema(BaseModel):
    """Evidence-sufficiency structured output (#70 critic).

    ``confidence`` is deliberately an INTEGER 0–100, not a boolean: asked as a
    yes/no the model over-rejects (18/45 vs 3/45 misreject as a number — #65/#67
    §6). The graph compares it to an externalised threshold.
    """

    confidence: int = Field(
        default=0, ge=0, le=100, description="社内の実績でこの相談に答えられる確度(0-100)"
    )
    reason: str | None = Field(default=None, description="判断の根拠を一言で（任意）")


class CaseExtractionSchema(BaseModel):
    """Knowledge-unit extraction structured output (#357 slice 2, case type).

    The model reads ONE raw record (a chat conversation or a resolved thread) and, if it
    holds a reusable *case*, distils it into ``問題(状況) → 打ち手 → 結果``. Not every
    record is a case — a status note with no problem/action is not — so
    ``extractable`` lets the model pass (the caller then stores nothing, keeping a
    non-case out of the knowledge base). For a source that CARRIES precomputed tags
    (a thread's question) ``topics`` come from those tags, never the model, so the
    knowledge vocabulary can never drift. For a source with NO
    tags (chat, #448) the model instead proposes ``topic_hints`` and the caller snaps
    them onto the canonical vocabulary (``normalize_topics``) — same drift exposure
    as C1 routing, and anything off-vocabulary is dropped. The model must ground
    every field in the supplied text and never invent a ``result`` (leave it null).
    """

    extractable: bool = Field(
        default=False, description="この記録が再利用可能なケース(課題→打ち手→結果)を含むなら true"
    )
    problem: str = Field(default="", description="顧客の状況・課題（記録に書かれた範囲で）")
    action: str = Field(default="", description="打ち手・提案した商材やソリューション")
    result: str | None = Field(
        default=None, description="結果（受注・継続商談など）。記録に無ければ null"
    )
    industry: str | None = Field(
        default=None, description="顧客の業種（記録に明示があれば）。無ければ null"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="抽出の確信度")
    topic_hints: list[str] = Field(
        default_factory=list,
        description=(
            "この知識が属するトピック候補（タグの無いチャット等の抽出でのみ使用）。"
            "決められた語彙から選ぶ。タグを持つ記録では空のままでよい。"
        ),
    )

    @model_validator(mode="after")
    def _extractable_requires_problem_and_action(self) -> CaseExtractionSchema:
        # A case is meaningless without BOTH a problem and an action — if the model
        # claims extractable it must supply both, else the caller would store an
        # empty unit. (extractable=false with stray text is harmless: skipped.)
        if self.extractable:
            if not self.problem.strip():
                raise ValueError("problem is required when extractable is true")
            if not self.action.strip():
                raise ValueError("action is required when extractable is true")
        return self


class QuestionStructureSchema(BaseModel):
    """On-demand question re-draft structured output (#475 Screen 01).

    The model reshapes a raw, anxious question into the four fields a responder needs.
    Every field defaults to ``""`` and the prompt tells the model to LEAVE A FIELD
    EMPTY when the question does not state it — inventing an ``environment`` the asker
    never gave would mislead the responder and defeat the "answerer can trust this"
    goal. The asker edits these before sending, so an empty field is a prompt to fill
    in, not a failure. No field is a citation or an id, so no allow-list applies here;
    the text is purely presentational (never persisted, never fed to routing).
    """

    summary: str = Field(
        default="", description="起きていること（何が起きているかの一言要約／不明なら空）"
    )
    environment: str = Field(
        default="",
        description="環境（OS・バージョン・利用サービスなど、質問に書かれた範囲で／無ければ空）",
    )
    tried: str = Field(
        default="", description="試したこと（質問に書かれた範囲で／書かれていなければ空）"
    )
    blocker: str = Field(
        default="", description="詰まっている点（何が分からず止まっているか／不明なら空）"
    )


class SelfAnswerSchema(BaseModel):
    """Self-answer structured output (#291): a grounded, cited answer or a pass.

    The model must answer ONLY from the supplied evidence and list the source ids it
    used. When the evidence does not answer the question it sets ``grounded=false``
    (and leaves ``answer`` empty) so the graph falls back to a human hand-off rather
    than surfacing an ungrounded answer.
    """

    grounded: bool = Field(
        default=False, description="提供された根拠だけで質問に回答できるなら true"
    )
    answer: str = Field(default="", description="根拠のみに基づく回答本文（grounded=false なら空）")
    cited_source_ids: list[str] = Field(
        default_factory=list, description="回答に実際に用いた根拠の source_id（提供分の部分集合）"
    )

    @model_validator(mode="after")
    def _grounded_requires_answer_and_citation(self) -> SelfAnswerSchema:
        # A grounded answer must carry BOTH text and at least one citation — else the
        # graph would emit an empty/uncited "answer" instead of falling back to
        # routing. A grounded answer with zero citations is the strongest signal the
        # model fabricated it (the composer verifies the ids are real; here we at
        # least require it to claim one). (The reverse — grounded=false with stray
        # text — is harmless: the graph ignores it and routes.)
        if self.grounded:
            if not self.answer.strip():
                raise ValueError("answer is required when grounded is true")
            if not self.cited_source_ids:
                raise ValueError("cited_source_ids is required when grounded is true")
        return self

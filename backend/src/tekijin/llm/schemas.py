"""Pydantic schemas the LLM (C1/C2) is forced to emit via structured output.

These mirror the dataclasses in :mod:`tekijin.agent.protocols` but carry field
descriptions to steer the model, and are what ``with_structured_output`` binds.
The vLLM adapters convert an instance of these back into the protocol dataclasses.
"""

from __future__ import annotations

from typing import Literal

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


class KnowledgeDraftSchema(BaseModel):
    """One knowledge draft in the KCS shape, read from one record (#112, R6).

    The model reads one record (a resolved conversation, a part of a document) and,
    when it holds something worth keeping, writes the ``issue`` in the words a
    member would ask it, the ``resolution``, and optionally where it holds
    (``environment``) and why (``cause``). ``extractable`` false means there is
    nothing to keep, and the caller stores nothing. A missing ``resolution`` is
    allowed: the draft is stored as ``wip`` and never answers a question. Every
    field must come from the record.
    """

    extractable: bool = Field(
        default=False, description="記録に、残して再利用できる知識が含まれるなら true"
    )
    kind: Literal["case", "procedure", "fact", "decision"] = Field(
        default="case",
        description="case=困りごとと解決、procedure=手順、fact=事実・決まりごと、decision=決めたことと理由",
    )
    issue: str = Field(default="", description="困りごと、または問い。質問する人の言葉に近い形で")
    environment: str | None = Field(
        default=None, description="どの場面で成り立つか(活動、道具、時期など)。記録に無ければ null"
    )
    resolution: str | None = Field(
        default=None, description="答え・解決のしかた。記録に書かれていなければ null"
    )
    cause: str | None = Field(default=None, description="原因。記録に無ければ null")

    @model_validator(mode="after")
    def _extractable_requires_an_issue(self) -> KnowledgeDraftSchema:
        if self.extractable and not self.issue.strip():
            raise ValueError("issue is required when extractable is true")
        return self


#: How many drafts one passage of a document may give (#113). A manual's passage
#: holds a few procedures or facts; a long list means the model is splitting hairs.
MAX_DRAFTS_PER_PASSAGE = 6


class DocumentKnowledgeItem(BaseModel):
    """One procedure or fact read from a passage of a document (#113)."""

    kind: Literal["procedure", "fact"] = Field(
        default="fact", description="procedure=手順、fact=事実・決まりごと"
    )
    issue: str = Field(description="この知識が答える問い。質問する人の言葉に近い形で")
    environment: str | None = Field(
        default=None, description="どの場面で成り立つか。本文に無ければ null"
    )
    resolution: str = Field(description="問いへの答え。本文に書かれていることだけ")
    cause: str | None = Field(default=None, description="理由。本文に無ければ null")


class DocumentKnowledgeSchema(BaseModel):
    """The procedures and facts in one passage of a document (#113, R6).

    A manual or a hand-over note holds several pieces of knowledge, so the model
    returns a list, possibly empty. Every item must answer its question from the
    passage; items without an issue or an answer are dropped by the caller, and
    only the first :data:`MAX_DRAFTS_PER_PASSAGE` are kept.
    """

    items: list[DocumentKnowledgeItem] = Field(
        default_factory=list,
        description=f"本文から読み取れる手順と事実(多くて{MAX_DRAFTS_PER_PASSAGE}件)。無ければ空",
    )


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

import type { KnowledgeArticle } from "@/lib/api-types";

/** Japanese labels for a knowledge article's kind and state (#112). */
export const KIND_LABEL: Record<KnowledgeArticle["kind"], string> = {
  case: "事例",
  procedure: "手順",
  fact: "事実",
  decision: "決定",
};

export const STATE_LABEL: Record<KnowledgeArticle["state"], string> = {
  wip: "作業中",
  draft: "確認待ち",
  verified: "確認済み",
  retired: "引退",
};

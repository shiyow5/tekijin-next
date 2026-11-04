import { SourceCitations } from "@/components/SourceCitations";
import type { SourceCitation } from "@/lib/api-types";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("next/link", () => ({
  default: ({ href, children }: { href: string; children: React.ReactNode }) => (
    <a href={href}>{children}</a>
  ),
}));

function renderCitations(citations: SourceCitation[]) {
  return render(<SourceCitations citations={citations} sessionId="s1" />);
}

describe("SourceCitations", () => {
  it("links a document citation to the internal viewer", () => {
    renderCitations([{ source_id: "doc_1", kind: "document" }]);
    expect(screen.getByRole("link", { name: /doc_1/ })).toHaveAttribute(
      "href",
      "/documents/doc_1?from=s1",
    );
  });

  it("shows an unknown kind as a label, not a link (#91)", () => {
    // A "daily" citation stored before #91 is no longer in the type, but old
    // history still carries it. It must not link to a page that does not exist.
    const stale = { source_id: "daily_42", kind: "daily" } as unknown as SourceCitation;
    renderCitations([stale]);
    expect(screen.getByText("出典 daily_42")).toBeInTheDocument();
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.queryByText(/過去の回答/)).toBeNull();
  });

  it("labels a knowledge unit as knowledge, not as a past answer (#366)", () => {
    // #357 slice 4c emits `{source_id: "ku_…", kind: "knowledge"}` when a grounded
    // knowledge answer fires. The renderer's final branch is a catch-all that says
    // 「過去の回答」, so before this the knowledge layer would have mislabelled every
    // one of its own citations the moment `knowledge_retrieval_enabled` went true.
    // That is why this is the enablement gate for #357.
    renderCitations([{ source_id: "ku_12", kind: "knowledge" }]);
    expect(screen.queryByText(/過去の回答/)).toBeNull();
    expect(screen.getByText(/ナレッジ/)).toBeInTheDocument();
  });

  it("still labels a qa citation as a past answer", () => {
    renderCitations([{ source_id: "ans_9", kind: "qa" }]);
    expect(screen.getByText(/過去の回答/)).toBeInTheDocument();
  });

  it("renders nothing without citations", () => {
    const { container } = render(<SourceCitations citations={[]} />);
    expect(container.firstChild).toBeNull();
  });
});

import { ApiError, getArticle, getArticles, patchArticle, reviewArticle } from "@/lib/api-client";
import { describe, expect, it, vi } from "vitest";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function call(fetchImpl: ReturnType<typeof vi.fn>, index = 0) {
  const [url, init] = fetchImpl.mock.calls[index] as [string, RequestInit];
  return { url, init };
}

describe("api-client knowledge articles (#112)", () => {
  it("lists with the view, search and paging in the query", async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(jsonResponse({ items: [], total: 0 })));
    await getArticles({
      fetchImpl,
      baseUrl: "http://api",
      view: "review",
      q: "鍵",
      offset: 50,
      limit: 50,
    });
    expect(call(fetchImpl).url).toBe(
      `http://api/articles?view=review&q=${encodeURIComponent("鍵")}&offset=50&limit=50`,
    );
    await getArticles({ fetchImpl, baseUrl: "http://api" });
    expect(call(fetchImpl, 1).url).toBe("http://api/articles");
  });

  it("reads one article", async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(jsonResponse({ id: 3 })));
    await getArticle(3, { fetchImpl, baseUrl: "http://api" });
    expect(call(fetchImpl).url).toBe("http://api/articles/3");
  });

  it("edits with PATCH", async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(jsonResponse({ id: 3 })));
    await patchArticle(3, { resolution: "箱" }, { fetchImpl, baseUrl: "http://api" });
    const { url, init } = call(fetchImpl);
    expect(url).toBe("http://api/articles/3");
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(String(init.body))).toEqual({ resolution: "箱" });
  });

  it("posts the review actions", async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(jsonResponse({ id: 3 })));
    await reviewArticle(3, "verify", { fetchImpl, baseUrl: "http://api" });
    await reviewArticle(3, "retire", { fetchImpl, baseUrl: "http://api", supersededBy: 9 });
    expect(call(fetchImpl).url).toBe("http://api/articles/3/verify");
    expect(JSON.parse(String(call(fetchImpl).init.body))).toEqual({});
    expect(call(fetchImpl, 1).url).toBe("http://api/articles/3/retire");
    expect(JSON.parse(String(call(fetchImpl, 1).init.body))).toEqual({ superseded_by: 9 });
  });

  it("raises the server's message", async () => {
    const fetchImpl = vi.fn(() => Promise.resolve(jsonResponse({ detail: "だめ" }, 403)));
    await expect(
      reviewArticle(3, "send-back", { fetchImpl, baseUrl: "http://api" }),
    ).rejects.toEqual(new ApiError(403, "だめ"));
  });
});

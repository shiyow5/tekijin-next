import { expect, test } from "@playwright/test";
import {
  API_BASE,
  COMPLETED_PERSON_ROUTE_FRAMES,
  RECENT_QUESTIONS,
  fulfillJson,
  fulfillSse,
  mockAuth,
  mockEmployees,
  mockRecentQuestions,
  sseBody,
} from "./support/mocks";

/**
 * The /history screen (#397): every card with a session_id is a single click
 * target to `/session/{id}/result` (no separate "結果を見る" link); delete and
 * self-resolve moved from two always-visible buttons into one "…" options
 * menu; the list paginates 5 cards per page.
 */
test.describe("history", () => {
  test("a card with a session navigates on click anywhere in it", async ({ page }) => {
    await mockEmployees(page);
    await mockAuth(page);
    await mockRecentQuestions(page, RECENT_QUESTIONS);
    // The destination is /session/{id}/result; only its URL matters here, so a
    // bare (never-resolving) SSE connection is enough to avoid a network error.
    await page.route(`${API_BASE}/events/**`, (route) => fulfillSse(route, sseBody([])));

    await page.goto("/history");
    await expect(page.getByRole("heading", { name: "質問履歴" })).toBeVisible();

    // No small "結果を見る" text link exists anymore — the whole card is the target.
    await expect(page.getByRole("link", { name: "結果を見る" })).toHaveCount(0);
    await page.getByText("UTMの移行時の注意点").click();
    await page.waitForURL(/\/session\/sess-rq1\/result\?from=history$/);

    // Reached from a history card: the back link returns to /history, not home.
    const back = page.getByRole("link", { name: "履歴へ戻る" });
    await expect(back).toBeVisible();
    await expect(page.getByRole("link", { name: "ホームへ戻る" })).toHaveCount(0);
    await back.click();
    await page.waitForURL(/\/history$/);
  });

  test("a completed hand-off still shows the candidates and why they were picked", async ({
    page,
  }) => {
    // #520 (with #512): the outcome alone does not say WHY this person. Reaching a
    // finished session from the history list must still answer that — this is the
    // journey that used to dead-end on 「依頼は送信済みです」.
    await mockEmployees(page);
    await mockAuth(page);
    await mockRecentQuestions(page, RECENT_QUESTIONS);
    await page.route(`${API_BASE}/events/**`, (route) =>
      fulfillSse(route, sseBody(COMPLETED_PERSON_ROUTE_FRAMES)),
    );

    await page.goto("/history");
    await page.getByText("UTMの移行時の注意点").click();
    await page.waitForURL(/\/session\/sess-rq1\/result\?from=history$/);

    await expect(page.getByRole("heading", { name: "依頼は送信済みです" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "候補と根拠" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "高梨 健太" })).toBeVisible();
    await expect(page.getByText("類似の質問に過去3件回答")).toBeVisible();
    // Read-only: a hand-off that already happened cannot be re-targeted.
    await expect(page.getByRole("button", { name: "選択する" })).toHaveCount(0);
  });

  test("a history-only card (no session) stays non-interactive", async ({ page }) => {
    await mockEmployees(page);
    await mockAuth(page);
    await mockRecentQuestions(page, RECENT_QUESTIONS);

    await page.goto("/history");
    const card = page.locator("li").filter({ hasText: "社内PCのセットアップ手順" });
    await expect(card.getByRole("link")).toHaveCount(0);
  });

  test("the options menu deletes a card after confirmation", async ({ page }) => {
    await mockEmployees(page);
    await mockAuth(page);
    await mockRecentQuestions(page, RECENT_QUESTIONS);
    await page.route(`${API_BASE}/questions/api_rq2`, (route) =>
      fulfillJson(route, { question_id: "api_rq2", deleted: true }),
    );

    await page.goto("/history");
    await expect(page.getByText("社内Wi-Fiの申請方法")).toBeVisible();

    await page.getByRole("button", { name: "「社内Wi-Fiの申請方法」の操作" }).click();
    await page.getByRole("menuitem", { name: "削除" }).click();
    const dialog = page.getByRole("dialog", { name: "削除しますか？" });
    await expect(dialog).toBeVisible();
    await dialog.getByRole("button", { name: "削除" }).click();

    await expect(page.getByText("社内Wi-Fiの申請方法")).toHaveCount(0);
  });

  test("the options menu marks a pending card self-resolved after confirmation", async ({
    page,
  }) => {
    await mockEmployees(page);
    await mockAuth(page);
    await mockRecentQuestions(page, RECENT_QUESTIONS);
    await page.route(`${API_BASE}/questions/api_rq2/resolve`, (route) =>
      fulfillJson(route, { question_id: "api_rq2", resolved: true }),
    );

    await page.goto("/history");
    await page.getByRole("button", { name: "「社内Wi-Fiの申請方法」の操作" }).click();
    await page.getByRole("menuitem", { name: "自分で解決した" }).click();
    const dialog = page.getByRole("dialog", { name: "自分で解決しましたか？" });
    await expect(dialog).toBeVisible();
    await dialog.getByRole("button", { name: "解決済みにする" }).click();

    await expect(dialog).toHaveCount(0);
    // A card already resolved offers no 自分で解決した option.
    await page.getByRole("button", { name: "「社内Wi-Fiの申請方法」の操作" }).click();
    await expect(page.getByRole("menuitem", { name: "自分で解決した" })).toHaveCount(0);
  });

  test("an open dialog blocks the options menu of a card further down the list (#397 follow-up)", async ({
    page,
  }) => {
    await mockEmployees(page);
    await mockAuth(page);
    // A full page of 5 spreads the cards well past the centered dialog panel's
    // own bounds, unlike the 3-item RECENT_QUESTIONS fixture — with only 3
    // short cards the panel can happen to visually cover all of them anyway,
    // which would pass even a buggy (non-portalled, stacking-context-trapped)
    // dialog for the wrong reason.
    const items = Array.from({ length: 5 }, (_, i) => ({
      question_id: `q${i + 1}`,
      title: `質問${i + 1}`,
      resolved: false,
      resolution: "pending",
      responder_name: null,
      session_id: null,
      created_at: "2026-08-19T09:30:00",
    }));
    await mockRecentQuestions(page, items);

    await page.goto("/history");
    await page.getByRole("button", { name: "「質問1」の操作" }).click();
    await page.getByRole("menuitem", { name: "削除" }).click();
    const dialog = page.getByRole("dialog", { name: "削除しますか？" });
    await expect(dialog).toBeVisible();

    // Sanity-check the setup itself: the target card's trigger must sit
    // genuinely outside the dialog panel's own bounds, or a click landing on
    // the panel proves nothing about the backdrop stacking bug this guards.
    const dialogBox = await dialog.boundingBox();
    const lastTrigger = page.getByRole("button", { name: "「質問5」の操作" });
    const triggerBox = await lastTrigger.boundingBox();
    expect(dialogBox).not.toBeNull();
    expect(triggerBox).not.toBeNull();
    expect(triggerBox?.y ?? 0).toBeGreaterThan((dialogBox?.y ?? 0) + (dialogBox?.height ?? 0));

    // A click where that lower, unrelated card's own "…" trigger sits must
    // hit the modal's backdrop (dismissing it), not the trigger underneath —
    // real native hit-testing decides the target, not Playwright's own
    // actionability check, hence `force` here.
    await lastTrigger.click({ force: true });
    await expect(dialog).toHaveCount(0);
    await expect(page.getByRole("menu")).toHaveCount(0);
  });

  test("the corner close button dismisses the dialog without acting (#397 follow-up)", async ({
    page,
  }) => {
    await mockEmployees(page);
    await mockAuth(page);
    await mockRecentQuestions(page, RECENT_QUESTIONS);

    await page.goto("/history");
    await page.getByRole("button", { name: "「社内Wi-Fiの申請方法」の操作" }).click();
    await page.getByRole("menuitem", { name: "削除" }).click();
    const dialog = page.getByRole("dialog", { name: "削除しますか？" });
    await expect(dialog).toBeVisible();

    await dialog.getByRole("button", { name: "ダイアログを閉じる" }).click();
    await expect(dialog).toHaveCount(0);
    await expect(page.getByText("社内Wi-Fiの申請方法")).toBeVisible();
  });

  test("the list paginates 5 cards per page", async ({ page }) => {
    await mockEmployees(page);
    await mockAuth(page);
    const items = Array.from({ length: 6 }, (_, i) => ({
      question_id: `q${i + 1}`,
      title: `質問${i + 1}`,
      resolved: false,
      resolution: "pending",
      responder_name: null,
      session_id: null,
      created_at: "2026-08-19T09:30:00",
    }));
    await mockRecentQuestions(page, items);

    await page.goto("/history");
    await expect(page.getByText("質問1")).toBeVisible();
    await expect(page.getByText("質問6")).toHaveCount(0);
    await expect(page.getByText("1 / 2")).toBeVisible();
    await expect(page.getByRole("button", { name: "前へ" })).toBeDisabled();

    await page.getByRole("button", { name: "次へ" }).click();
    await expect(page.getByText("質問6")).toBeVisible();
    await expect(page.getByText("質問1")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "次へ" })).toBeDisabled();
  });
});

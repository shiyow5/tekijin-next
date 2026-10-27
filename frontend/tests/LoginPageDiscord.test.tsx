import LoginPage from "@/../app/login/page";
import type { AuthContextValue } from "@/components/AuthProvider";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const useAuthMock = vi.fn<() => AuthContextValue>();
vi.mock("@/components/AuthProvider", () => ({
  useAuth: () => useAuthMock(),
}));

const replaceMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: replaceMock, push: vi.fn() }),
}));

const getSlackLoginUrlMock = vi.fn();
const redeemMock = vi.fn();
vi.mock("@/lib/api-client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api-client")>()),
  getSlackLoginUrl: () => getSlackLoginUrlMock(),
  redeemDiscordLogin: (token: string) => redeemMock(token),
}));

function auth(over: Partial<AuthContextValue> = {}): AuthContextValue {
  return {
    principal: null,
    loading: false,
    login: vi.fn(),
    logout: vi.fn(),
    adoptToken: vi.fn().mockResolvedValue(undefined),
    ...over,
  };
}

function setUrl(search: string, hash: string) {
  window.history.replaceState({}, "", `/login${search}${hash}`);
}

beforeEach(() => {
  replaceMock.mockReset();
  getSlackLoginUrlMock.mockReset();
  getSlackLoginUrlMock.mockResolvedValue({ url: "https://slack.com/oauth/v2/authorize?x=1" });
  redeemMock.mockReset();
  setUrl("", "");
});

afterEach(() => {
  useAuthMock.mockReset();
});

describe("LoginPage — Discord /tekijin join (#66)", () => {
  it("redeems the one-time token, adopts the bearer token and goes home", async () => {
    const adoptToken = vi.fn().mockResolvedValue(undefined);
    useAuthMock.mockReturnValue(auth({ adoptToken }));
    redeemMock.mockResolvedValue({ access_token: "jwt-xyz", token_type: "bearer" });
    setUrl("", "#discord_login=once-123");

    render(<LoginPage />);

    await waitFor(() => expect(redeemMock).toHaveBeenCalledWith("once-123"));
    await waitFor(() => expect(adoptToken).toHaveBeenCalledWith("jwt-xyz"));
    await waitFor(() => expect(replaceMock).toHaveBeenCalledWith("/"));
    expect(window.location.hash).toBe("");
  });

  it("asks for a fresh /tekijin join when the token is spent or expired", async () => {
    useAuthMock.mockReturnValue(auth());
    redeemMock.mockRejectedValue(new Error("401"));
    setUrl("", "#discord_login=used");

    render(<LoginPage />);

    expect(await screen.findByRole("alert")).toHaveTextContent(/tekijin join/);
  });
});

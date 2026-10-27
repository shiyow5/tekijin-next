"use client";

/**
 * Login screen (#241). Email + password → bearer token via {@link useAuth}. On
 * success the {@link AuthGate} redirects home (and we replace() as well). Renders
 * without the app header (the shell suppresses chrome on this route).
 */

import { useAuth } from "@/components/AuthProvider";
import { ApiError, getSlackLoginUrl, redeemDiscordLogin } from "@/lib/api-client";
import { useRouter } from "next/navigation";
import { type FormEvent, useEffect, useState } from "react";

function messageForError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return "メールアドレスまたはパスワードが違います。";
    if (error.status === 429) {
      return "ログイン試行が多すぎます。しばらくしてからお試しください。";
    }
  }
  return "ログインに失敗しました。時間をおいて、もう一度お試しください。";
}

/** `?slack=` outcomes the callback can send back (#406). */
function messageForSlackOutcome(outcome: string | null): string | null {
  if (outcome === "unlinked") {
    return "このSlackアカウントは、まだ社員アカウントと連携されていません。管理者にお問い合わせください。";
  }
  if (outcome === "error") return "Slackでのログインに失敗しました。もう一度お試しください。";
  return null;
}

export default function LoginPage() {
  const { login, adoptToken } = useAuth();
  const router = useRouter();
  const [slackUrl, setSlackUrl] = useState<string | null>(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canSubmit = email.trim() !== "" && password !== "" && !submitting;

  // The Slack callback hands the token back in the FRAGMENT, which never
  // reaches a server (a query parameter would be written to the access log).
  useEffect(() => {
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    // `/tekijin join` in Discord (#66) sends a one-time token the same way; it is
    // traded for a bearer token here rather than being one itself.
    const discordLogin = params.get("discord_login");
    if (discordLogin) {
      window.history.replaceState({}, "", window.location.pathname);
      redeemDiscordLogin(discordLogin)
        .then((res) => adoptToken(res.access_token))
        .then(() => router.replace("/"))
        .catch(() =>
          setError(
            "ログインURLが無効か、期限が切れています。Discordで `/tekijin join` をもう一度打ってください。",
          ),
        );
      return;
    }
    const token = params.get("slack_token");
    if (!token) {
      setError(messageForSlackOutcome(new URLSearchParams(window.location.search).get("slack")));
      return;
    }
    // Drop it from the address bar (and history) before doing anything else —
    // it is a live credential, and the user can see and copy what is up there.
    window.history.replaceState({}, "", window.location.pathname);
    adoptToken(token)
      .then(() => router.replace("/"))
      .catch(() => setError("Slackでのログインに失敗しました。もう一度お試しください。"));
  }, [adoptToken, router]);

  // Absence is the signal: the endpoint 503s while Slack login is off, so the
  // button simply does not appear rather than failing when pressed.
  useEffect(() => {
    let active = true;
    getSlackLoginUrl()
      .then((res) => {
        if (active) setSlackUrl(res.url);
      })
      .catch(() => {
        if (active) setSlackUrl(null);
      });
    return () => {
      active = false;
    };
  }, []);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      await login(email.trim(), password);
      router.replace("/");
    } catch (err) {
      setError(messageForError(err));
      setSubmitting(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-surface px-margin py-lg">
      <div className="w-full max-w-sm rounded-lg border border-outline-variant bg-surface-container-lowest p-lg shadow-sm">
        <div className="mb-lg flex justify-center">
          <img src="/tekijin-logo.png" alt="TEKIJIN" className="h-12 w-auto" />
        </div>
        <h1 className="mb-md text-center font-bold text-lg text-on-surface">ログイン</h1>

        <form onSubmit={onSubmit} className="flex flex-col gap-md" noValidate>
          <label className="flex flex-col gap-xs text-on-surface-variant text-sm">
            メールアドレス
            <input
              type="email"
              autoComplete="username"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="rounded-md border border-outline bg-surface px-sm py-xs text-base text-on-surface"
              aria-label="メールアドレス"
              required
            />
          </label>
          <label className="flex flex-col gap-xs text-on-surface-variant text-sm">
            パスワード
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="rounded-md border border-outline bg-surface px-sm py-xs text-base text-on-surface"
              aria-label="パスワード"
              required
            />
          </label>

          {error ? (
            <p role="alert" className="text-error text-sm">
              {error}
            </p>
          ) : null}

          <button
            type="submit"
            disabled={!canSubmit}
            className="rounded-md bg-primary px-md py-sm font-bold text-on-primary transition-opacity disabled:opacity-50"
          >
            {submitting ? "ログイン中…" : "ログイン"}
          </button>
        </form>

        {slackUrl ? (
          <>
            <div className="my-md flex items-center gap-sm text-on-surface-variant text-xs">
              <span className="h-px flex-1 bg-outline-variant" />
              または
              <span className="h-px flex-1 bg-outline-variant" />
            </div>
            <button
              type="button"
              onClick={() => {
                window.location.assign(slackUrl);
              }}
              className="w-full rounded-md border border-outline px-md py-sm font-bold text-on-surface transition-colors hover:bg-surface-container-low"
            >
              Slackでログイン
            </button>
          </>
        ) : null}
      </div>
    </div>
  );
}

"use client";

/**
 * Application header: product identity, global navigation, the current user, and
 * (admin only) the demo user switcher.
 *
 * Navigation (#122, unified to a single hamburger menu at every width by #391 —
 * the old desktop tab row and its own separate logout button are gone) makes
 * every main screen reachable in one click. The dashboard link is ADMIN-ONLY
 * (#241) — it aggregates everyone's activity. The brand links home; the active
 * link is marked with ``aria-current``. "質問する" is deliberately NOT in the
 * nav — it is folded into the top page's own question bar instead (#391).
 *
 * Auth (#241): the menu's first line is the acting principal's own name
 * (display-only — #391 explicitly keeps this un-clickable), followed by the
 * nav links, then the one and only logout button. The ADMIN additionally gets
 * the demo switcher OUTSIDE the menu, in the header row — choosing the acting
 * employee from the directory (``GET /employees`` via {@link useCurrentUser});
 * the asker screen's ``asker_id`` and the inbox follow the selection.
 * Switching also navigates home (#210): becoming a different person mid-flow
 * makes the previous screen meaningless, so we start over at the hub. Because
 * that discards whatever was on the previous screen, the switch is EXPLICIT
 * (#231): choosing in the select only points at someone; 切替 — or Enter — is
 * what actually switches. A native select fires `change` on every arrow key, so
 * acting on `change` meant browsing the list repeatedly threw the admin home and
 * took any unsent draft with it.
 */

import { useAuth } from "@/components/AuthProvider";
import { useCurrentUser } from "@/components/CurrentUserProvider";
import { NotificationBell } from "@/components/NotificationBell";
import { useFocusTrap } from "@/hooks/useFocusTrap";
import type { EmployeeSummary } from "@/lib/api-types";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { type ComponentType, useCallback, useEffect, useRef, useState } from "react";

function IconHome({ className }: { className: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
      className={className}
    >
      <path d="M4 11l8-7 8 7M6 10v9h5v-5h2v5h5v-9" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconHistory({ className }: { className: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
      className={className}
    >
      <path d="M7 3h7l4 4v14H7z" strokeLinejoin="round" />
      <path d="M14 3v4h4" strokeLinejoin="round" />
      <path d="M9 12h6M9 16h6" strokeLinecap="round" />
    </svg>
  );
}

function IconInboxNav({ className }: { className: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
      className={className}
    >
      <path d="M4 13l2.5-7h11L20 13v5H4v-5z" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M4 13h4l1.5 2.5h5L16 13h4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconChatNav({ className }: { className: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
      className={className}
    >
      <path d="M4 5h16v11H8l-4 4V5z" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M8 9h8M8 12h5" strokeLinecap="round" />
    </svg>
  );
}

function IconSources({ className }: { className: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
      className={className}
    >
      <path
        d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5M9 13h6M9 17h6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function IconKnowledge({ className }: { className: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
      className={className}
    >
      <path
        d="M12 6c-1.5-1.3-3.5-2-6-2v14c2.5 0 4.5.7 6 2 1.5-1.3 3.5-2 6-2V4c-2.5 0-4.5.7-6 2z"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function IconDashboardNav({ className }: { className: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
      className={className}
    >
      <path d="M4 20V4M4 20h16" strokeLinecap="round" />
      <path d="M8 20v-6M12 20V8M16 20v-9" strokeLinecap="round" />
    </svg>
  );
}

const NAV = [
  // Returns to the hub — placed first so it reads as "back to the top" right
  // below the acting principal's name in the drawer.
  { href: "/", label: "トップ", icon: IconHome, adminOnly: false },
  { href: "/history", label: "質問履歴", icon: IconHistory, adminOnly: false },
  { href: "/inbox", label: "受信箱", icon: IconInboxNav, adminOnly: false },
  // Chat is per-person (only your own accepted threads), so unlike the dashboard
  // it is NOT admin-only (#224).
  { href: "/chat", label: "チャット", icon: IconChatNav, adminOnly: false },
  // Company-wide (not scoped to the acting user), but still NOT admin-only:
  // the point is every user can discover someone else's past answer (#293, #301).
  { href: "/knowledge", label: "ナレッジ", icon: IconKnowledge, adminOnly: false },
  // The organization's documents and the form to add one (#72).
  { href: "/sources", label: "資料", icon: IconSources, adminOnly: false },
  { href: "/dashboard", label: "ダッシュボード", icon: IconDashboardNav, adminOnly: true },
] as const;

function isActive(pathname: string, href: string): boolean {
  return pathname === href || pathname.startsWith(`${href}/`);
}

function IconMenu() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
      className="h-6 w-6"
    >
      <path d="M4 7h16M4 12h16M4 17h16" strokeLinecap="round" />
    </svg>
  );
}

function IconClose() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      aria-hidden="true"
      className="h-6 w-6"
    >
      <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
    </svg>
  );
}

function NavLinks({
  items,
  pathname,
  onNavigate,
  className,
}: {
  items: readonly { href: string; label: string; icon: ComponentType<{ className: string }> }[];
  pathname: string;
  onNavigate?: () => void;
  className: string;
}) {
  return (
    // `divide-y` + a border framing the whole list turns the items into a
    // clearly separated block (each entry its own row), rather than a loose
    // stack of pills floating in the drawer's own padding. Every row keeps a
    // 4px left border — transparent unless active — so the accent bar doesn't
    // shift the icon/label horizontally when a row becomes current.
    <ul className={`divide-y divide-outline-variant border-outline-variant border-y ${className}`}>
      {items.map((item) => {
        const active = isActive(pathname, item.href);
        const Icon = item.icon;
        return (
          <li key={item.href}>
            <Link
              href={item.href}
              aria-current={active ? "page" : undefined}
              onClick={onNavigate}
              className={
                active
                  ? "flex items-center gap-sm border-l-4 border-primary bg-secondary-container px-md py-sm font-bold text-base text-on-secondary-container"
                  : "flex items-center gap-sm border-l-4 border-transparent px-md py-sm text-base text-on-surface-variant transition-colors hover:bg-surface-container-low"
              }
            >
              <Icon className="h-5 w-5 shrink-0" />
              {item.label}
            </Link>
          </li>
        );
      })}
    </ul>
  );
}

// The admin-only demo switcher (badge + label + select + retry button) stays
// in the header row at every width — the menu (#391) holds only navigation +
// identity + logout, not this demo-only control (#293/#301 review-adjacent:
// keep each surface's own concern where it belongs).
function UserSwitcher({
  loading,
  error,
  reload,
  ready,
  currentUserId,
  employees,
  onChange,
  className,
}: {
  loading: boolean;
  error: boolean;
  reload: () => void;
  ready: boolean;
  currentUserId: string | null;
  employees: EmployeeSummary[];
  onChange: (id: string) => void;
  className: string;
}) {
  // Who the select is POINTING AT, which is not yet who the app acts as (#231).
  //
  // A native <select> fires `change` for every arrow key while the popup is closed
  // — it does not wait for Enter — so switching straight from `change` turned "look
  // through the list" into one identity switch and one router.push per keypress,
  // discarding an unsent draft the admin never chose to leave.
  //
  // The first fix tried to tell browsing apart from confirming by watching the
  // events (defer a keyboard `change`, commit on Enter/blur/pointer). That is not
  // decidable from the DOM: on macOS ArrowDown OPENS the popup and the confirming
  // Enter never reaches the page, so a real selection would never commit; and
  // committing on blur meant clicking back into your own textarea still threw you
  // home — the very harm #231 reports. So the confirmation is explicit instead. It
  // costs mouse users one click on a demo-only control and is unambiguous on every
  // platform, with no event-order guessing.
  //
  // The pending choice remembers WHICH acting user it was made against, so a
  // change from elsewhere (a reload restoring the default, another tab) discards
  // it by construction rather than through a reset effect. An effect would clear
  // it one render late — and would show someone the app is no longer acting as in
  // between.
  const [pending, setPending] = useState<{ basedOn: string | null; value: string } | null>(null);
  const live = pending && pending.basedOn === currentUserId ? pending.value : null;

  const shown = live ?? currentUserId ?? "";
  // Re-selecting the current user is not a switch: nothing to confirm.
  const canApply = shown !== "" && shown !== currentUserId;

  function apply() {
    if (!canApply) {
      return;
    }
    setPending(null);
    onChange(shown);
  }

  return (
    <label
      className={className}
      aria-busy={loading}
      title="管理者のデモ機能。動作確認のため、任意の利用者になりかわって表示します。"
    >
      <span className="rounded bg-surface-container-high px-xs py-[1px] text-on-surface-variant">
        デモ用
      </span>
      <span>利用者を切替</span>
      <select
        aria-label="利用者を切替（管理者デモ機能）"
        className="rounded-md border border-outline bg-surface-container-lowest px-sm py-xs text-sm disabled:text-on-surface-variant"
        value={shown}
        disabled={!ready}
        onChange={(e) => setPending({ basedOn: currentUserId, value: e.target.value })}
        onKeyDown={(e) => {
          // Enter in a closed select submits the surrounding form by default; here
          // it is the natural "yes, this one" for a keyboard user, so it applies
          // rather than making them tab to the button.
          if (e.key === "Enter") {
            e.preventDefault();
            apply();
          }
        }}
      >
        {ready ? (
          employees.map((employee) => (
            <option key={employee.id} value={employee.id}>
              {employee.dept ? `${employee.name}（${employee.dept}）` : employee.name}
            </option>
          ))
        ) : (
          <option value="">{loading ? "読み込み中…" : "利用できません"}</option>
        )}
      </select>
      <button
        type="button"
        onClick={apply}
        disabled={!canApply}
        className="rounded-md border border-outline px-sm py-xs text-primary text-xs transition-colors hover:bg-surface-container-low disabled:border-outline-variant disabled:text-on-surface-variant"
      >
        切替
      </button>
      {error ? (
        <button
          type="button"
          onClick={reload}
          className="rounded-md border border-outline px-sm py-xs text-primary text-xs transition-colors hover:bg-surface-container-low"
        >
          利用者一覧の取得に失敗しました。再試行
        </button>
      ) : null}
    </label>
  );
}

export function AppHeader() {
  const { employees, currentUserId, setCurrentUserId, loading, error, reload, canSwitch } =
    useCurrentUser();
  const { principal, logout } = useAuth();
  const ready = employees.length > 0 && currentUserId !== null;
  const pathname = usePathname() ?? "";
  const router = useRouter();

  const handleLogout = async () => {
    await logout();
    router.replace("/login");
  };

  const nav = NAV.filter((item) => !item.adminOnly || principal?.is_admin);
  const [menuOpen, setMenuOpen] = useState(false);
  // Drives the slide-in transform separately from `menuOpen`: mounting the
  // panel already translated off-screen and flipping this on the next frame
  // is what makes the entrance animate instead of popping in at its resting
  // position. Closing un-mounts immediately (see `menuOpen ? ... : null`
  // below) — the drawer is not kept around for an exit animation.
  const [menuVisible, setMenuVisible] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!menuOpen) {
      setMenuVisible(false);
      return;
    }
    const frame = requestAnimationFrame(() => setMenuVisible(true));
    return () => cancelAnimationFrame(frame);
  }, [menuOpen]);

  // Route changes (including the switcher's "go home") always close the menu —
  // staying open after navigating away would sit there stale.
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-run only when pathname changes.
  useEffect(() => {
    setMenuOpen(false);
  }, [pathname]);

  // Shared with `Tab` cycling within the open drawer (#391 review: the new
  // full-screen overlay blocked mouse clicks on the page behind it but not
  // keyboard/AT focus, which could still tab into — and type into — the
  // hidden question bar).
  useFocusTrap(menuRef, menuOpen);

  // Every close path — Escape, the drawer's own close button, and a click on
  // the overlay — returns focus to the toggle button, not just Escape (#391
  // review: the other two paths dropped focus to `body`). `useCallback` with
  // no deps: it only touches the stable `setMenuOpen` setter and a ref.
  const closeMenu = useCallback(() => {
    setMenuOpen(false);
    menuButtonRef.current?.focus();
  }, []);

  useEffect(() => {
    if (!menuOpen) return;

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") closeMenu();
    }

    function handlePointerDown(e: MouseEvent) {
      const target = e.target as Node;
      if (menuRef.current?.contains(target) || menuButtonRef.current?.contains(target)) return;
      // `mousedown` fires before the backdrop's own `onClick` — closing here
      // first unmounts the backdrop before its click handler can run, so this
      // path (not just the backdrop's onClick) is what actually needs to
      // restore focus (#391 review: it previously dropped to `body`).
      // `preventDefault` matters here: clicking a non-focusable element like
      // the backdrop is itself a browser default action that blurs whatever
      // is currently focused, and that default runs AFTER this handler — so
      // without suppressing it, it would silently undo the `.focus()` call
      // in `closeMenu()` a moment later.
      e.preventDefault();
      closeMenu();
    }

    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("mousedown", handlePointerDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("mousedown", handlePointerDown);
    };
  }, [menuOpen, closeMenu]);

  return (
    <>
      {/* The white background spans the full viewport; the CONTENT is centred at
        `max-w-content` by the inner wrapper. Constraining the <header> itself let
        the body's tinted background show through beside it above 1440px (#250).

        `sticky top-0`: the header stays reachable on every screen, including the
        tall ones you cannot see the top of while waiting — the processing screen
        (#392's hero sends you straight there), the result screen, /dashboard and
        /knowledge all scroll well past a viewport (#415). `z-40` sits above page
        content and the 使い方 FAB (z-30) but below the nav drawer and
        ModalDialog's overlay (both z-50), so either still covers the header. The
        background is opaque, so content scrolling underneath is hidden. */}
      <header className="sticky top-0 z-40 border-outline-variant border-b bg-surface-container-lowest px-margin py-sm">
        <div className="mx-auto flex w-full max-w-content flex-wrap items-center justify-between gap-sm">
          <Link href="/" aria-label="TEKIJIN ホーム">
            {/* Transparent-background logo from Next's /public (aspect ≈ 2.8:1).
                alt carries the brand name so the link's accessible name stays
                "TEKIJIN". */}
            {/* Bigger than the original h-10: the brand is the only thing anchoring
                the header now that the nav lives in the drawer (#391). Capped at
                `h-12` below `md` because the header already wraps to two rows on
                a phone, and it is sticky (#415) — every pixel is permanently
                spent. */}
            <img src="/tekijin-logo.png" alt="TEKIJIN" className="h-12 w-auto md:h-14" />
          </Link>

          <div className="flex flex-wrap items-center gap-sm">
            {/* Decline notifications (#225). Logged out there is no acting user to
              poll for, so the bell stays out of the tree entirely. */}
            {principal ? <NotificationBell /> : null}
            {canSwitch ? (
              <UserSwitcher
                loading={loading}
                error={error}
                reload={reload}
                ready={ready}
                currentUserId={currentUserId}
                employees={employees}
                onChange={(id) => {
                  setCurrentUserId(id);
                  router.push("/");
                }}
                className="flex flex-wrap items-center gap-xs text-on-surface-variant text-xs"
              />
            ) : null}

            <button
              ref={menuButtonRef}
              type="button"
              aria-expanded={menuOpen}
              aria-controls="nav-menu"
              aria-label={menuOpen ? "メニューを閉じる" : "メニューを開く"}
              onClick={() => setMenuOpen((open) => !open)}
              className="flex h-9 w-9 items-center justify-center rounded-md text-on-surface-variant transition-colors hover:bg-surface-container-low"
            >
              {menuOpen ? <IconClose /> : <IconMenu />}
            </button>
          </div>
        </div>
      </header>

      {menuOpen ? (
        <>
          {/* Purely decorative dimming — closing on a click here is handled by
              the document-level `mousedown` listener below (`handlePointerDown`),
              not a click handler on this element: `mousedown` fires first and
              unmounts this div before a `click` here would ever get a chance to. */}
          <div aria-hidden="true" className="fixed inset-0 z-50 bg-on-surface/40" />
          {/* biome-ignore lint/a11y/useSemanticElements: role="dialog" + aria-modal
          matches this component's own Escape/focus-trap handling, the same
          reasoning as ModalDialog's overlay. */}
          <div
            role="dialog"
            id="nav-menu"
            ref={menuRef}
            aria-modal="true"
            aria-label="ナビゲーションメニュー"
            className={`fixed inset-y-0 right-0 z-50 flex w-full max-w-xs flex-col gap-md overflow-y-auto border-outline-variant border-l bg-surface-container-lowest p-lg shadow-md transition-transform duration-200 ${
              menuVisible ? "translate-x-0" : "translate-x-full"
            }`}
          >
            {/* The drawer sits above the header (it can be taller than the
                header on narrow admin-switcher widths where the header row
                wraps), so it needs its own close control — the header's
                toggle button underneath is not reachable while this is open. */}
            <button
              type="button"
              aria-label="閉じる"
              onClick={closeMenu}
              className="ml-auto flex h-9 w-9 items-center justify-center rounded-md text-on-surface-variant transition-colors hover:bg-surface-container-low"
            >
              <IconClose />
            </button>
            {/* Display-only (#391 scope: no profile-edit link) — just names who is
                acting, ahead of the nav links. */}
            {principal ? (
              <p className="px-md py-sm font-bold text-lg text-on-surface-variant">
                {principal.name}
                {principal.is_admin ? "（管理者）" : ""}
              </p>
            ) : null}
            <nav aria-label="メインナビゲーション">
              <NavLinks
                items={nav}
                pathname={pathname}
                onNavigate={() => setMenuOpen(false)}
                className="flex flex-col"
              />
            </nav>
            {principal ? (
              // `mt-auto` pins logout to the bottom of the full-height drawer,
              // rather than immediately trailing a short nav list.
              <div className="mt-auto border-outline-variant border-t pt-md">
                <button
                  type="button"
                  onClick={handleLogout}
                  className="rounded-md border border-outline px-md py-sm text-base text-on-surface-variant transition-colors hover:bg-surface-container-low"
                >
                  ログアウト
                </button>
              </div>
            ) : null}
          </div>
        </>
      ) : null}
    </>
  );
}

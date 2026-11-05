import { ArticlesScreen } from "@/components/ArticlesScreen";

/**
 * 知識 route (#112) — the organization's verified knowledge, and for the admin the
 * drafts waiting to be checked. The screen is a client component; this is a thin
 * wrapper.
 */
export default function ArticlesPage() {
  return <ArticlesScreen />;
}

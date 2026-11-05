import { ArticleDetailScreen } from "@/components/ArticleDetailScreen";

/**
 * One knowledge article (#112). Thin server wrapper: it resolves the id from the
 * path and hands it to the client screen, which loads GET /articles/{id}.
 */
export default async function ArticleDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <ArticleDetailScreen articleId={id} />;
}

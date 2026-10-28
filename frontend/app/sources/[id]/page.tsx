import { SourceDetailScreen } from "@/components/SourceDetailScreen";

/**
 * One source's full text (#74). Thin server wrapper: it resolves the id from the
 * path and hands it to the client screen, which loads GET /sources/{id}.
 */
export default async function SourceDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <SourceDetailScreen sourceId={id} />;
}

import { notFound } from "next/navigation";
import { MatchTerminal } from "@/components/MatchTerminal";
import { getMatchById, getMatches } from "@/lib/matches";

export function generateStaticParams() {
  return getMatches().map((match) => ({ id: String(match.id) }));
}

export default async function MatchPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const match = getMatchById(id);

  if (!match) {
    notFound();
  }

  return (
    <main className="mx-auto max-w-[1100px] px-4 pt-3 pb-8">
      <MatchTerminal match={match} />
    </main>
  );
}

import { MatchBoard } from "@/components/MatchBoard";
import { getMatches, groupMatchesForBoard } from "@/lib/matches";

export default function HomePage() {
  const sections = groupMatchesForBoard(getMatches());

  return (
    <main className="mx-auto max-w-[1100px] px-4 pb-10">
      <MatchBoard sections={sections} />
    </main>
  );
}

import { mockMatches } from "@/data/mockMatches";
import type { BoardSection, BoardSectionId, Match } from "@/types/match";

const SECTION_ORDER: BoardSectionId[] = ["live", "draft", "today"];

const SECTION_TITLES: Record<BoardSectionId, string> = {
  live: "LIVE",
  draft: "DRAFT",
  today: "TODAY",
};

export function getMatches(): Match[] {
  return mockMatches;
}

export function getMatchById(id: string | number): Match | undefined {
  const numericId = typeof id === "number" ? id : Number.parseInt(id, 10);
  if (!Number.isFinite(numericId)) {
    return undefined;
  }
  return mockMatches.find((match) => match.id === numericId);
}

function sectionIdForMatch(match: Match): BoardSectionId | null {
  if (match.status === "live") {
    return "live";
  }
  if (match.status === "draft") {
    return "draft";
  }
  if (match.status === "scheduled") {
    return "today";
  }
  return null;
}

export function groupMatchesForBoard(matches: Match[]): BoardSection[] {
  const buckets: Record<BoardSectionId, Match[]> = {
    live: [],
    draft: [],
    today: [],
  };

  for (const match of matches) {
    const sectionId = sectionIdForMatch(match);
    if (sectionId !== null) {
      buckets[sectionId].push(match);
    }
  }

  return SECTION_ORDER.filter((id) => buckets[id].length > 0).map((id) => ({
    id,
    title: SECTION_TITLES[id],
    matches: buckets[id],
  }));
}

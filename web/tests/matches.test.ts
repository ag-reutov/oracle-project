import { describe, expect, it } from "vitest";
import { mockMatches } from "@/data/mockMatches";
import { getMatchById, groupMatchesForBoard } from "@/lib/matches";

describe("getMatchById", () => {
  it("resolves a mocked match by numeric id", () => {
    const match = getMatchById(8462011001);
    expect(match?.radiant.shortName).toBe("FLCN");
    expect(match?.dire.shortName).toBe("PARI");
  });

  it("resolves a mocked match from a route param string", () => {
    const match = getMatchById("8462011002");
    expect(match?.status).toBe("draft");
    expect(match?.radiant.name).toBe("Team Spirit");
  });

  it("returns undefined for unknown ids", () => {
    expect(getMatchById("missing")).toBeUndefined();
    expect(getMatchById(0)).toBeUndefined();
  });
});

describe("groupMatchesForBoard", () => {
  it("groups live, draft, and scheduled matches into board sections", () => {
    const sections = groupMatchesForBoard(mockMatches);
    expect(sections.map((section) => section.id)).toEqual([
      "live",
      "draft",
      "today",
    ]);
    expect(sections[0].matches).toHaveLength(1);
    expect(sections[1].matches).toHaveLength(1);
    expect(sections[2].matches).toHaveLength(2);
  });

  it("omits finished matches from the board", () => {
    const finished = {
      ...mockMatches[0],
      id: 1,
      status: "finished" as const,
    };
    const sections = groupMatchesForBoard([finished]);
    expect(sections).toEqual([]);
  });
});

import { describe, expect, it } from "vitest";
import { lastMoveHeadline, lastProbabilityMove } from "@/lib/probability";
import { mockMatches } from "@/data/mockMatches";

describe("lastProbabilityMove", () => {
  it("returns the latest radiant probability delta", () => {
    const live = mockMatches[0];
    const move = lastProbabilityMove(live.probabilityHistory);
    expect(move?.from).toBe(0.435);
    expect(move?.to).toBe(0.472);
    expect(move?.delta).toBeCloseTo(0.037);
  });

  it("returns null when there is only a pre-game point", () => {
    const scheduled = mockMatches.find((match) => match.draft === null);
    expect(lastProbabilityMove(scheduled!.probabilityHistory)).toBeNull();
  });
});

describe("lastMoveHeadline", () => {
  it("names the side that the latest draft move favored", () => {
    expect(lastMoveHeadline(mockMatches[0])).toBe(
      "Why Team Falcons moved +3.7",
    );
  });

  it("falls back to a pre-game label when history has one point", () => {
    const scheduled = mockMatches.find((match) => match.draft === null)!;
    expect(lastMoveHeadline(scheduled)).toBe("Pre-game view");
  });
});

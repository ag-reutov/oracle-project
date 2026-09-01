import { describe, expect, it } from "vitest";
import {
  draftProgressLabel,
  TEAM_PICK_SLOTS,
  teamSlots,
} from "@/lib/draft";
import { mockMatches } from "@/data/mockMatches";

describe("draft helpers", () => {
  it("labels draft progress from completed and total picks", () => {
    const drafting = mockMatches.find((match) => match.status === "draft");
    expect(drafting?.draft).not.toBeNull();
    expect(draftProgressLabel(drafting!.draft!)).toBe("Draft 6/10");
  });

  it("always returns five slots per team, padding incomplete drafts", () => {
    const drafting = mockMatches.find((match) => match.status === "draft");
    const radiantSlots = teamSlots(drafting!.draft, "radiant");
    const direSlots = teamSlots(drafting!.draft, "dire");

    expect(radiantSlots).toHaveLength(TEAM_PICK_SLOTS);
    expect(direSlots).toHaveLength(TEAM_PICK_SLOTS);
    expect(radiantSlots.filter((slot) => slot !== null)).toHaveLength(3);
    expect(direSlots.filter((slot) => slot !== null)).toHaveLength(3);
    expect(radiantSlots[3]).toBeNull();
    expect(radiantSlots[4]).toBeNull();
  });

  it("returns five empty slots when draft is null", () => {
    const slots = teamSlots(null, "radiant");
    expect(slots).toEqual([null, null, null, null, null]);
  });
});

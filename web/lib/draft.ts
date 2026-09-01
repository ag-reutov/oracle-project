import type { DraftPick, DraftState, Side } from "@/types/match";

export const TEAM_PICK_SLOTS = 5;

export function draftProgressLabel(draft: DraftState): string {
  return `Draft ${draft.picksCompleted}/${draft.totalPicks}`;
}

export function picksForTeam(draft: DraftState, team: Side): DraftPick[] {
  return draft.picks
    .filter((pick) => pick.team === team)
    .sort((left, right) => left.order - right.order);
}

export function teamSlots(
  draft: DraftState | null,
  team: Side,
): Array<DraftPick | null> {
  const picks = draft === null ? [] : picksForTeam(draft, team);
  return Array.from({ length: TEAM_PICK_SLOTS }, (_, index) => picks[index] ?? null);
}

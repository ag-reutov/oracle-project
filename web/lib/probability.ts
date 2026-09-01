import type { Match, PredictionPoint, Side, TeamSnapshot } from "@/types/match";
import { formatSignedPoints } from "@/lib/format";

export interface ProbabilityMove {
  from: number;
  to: number;
  delta: number;
}

export function lastProbabilityMove(
  history: PredictionPoint[],
): ProbabilityMove | null {
  if (history.length < 2) {
    return null;
  }
  const from = history[history.length - 2].radiantProbability;
  const to = history[history.length - 1].radiantProbability;
  return { from, to, delta: to - from };
}

export function favoredSide(radiantProbability: number): Side {
  return radiantProbability >= 0.5 ? "radiant" : "dire";
}

export function teamForSide(match: Match, side: Side): TeamSnapshot {
  return side === "radiant" ? match.radiant : match.dire;
}

export function lastMoveHeadline(match: Match): string {
  const move = lastProbabilityMove(match.probabilityHistory);
  if (move === null || move.delta === 0) {
    return "Pre-game view";
  }
  const movedSide: Side = move.delta > 0 ? "radiant" : "dire";
  const team = teamForSide(match, movedSide);
  const magnitude = formatSignedPoints(Math.abs(move.delta));
  return `Why ${team.name} moved ${magnitude}`;
}

export function historyDeltas(history: PredictionPoint[]): Array<number | null> {
  return history.map((point, index) => {
    if (index === 0) {
      return null;
    }
    return point.radiantProbability - history[index - 1].radiantProbability;
  });
}

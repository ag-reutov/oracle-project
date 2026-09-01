export type MatchStatus = "scheduled" | "draft" | "live" | "finished";

export type Side = "radiant" | "dire";

export interface TeamSnapshot {
  id: number;
  name: string;
  shortName: string;
  elo: number;
}

export interface MatchPrediction {
  radiantProbability: number;
  direProbability: number;
  fairOddsRadiant: number;
  fairOddsDire: number;
}

export interface DraftPick {
  order: number;
  team: Side;
  heroId: number;
  heroName: string;
}

export interface DraftState {
  picksCompleted: number;
  totalPicks: number;
  picks: DraftPick[];
}

export interface PredictionPoint {
  label: string;
  radiantProbability: number;
}

export interface ModelSignal {
  id: string;
  direction: Side;
  label: string;
  magnitude?: number;
}

export interface TeamComparisonMetrics {
  teamElo: [number, number];
  recentEloChange: [number, number];
  rosterRating: [number, number];
  patchPerformance: [number, number];
}

export interface Match {
  id: number;
  tournament: string;
  bestOf: number;
  startTime: string;
  status: MatchStatus;
  radiant: TeamSnapshot;
  dire: TeamSnapshot;
  prediction: MatchPrediction;
  comparison: TeamComparisonMetrics;
  draft: DraftState | null;
  probabilityHistory: PredictionPoint[];
  signals: ModelSignal[];
}

export type BoardSectionId = "live" | "draft" | "today";

export interface BoardSection {
  id: BoardSectionId;
  title: string;
  matches: Match[];
}

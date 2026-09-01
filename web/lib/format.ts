const PERCENT_DIGITS = 1;
const ODDS_DIGITS = 2;

/** Format a 0–1 probability as a one-decimal percentage, e.g. 0.472 → "47.2%". */
export function formatPercent(probability: number): string {
  return `${(probability * 100).toFixed(PERCENT_DIGITS)}%`;
}

/** Format decimal odds to two places, e.g. 2.118 → "2.12". */
export function formatOdds(odds: number): string {
  return odds.toFixed(ODDS_DIGITS);
}

/** Format a 0–1 probability delta as signed percentage points, e.g. 0.037 → "+3.7". */
export function formatSignedPoints(delta: number): string {
  const points = delta * 100;
  const absolute = Math.abs(points).toFixed(PERCENT_DIGITS);
  if (points > 0) {
    return `+${absolute}`;
  }
  if (points < 0) {
    return `−${absolute}`;
  }
  return absolute;
}

export function formatSignedInt(value: number): string {
  if (value > 0) {
    return `+${value}`;
  }
  return String(value);
}

export function formatPatchPercent(rate: number): string {
  return `${Math.round(rate * 100)}%`;
}

/** Clock in UTC so board times are deterministic across environments. */
export function formatClockUtc(iso: string): string {
  const date = new Date(iso);
  const hours = String(date.getUTCHours()).padStart(2, "0");
  const minutes = String(date.getUTCMinutes()).padStart(2, "0");
  return `${hours}:${minutes} UTC`;
}

export function formatBestOf(bestOf: number): string {
  return `BO${bestOf}`;
}

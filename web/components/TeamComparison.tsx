import { formatPatchPercent, formatSignedInt } from "@/lib/format";
import type { Match, TeamComparisonMetrics } from "@/types/match";

type ComparisonKey = keyof TeamComparisonMetrics;
type FormatKind = "int" | "signed" | "percent";

const ROWS: Array<{ key: ComparisonKey; label: string; kind: FormatKind }> = [
  { key: "teamElo", label: "Team Elo", kind: "int" },
  { key: "recentEloChange", label: "Recent Elo Δ", kind: "signed" },
  { key: "rosterRating", label: "Roster rating", kind: "int" },
  { key: "patchPerformance", label: "Patch", kind: "percent" },
];

function formatValue(value: number, kind: FormatKind): string {
  if (kind === "signed") {
    return formatSignedInt(value);
  }
  if (kind === "percent") {
    return formatPatchPercent(value);
  }
  return String(value);
}

function leadingIndex(values: [number, number]): 0 | 1 | null {
  if (values[0] === values[1]) {
    return null;
  }
  return values[0] > values[1] ? 0 : 1;
}

export function TeamComparison({ match }: { match: Match }) {
  return (
    <section aria-labelledby="comparison-heading">
      <h2
        id="comparison-heading"
        className="mb-1.5 text-[10px] font-medium tracking-[0.2em] text-muted"
      >
        COMPARISON
      </h2>
      <table className="w-full text-left text-[12px]">
        <thead>
          <tr className="text-[10px] tracking-[0.12em] text-dim uppercase">
            <th className="pb-1 font-medium">Metric</th>
            <th className="pb-1 text-right font-medium">
              {match.radiant.shortName}
            </th>
            <th className="pb-1 text-right font-medium">
              {match.dire.shortName}
            </th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map((row) => {
            const values = match.comparison[row.key];
            const leader = leadingIndex(values);
            return (
              <tr key={row.key} className="border-t border-line">
                <th className="py-1 font-normal text-muted">{row.label}</th>
                {values.map((value, index) => (
                  <td
                    key={`${row.key}-${index}`}
                    className={`py-1 text-right font-mono ${
                      leader === index ? "text-foreground" : "text-muted"
                    }`}
                  >
                    {formatValue(value, row.kind)}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

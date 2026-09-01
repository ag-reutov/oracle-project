import type { MatchPrediction, TeamSnapshot } from "@/types/match";
import { formatOdds, formatPercent } from "@/lib/format";

export function ProbabilityDisplay({
  radiant,
  dire,
  prediction,
  variant,
}: {
  radiant: TeamSnapshot;
  dire: TeamSnapshot;
  prediction: MatchPrediction;
  variant: "compact" | "hero";
}) {
  if (variant === "compact") {
    return (
      <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-center gap-x-3">
        <p className="truncate text-[14px] text-foreground">{radiant.name}</p>
        <div className="flex items-start gap-6">
          <div className="text-right">
            <p className="font-mono text-[16px] font-medium leading-none text-foreground">
              {formatPercent(prediction.radiantProbability)}
            </p>
            <p className="mt-1 font-mono text-[11px] text-muted">
              {formatOdds(prediction.fairOddsRadiant)}{" "}
              <span className="text-[9px] tracking-[0.12em] text-dim uppercase">
                fair
              </span>
            </p>
          </div>
          <div>
            <p className="font-mono text-[16px] font-medium leading-none text-foreground">
              {formatPercent(prediction.direProbability)}
            </p>
            <p className="mt-1 font-mono text-[11px] text-muted">
              {formatOdds(prediction.fairOddsDire)}{" "}
              <span className="text-[9px] tracking-[0.12em] text-dim uppercase">
                fair
              </span>
            </p>
          </div>
        </div>
        <p className="truncate text-right text-[14px] text-foreground">
          {dire.name}
        </p>
      </div>
    );
  }

  const radiantWidth = `${(prediction.radiantProbability * 100).toFixed(1)}%`;

  return (
    <div>
      <div className="grid grid-cols-2 items-end gap-4 md:grid-cols-[1fr_auto_1fr]">
        <div>
          <p className="text-[12px] tracking-[0.12em] text-muted uppercase">
            {radiant.name}
          </p>
          <p className="mt-1 font-mono text-[40px] leading-none font-medium tracking-tight md:text-[48px]">
            {formatPercent(prediction.radiantProbability)}
          </p>
        </div>
        <p className="hidden pb-1.5 text-center text-[10px] tracking-[0.24em] text-dim uppercase md:block">
          Model
        </p>
        <div className="text-right">
          <p className="text-[12px] tracking-[0.12em] text-muted uppercase">
            {dire.name}
          </p>
          <p className="mt-1 font-mono text-[40px] leading-none font-medium tracking-tight md:text-[48px]">
            {formatPercent(prediction.direProbability)}
          </p>
        </div>
      </div>
      <div className="mt-2.5 flex h-px w-full overflow-hidden bg-fill">
        <div className="bg-foreground/80" style={{ width: radiantWidth }} />
        <div className="flex-1 bg-foreground/20" />
      </div>
      <div className="mt-2 grid grid-cols-2 items-baseline gap-4 font-mono text-[15px] text-foreground md:grid-cols-[1fr_auto_1fr]">
        <p>
          <span className="mr-2 text-[10px] tracking-[0.14em] text-dim uppercase">
            Fair
          </span>
          {formatOdds(prediction.fairOddsRadiant)}
        </p>
        <p className="hidden text-[10px] tracking-[0.16em] text-dim uppercase md:block">
          Odds
        </p>
        <p className="text-right">
          <span className="mr-2 text-[10px] tracking-[0.14em] text-dim uppercase">
            Fair
          </span>
          {formatOdds(prediction.fairOddsDire)}
        </p>
      </div>
    </div>
  );
}

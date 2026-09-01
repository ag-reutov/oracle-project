import { formatPercent, formatSignedPoints } from "@/lib/format";
import { historyDeltas } from "@/lib/probability";
import type { PredictionPoint } from "@/types/match";

export function ProbabilityHistory({ history }: { history: PredictionPoint[] }) {
  const deltas = historyDeltas(history);
  const hasDraftPoints = history.some((point) => point.label !== "Pre-game");

  return (
    <section aria-labelledby="probability-history-heading">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h2
          id="probability-history-heading"
          className="text-[10px] font-medium tracking-[0.2em] text-muted"
        >
          PROBABILITY
        </h2>
        <p className="text-[10px] tracking-[0.14em] text-dim uppercase">
          Pre-game
          {hasDraftPoints ? " → Draft" : ""}
        </p>
      </div>
      <ol className="flex flex-wrap items-end gap-y-2">
        {history.map((point, index) => {
          const delta = deltas[index];
          return (
            <li key={`${point.label}-${index}`} className="flex items-end">
              {index > 0 ? (
                <span className="mb-[18px] px-2 font-mono text-[13px] text-dim">
                  →
                </span>
              ) : null}
              <div className="min-w-[3.75rem]">
                <p className="text-[10px] tracking-[0.12em] text-dim uppercase">
                  {point.label}
                </p>
                <p className="mt-0.5 font-mono text-[17px] leading-none text-foreground">
                  {formatPercent(point.radiantProbability)}
                </p>
                {delta !== null ? (
                  <p
                    className={`mt-1 font-mono text-[12px] leading-none ${
                      delta > 0
                        ? "text-up"
                        : delta < 0
                          ? "text-down"
                          : "text-dim"
                    }`}
                  >
                    {formatSignedPoints(delta)}
                  </p>
                ) : (
                  <p className="mt-1 font-mono text-[12px] leading-none text-dim">
                    —
                  </p>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

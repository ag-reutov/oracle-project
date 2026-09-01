import { formatPercent, formatSignedPoints } from "@/lib/format";
import { historyDeltas } from "@/lib/probability";
import type { DraftState, PredictionPoint } from "@/types/match";

function yPercent(value: number, lo: number, range: number): number {
  return ((value - lo) / range) * 100;
}

function Trajectory({
  values,
  deltas,
}: {
  values: number[];
  deltas: Array<number | null>;
}) {
  const count = values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 0.04;
  const lo = min - span * 0.22;
  const hi = max + span * 0.22;
  const range = hi - lo;

  const points = values.map((value, index) => ({
    x: ((index + 0.5) / count) * 100,
    y: 100 - yPercent(value, lo, range),
  }));

  return (
    <div className="relative h-14 w-full">
      <svg
        className="absolute inset-0 h-full w-full overflow-visible"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        {points.slice(1).map((point, index) => {
          const previous = points[index];
          const delta = deltas[index + 1] ?? 0;
          const tone =
            delta > 0 ? "stroke-up" : delta < 0 ? "stroke-down" : "stroke-muted";
          return (
            <line
              key={`seg-${index}`}
              x1={previous.x}
              y1={previous.y}
              x2={point.x}
              y2={point.y}
              className={tone}
              strokeWidth="1.4"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>
      <div
        className="absolute inset-0 grid"
        style={{ gridTemplateColumns: `repeat(${count}, minmax(0, 1fr))` }}
      >
        {points.map((point, index) => {
          const delta = deltas[index];
          const tone =
            delta === null || delta === 0
              ? "bg-muted"
              : delta > 0
                ? "bg-up"
                : "bg-down";
          return (
            <div key={`dot-${index}`} className="relative">
              <span
                className={`absolute left-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full ${tone}`}
                style={{ top: `${point.y}%` }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ProbabilityHistory({
  history,
  draft,
}: {
  history: PredictionPoint[];
  draft?: DraftState | null;
}) {
  const deltas = historyDeltas(history);
  const hasDraftPoints = history.some((point) => point.label !== "Pre-game");
  const draftCheckpoints = history.filter((point) => point.label !== "Pre-game");
  const selectedCheckpoints =
    draft !== null &&
    draft !== undefined &&
    draftCheckpoints.length < draft.picksCompleted;
  const values = history.map((point) => point.radiantProbability);
  const columnCount = Math.max(history.length, 1);

  return (
    <section aria-labelledby="probability-history-heading">
      <div className="mb-2 flex items-baseline justify-between gap-3">
        <h2
          id="probability-history-heading"
          className="text-[10px] font-medium tracking-[0.2em] text-muted"
        >
          PROBABILITY MOVEMENT
        </h2>
        <p className="text-[10px] tracking-[0.14em] text-dim uppercase">
          Pre-game
          {hasDraftPoints ? " → Draft" : ""}
        </p>
      </div>
      <div
        className="grid"
        style={{ gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))` }}
      >
        {history.map((point) => (
          <p
            key={`pct-${point.label}`}
            className="text-center font-mono text-[15px] leading-none text-foreground"
          >
            {formatPercent(point.radiantProbability)}
          </p>
        ))}
      </div>
      <Trajectory values={values} deltas={deltas} />
      <div
        className="grid"
        style={{ gridTemplateColumns: `repeat(${columnCount}, minmax(0, 1fr))` }}
      >
        {history.map((point, index) => {
          const delta = deltas[index];
          return (
            <div key={`meta-${point.label}`} className="text-center">
              <p className="text-[10px] tracking-[0.12em] text-dim uppercase">
                {point.label}
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
          );
        })}
      </div>
      {selectedCheckpoints ? (
        <p className="mt-2 text-[10px] text-dim">Selected model checkpoints</p>
      ) : null}
    </section>
  );
}

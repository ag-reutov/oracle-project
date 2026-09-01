import { lastMoveHeadline } from "@/lib/probability";
import type { Match } from "@/types/match";

export function ModelSignals({ match }: { match: Match }) {
  return (
    <section aria-labelledby="signals-heading">
      <h2
        id="signals-heading"
        className="text-[10px] font-medium tracking-[0.2em] text-muted"
      >
        SIGNALS
      </h2>
      <p className="mt-1.5 text-[14px] text-foreground">
        {lastMoveHeadline(match)}
      </p>
      <ul className="mt-2.5 border-t border-line">
        {match.signals.map((signal) => {
          const supportsRadiant = signal.direction === "radiant";
          return (
            <li
              key={signal.id}
              className="flex items-baseline gap-4 border-b border-line py-2.5"
            >
              <span
                className={`w-3 shrink-0 font-mono text-[13px] ${
                  supportsRadiant ? "text-up" : "text-down"
                }`}
                aria-hidden="true"
              >
                {supportsRadiant ? "+" : "−"}
              </span>
              <p className="min-w-0 flex-1 text-[13px] leading-5 text-foreground/90">
                {signal.label}
              </p>
              {signal.magnitude !== undefined ? (
                <span className="ml-2 shrink-0 font-mono text-[12px] text-muted">
                  {(signal.magnitude * 100).toFixed(1)}
                </span>
              ) : null}
            </li>
          );
        })}
      </ul>
      <p className="mt-2 text-[10px] text-dim">
        Mock explanations · not live model output
      </p>
    </section>
  );
}

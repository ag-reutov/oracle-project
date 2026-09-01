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
      <p className="mt-1 text-[13px] text-foreground">{lastMoveHeadline(match)}</p>
      <ul className="mt-2 border-t border-line">
        {match.signals.map((signal) => {
          const supportsRadiant = signal.direction === "radiant";
          return (
            <li
              key={signal.id}
              className="flex items-baseline gap-2.5 border-b border-line py-1.5"
            >
              <span
                className={`w-3 shrink-0 font-mono text-[13px] ${
                  supportsRadiant ? "text-up" : "text-down"
                }`}
                aria-hidden="true"
              >
                {supportsRadiant ? "+" : "−"}
              </span>
              <p className="flex-1 text-[12px] leading-4 text-foreground/90">
                {signal.label}
              </p>
              {signal.magnitude !== undefined ? (
                <span className="font-mono text-[11px] text-muted">
                  {(signal.magnitude * 100).toFixed(1)}
                </span>
              ) : null}
            </li>
          );
        })}
      </ul>
      <p className="mt-1.5 text-[10px] text-dim">
        Mock explanations · not live model output
      </p>
    </section>
  );
}

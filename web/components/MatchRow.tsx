import Link from "next/link";
import { ProbabilityDisplay } from "@/components/ProbabilityDisplay";
import { StatusBadge } from "@/components/StatusBadge";
import { draftProgressLabel } from "@/lib/draft";
import type { Match } from "@/types/match";

export function MatchRow({ match }: { match: Match }) {
  const label = `${match.radiant.name} vs ${match.dire.name}`;

  return (
    <Link
      href={`/matches/${match.id}`}
      aria-label={label}
      className="block border-b border-line py-2.5 transition-colors hover:bg-white/[0.03]"
    >
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <p className="truncate text-[10px] tracking-[0.14em] text-dim uppercase">
          {match.tournament}
        </p>
        <div className="flex shrink-0 items-baseline gap-2.5">
          {match.draft !== null ? (
            <span className="font-mono text-[11px] text-muted">
              {draftProgressLabel(match.draft)}
            </span>
          ) : null}
          <StatusBadge status={match.status} startTime={match.startTime} />
        </div>
      </div>
      <ProbabilityDisplay
        radiant={match.radiant}
        dire={match.dire}
        prediction={match.prediction}
        variant="compact"
      />
    </Link>
  );
}

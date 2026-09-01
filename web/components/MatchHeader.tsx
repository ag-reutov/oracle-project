import Link from "next/link";
import { ProbabilityDisplay } from "@/components/ProbabilityDisplay";
import { StatusBadge } from "@/components/StatusBadge";
import { formatBestOf } from "@/lib/format";
import type { Match } from "@/types/match";

export function MatchHeader({ match }: { match: Match }) {
  return (
    <header>
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <p className="text-[11px] tracking-[0.12em] text-muted uppercase">
          <Link href="/" className="text-dim hover:text-muted">
            Board
          </Link>
          <span className="mx-2 text-dim">/</span>
          <span>{match.tournament}</span>
          <span className="mx-2 text-dim">·</span>
          <span>{formatBestOf(match.bestOf)}</span>
        </p>
        <StatusBadge status={match.status} startTime={match.startTime} />
      </div>
      <div className="mt-3">
        <ProbabilityDisplay
          radiant={match.radiant}
          dire={match.dire}
          prediction={match.prediction}
          variant="hero"
        />
      </div>
    </header>
  );
}

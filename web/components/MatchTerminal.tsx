import { DraftBoard } from "@/components/DraftBoard";
import { MatchHeader } from "@/components/MatchHeader";
import { ModelSignals } from "@/components/ModelSignals";
import { ProbabilityHistory } from "@/components/ProbabilityHistory";
import { TeamComparison } from "@/components/TeamComparison";
import type { Match } from "@/types/match";

export function MatchTerminal({ match }: { match: Match }) {
  return (
    <article>
      <MatchHeader match={match} />
      <div className="mt-4 border-t border-line pt-3">
        <ProbabilityHistory history={match.probabilityHistory} />
      </div>
      <div className="mt-4 grid gap-6 border-t border-line pt-3 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
        <ModelSignals match={match} />
        <DraftBoard
          radiant={match.radiant}
          dire={match.dire}
          draft={match.draft}
        />
      </div>
      <div className="mt-4 border-t border-line pt-3">
        <TeamComparison match={match} />
      </div>
    </article>
  );
}

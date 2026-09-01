import { MatchRow } from "@/components/MatchRow";
import type { BoardSection } from "@/types/match";

export function MatchBoard({ sections }: { sections: BoardSection[] }) {
  return (
    <div>
      {sections.map((section) => (
        <section key={section.id} aria-labelledby={`section-${section.id}`}>
          <h2
            id={`section-${section.id}`}
            className="mt-5 mb-0 pt-1 text-[10px] font-medium tracking-[0.2em] text-muted"
          >
            {section.title}
          </h2>
          <div className="border-t border-line">
            {section.matches.map((match) => (
              <MatchRow key={match.id} match={match} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

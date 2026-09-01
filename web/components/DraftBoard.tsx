import { draftProgressLabel, teamSlots } from "@/lib/draft";
import type { DraftPick, DraftState, TeamSnapshot } from "@/types/match";

function Slot({ pick }: { pick: DraftPick | null }) {
  if (pick === null) {
    return (
      <li
        aria-label="Empty pick"
        className="flex h-7 items-center font-mono text-[13px] text-dim"
      >
        ?
      </li>
    );
  }

  return (
    <li className="flex h-7 items-center gap-2">
      <span className="w-4 font-mono text-[11px] text-dim">
        {String(pick.order).padStart(2, "0")}
      </span>
      <span className="truncate text-[13px] text-foreground">{pick.heroName}</span>
    </li>
  );
}

function TeamColumn({
  team,
  slots,
}: {
  team: TeamSnapshot;
  slots: Array<DraftPick | null>;
}) {
  return (
    <div>
      <p className="mb-0.5 text-[10px] tracking-[0.14em] text-dim uppercase">
        {team.shortName}
      </p>
      <ol>
        {slots.map((pick, index) => (
          <Slot key={`${team.id}-${index}`} pick={pick} />
        ))}
      </ol>
    </div>
  );
}

export function DraftBoard({
  radiant,
  dire,
  draft,
}: {
  radiant: TeamSnapshot;
  dire: TeamSnapshot;
  draft: DraftState | null;
}) {
  const caption =
    draft === null ? "Draft not started" : draftProgressLabel(draft);

  return (
    <section aria-labelledby="draft-heading">
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <h2
          id="draft-heading"
          className="text-[10px] font-medium tracking-[0.2em] text-muted"
        >
          DRAFT
        </h2>
        <p className="font-mono text-[11px] text-muted">{caption}</p>
      </div>
      <div className="grid grid-cols-2 gap-6 border-t border-line pt-1.5">
        <TeamColumn team={radiant} slots={teamSlots(draft, "radiant")} />
        <TeamColumn team={dire} slots={teamSlots(draft, "dire")} />
      </div>
    </section>
  );
}

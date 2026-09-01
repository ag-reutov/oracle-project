import type { MatchStatus } from "@/types/match";
import { formatClockUtc } from "@/lib/format";

export function StatusBadge({
  status,
  startTime,
}: {
  status: MatchStatus;
  startTime: string;
}) {
  if (status === "live") {
    return (
      <span className="inline-flex items-center gap-1.5 text-[11px] font-medium tracking-[0.14em] text-live uppercase">
        <span className="live-dot h-1.5 w-1.5 rounded-full bg-live" />
        Live
      </span>
    );
  }

  if (status === "draft") {
    return (
      <span className="text-[11px] font-medium tracking-[0.14em] text-foreground uppercase">
        Draft
      </span>
    );
  }

  if (status === "finished") {
    return (
      <span className="text-[11px] tracking-[0.14em] text-dim uppercase">
        Final
      </span>
    );
  }

  return (
    <time
      dateTime={startTime}
      className="font-mono text-[12px] text-muted"
    >
      {formatClockUtc(startTime)}
    </time>
  );
}

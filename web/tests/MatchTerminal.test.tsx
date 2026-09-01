import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DraftBoard } from "@/components/DraftBoard";
import { MatchTerminal } from "@/components/MatchTerminal";
import { getMatchById } from "@/lib/matches";

describe("Match terminal", () => {
  it("opens the correct mocked match by id", () => {
    const match = getMatchById("8462011001");
    expect(match).toBeDefined();
    render(<MatchTerminal match={match!} />);

    expect(screen.getByText("FISSURE Playground 2")).toBeInTheDocument();
    expect(screen.getByText("BO3")).toBeInTheDocument();
    expect(screen.getAllByText("Team Falcons").length).toBeGreaterThan(0);
    expect(screen.getAllByText("PARIVISION").length).toBeGreaterThan(0);
  });

  it("renders the current model probabilities and fair odds", () => {
    const match = getMatchById(8462011001)!;
    render(<MatchTerminal match={match} />);

    expect(screen.getAllByText("47.2%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("52.8%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("2.12").length).toBeGreaterThan(0);
    expect(screen.getAllByText("1.89").length).toBeGreaterThan(0);
  });

  it("shows how the prediction moved through the draft", () => {
    const match = getMatchById(8462011001)!;
    render(<MatchTerminal match={match} />);

    expect(screen.getAllByText("44.1%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("+1.7").length).toBeGreaterThan(0);
    expect(screen.getAllByText("−2.3").length).toBeGreaterThan(0);
    expect(screen.getAllByText("+3.7").length).toBeGreaterThan(0);
    expect(
      screen.getByText("Why Team Falcons moved +3.7"),
    ).toBeInTheDocument();
  });
});

describe("Draft board rendering", () => {
  it("renders incomplete draft slots as empty picks", () => {
    const match = getMatchById(8462011002)!;
    render(
      <DraftBoard
        radiant={match.radiant}
        dire={match.dire}
        draft={match.draft}
      />,
    );

    expect(screen.getByText("Draft 6/10")).toBeInTheDocument();
    expect(screen.getByText("Pudge")).toBeInTheDocument();
    expect(screen.getByText("Queen of Pain")).toBeInTheDocument();
    expect(screen.getByText("Nature's Prophet")).toBeInTheDocument();
    expect(screen.getAllByLabelText("Empty pick")).toHaveLength(4);
  });

  it("renders five empty slots when draft is null", () => {
    const match = getMatchById(8462011003)!;
    expect(match.draft).toBeNull();

    render(<MatchTerminal match={match} />);

    expect(screen.getByText("Draft not started")).toBeInTheDocument();
    expect(screen.getAllByLabelText("Empty pick")).toHaveLength(10);
    expect(screen.getByText("16:00 UTC")).toBeInTheDocument();
    expect(screen.getByText("Pre-game view")).toBeInTheDocument();
  });
});

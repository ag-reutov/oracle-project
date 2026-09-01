import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import HomePage from "@/app/page";
import { mockMatches } from "@/data/mockMatches";

describe("Match board", () => {
  it("renders mocked matches grouped into board sections", () => {
    render(<HomePage />);

    expect(screen.getByRole("heading", { name: "LIVE" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "DRAFT" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "TODAY" })).toBeInTheDocument();

    expect(screen.getByText("FISSURE Playground 2")).toBeInTheDocument();
    expect(screen.getByText("ESL One Birmingham")).toBeInTheDocument();
    expect(screen.getByText("PGL Wallachia")).toBeInTheDocument();
    expect(screen.getByText("BLAST Slam V")).toBeInTheDocument();

    expect(
      screen.getByRole("link", { name: "Team Falcons vs PARIVISION" }),
    ).toHaveAttribute("href", "/matches/8462011001");
    expect(
      screen.getByRole("link", { name: "Team Spirit vs Tundra Esports" }),
    ).toHaveAttribute("href", "/matches/8462011002");
    expect(
      screen.getByRole("link", { name: "Gaimin Gladiators vs Team Liquid" }),
    ).toHaveAttribute("href", "/matches/8462011003");
  });

  it("shows model probabilities and fair odds on the board", () => {
    render(<HomePage />);

    expect(screen.getAllByText("47.2%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("52.8%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("2.12").length).toBeGreaterThan(0);
    expect(screen.getAllByText("1.89").length).toBeGreaterThan(0);
  });

  it("shows draft progress for live and drafting matches", () => {
    render(<HomePage />);

    expect(screen.getAllByText("Draft 10/10").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Draft 6/10").length).toBeGreaterThan(0);
  });

  it("omits draft progress when a scheduled match has no draft", () => {
    render(<HomePage />);

    const scheduled = mockMatches.find(
      (match) => match.id === 8462011003,
    )!;
    const row = screen.getByRole("link", {
      name: `${scheduled.radiant.name} vs ${scheduled.dire.name}`,
    });
    expect(row).not.toHaveTextContent("Draft");
    expect(row).toHaveTextContent("16:00 UTC");
  });
});

import { describe, expect, it } from "vitest";
import {
  formatBestOf,
  formatClockUtc,
  formatOdds,
  formatPercent,
  formatSignedInt,
  formatSignedPoints,
} from "@/lib/format";

describe("formatPercent", () => {
  it("renders a 0–1 probability as a one-decimal percentage", () => {
    expect(formatPercent(0.472)).toBe("47.2%");
    expect(formatPercent(0.528)).toBe("52.8%");
  });
});

describe("formatOdds", () => {
  it("renders decimal odds to two places", () => {
    expect(formatOdds(2.12)).toBe("2.12");
    expect(formatOdds(1.8)).toBe("1.80");
  });
});

describe("formatSignedPoints", () => {
  it("renders probability deltas as signed percentage points", () => {
    expect(formatSignedPoints(0.037)).toBe("+3.7");
    expect(formatSignedPoints(-0.023)).toBe("−2.3");
    expect(formatSignedPoints(0)).toBe("0.0");
  });
});

describe("formatSignedInt", () => {
  it("keeps the sign on Elo changes", () => {
    expect(formatSignedInt(31)).toBe("+31");
    expect(formatSignedInt(-24)).toBe("-24");
    expect(formatSignedInt(0)).toBe("0");
  });
});

describe("formatClockUtc", () => {
  it("formats ISO timestamps in UTC", () => {
    expect(formatClockUtc("2026-09-01T16:00:00.000Z")).toBe("16:00 UTC");
  });
});

describe("formatBestOf", () => {
  it("labels series length", () => {
    expect(formatBestOf(3)).toBe("BO3");
  });
});

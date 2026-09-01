import type { Match } from "@/types/match";

const falcons = {
  id: 8255888,
  name: "Team Falcons",
  shortName: "FLCN",
  elo: 1842,
};

const parivision = {
  id: 9247354,
  name: "PARIVISION",
  shortName: "PARI",
  elo: 1897,
};

const spirit = {
  id: 7119388,
  name: "Team Spirit",
  shortName: "TSpirit",
  elo: 1914,
};

const tundra = {
  id: 8291895,
  name: "Tundra Esports",
  shortName: "Tundra",
  elo: 1866,
};

const gaimin = {
  id: 8599101,
  name: "Gaimin Gladiators",
  shortName: "GG",
  elo: 1794,
};

const liquid = {
  id: 2163,
  name: "Team Liquid",
  shortName: "Liquid",
  elo: 1881,
};

const xtreme = {
  id: 8260983,
  name: "Xtreme Gaming",
  shortName: "XG",
  elo: 1820,
};

const aurora = {
  id: 9467224,
  name: "Aurora",
  shortName: "Aurora",
  elo: 1808,
};

export const mockMatches: Match[] = [
  {
    id: 8462011001,
    tournament: "FISSURE Playground 2",
    bestOf: 3,
    startTime: "2026-09-01T08:30:00.000Z",
    status: "live",
    radiant: falcons,
    dire: parivision,
    prediction: {
      radiantProbability: 0.472,
      direProbability: 0.528,
      fairOddsRadiant: 2.12,
      fairOddsDire: 1.89,
    },
    comparison: {
      teamElo: [1842, 1897],
      recentEloChange: [31, 68],
      rosterRating: [1871, 1902],
      patchPerformance: [0.54, 0.61],
    },
    draft: {
      picksCompleted: 10,
      totalPicks: 10,
      picks: [
        { order: 1, team: "radiant", heroId: 69, heroName: "Doom" },
        { order: 2, team: "dire", heroId: 70, heroName: "Ursa" },
        { order: 3, team: "dire", heroId: 13, heroName: "Puck" },
        { order: 4, team: "radiant", heroId: 106, heroName: "Ember Spirit" },
        { order: 5, team: "radiant", heroId: 129, heroName: "Mars" },
        { order: 6, team: "dire", heroId: 66, heroName: "Chen" },
        { order: 7, team: "dire", heroId: 65, heroName: "Batrider" },
        { order: 8, team: "radiant", heroId: 30, heroName: "Witch Doctor" },
        { order: 9, team: "radiant", heroId: 119, heroName: "Dark Willow" },
        { order: 10, team: "dire", heroId: 48, heroName: "Luna" },
      ],
    },
    probabilityHistory: [
      { label: "Pre-game", radiantProbability: 0.441 },
      { label: "Pick 2", radiantProbability: 0.458 },
      { label: "Pick 4", radiantProbability: 0.435 },
      { label: "Pick 10", radiantProbability: 0.472 },
    ],
    signals: [
      {
        id: "ember-exp",
        direction: "radiant",
        label: "Malr1ne is highly experienced on Ember Spirit",
        magnitude: 0.021,
      },
      {
        id: "patch-fit",
        direction: "radiant",
        label: "Falcons draft fits the mocked patch profile",
        magnitude: 0.012,
      },
      {
        id: "doom-ember",
        direction: "radiant",
        label: "Doom + Ember rates positively in the mock model",
        magnitude: 0.008,
      },
      {
        id: "pari-baseline",
        direction: "dire",
        label: "PARIVISION has the stronger baseline team rating",
        magnitude: 0.019,
      },
    ],
  },
  {
    id: 8462011002,
    tournament: "ESL One Birmingham",
    bestOf: 3,
    startTime: "2026-09-01T09:00:00.000Z",
    status: "draft",
    radiant: spirit,
    dire: tundra,
    prediction: {
      radiantProbability: 0.551,
      direProbability: 0.449,
      fairOddsRadiant: 1.81,
      fairOddsDire: 2.23,
    },
    comparison: {
      teamElo: [1914, 1866],
      recentEloChange: [12, -24],
      rosterRating: [1928, 1854],
      patchPerformance: [0.58, 0.52],
    },
    draft: {
      picksCompleted: 6,
      totalPicks: 10,
      picks: [
        { order: 1, team: "radiant", heroId: 14, heroName: "Pudge" },
        { order: 2, team: "dire", heroId: 19, heroName: "Tiny" },
        { order: 3, team: "dire", heroId: 20, heroName: "Vengeful Spirit" },
        { order: 4, team: "radiant", heroId: 39, heroName: "Queen of Pain" },
        { order: 5, team: "radiant", heroId: 38, heroName: "Beastmaster" },
        { order: 6, team: "dire", heroId: 53, heroName: "Nature's Prophet" },
      ],
    },
    probabilityHistory: [
      { label: "Pre-game", radiantProbability: 0.538 },
      { label: "Pick 2", radiantProbability: 0.521 },
      { label: "Pick 4", radiantProbability: 0.544 },
      { label: "Pick 6", radiantProbability: 0.551 },
    ],
    signals: [
      {
        id: "spirit-tempo",
        direction: "radiant",
        label: "Pudge + Queen of Pain is a mocked tempo-positive opener",
        magnitude: 0.014,
      },
      {
        id: "spirit-elo",
        direction: "radiant",
        label: "Spirit holds the higher team and roster ratings",
        magnitude: 0.011,
      },
      {
        id: "tundra-np",
        direction: "dire",
        label: "Nature's Prophet is a mocked patch-stable Tundra pickup",
        magnitude: 0.009,
      },
      {
        id: "tundra-tiny",
        direction: "dire",
        label: "Tiny offlane still rates above Beastmaster in this mock snapshot",
        magnitude: 0.007,
      },
    ],
  },
  {
    id: 8462011003,
    tournament: "PGL Wallachia",
    bestOf: 3,
    startTime: "2026-09-01T16:00:00.000Z",
    status: "scheduled",
    radiant: gaimin,
    dire: liquid,
    prediction: {
      radiantProbability: 0.438,
      direProbability: 0.562,
      fairOddsRadiant: 2.28,
      fairOddsDire: 1.78,
    },
    comparison: {
      teamElo: [1794, 1881],
      recentEloChange: [44, -8],
      rosterRating: [1810, 1894],
      patchPerformance: [0.57, 0.49],
    },
    draft: null,
    probabilityHistory: [{ label: "Pre-game", radiantProbability: 0.438 }],
    signals: [
      {
        id: "liquid-rating",
        direction: "dire",
        label: "Liquid roster rating still leads the pre-game baseline",
        magnitude: 0.024,
      },
      {
        id: "liquid-elo",
        direction: "dire",
        label: "Team Elo gap favors Liquid before the draft",
        magnitude: 0.018,
      },
      {
        id: "gg-form",
        direction: "radiant",
        label: "Gaimin recent Elo change is the stronger mocked form signal",
        magnitude: 0.013,
      },
      {
        id: "gg-patch",
        direction: "radiant",
        label: "Gaimin patch performance is ahead in the mock window",
        magnitude: 0.01,
      },
    ],
  },
  {
    id: 8462011004,
    tournament: "BLAST Slam V",
    bestOf: 2,
    startTime: "2026-09-01T18:30:00.000Z",
    status: "scheduled",
    radiant: xtreme,
    dire: aurora,
    prediction: {
      radiantProbability: 0.514,
      direProbability: 0.486,
      fairOddsRadiant: 1.95,
      fairOddsDire: 2.06,
    },
    comparison: {
      teamElo: [1820, 1808],
      recentEloChange: [-11, 22],
      rosterRating: [1833, 1811],
      patchPerformance: [0.51, 0.55],
    },
    draft: null,
    probabilityHistory: [{ label: "Pre-game", radiantProbability: 0.514 }],
    signals: [
      {
        id: "xg-edge",
        direction: "radiant",
        label: "Xtreme holds a thin mocked rating edge",
        magnitude: 0.008,
      },
      {
        id: "aurora-form",
        direction: "dire",
        label: "Aurora recent Elo change is the cleaner form signal",
        magnitude: 0.009,
      },
      {
        id: "aurora-patch",
        direction: "dire",
        label: "Aurora patch win rate leads in the mock sample",
        magnitude: 0.006,
      },
    ],
  },
];

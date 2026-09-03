# Research status: Slices 13–23

Production features are only `FEATURE_COLUMNS` (33 PRE_DRAFT snapshot
columns: team/player history, roster continuity, team Elo). A `FROZEN_*`
name does not promote a column into that set.

## Terms

**Methodological freeze.** A definition or shrinkage constant is accepted
as a stable causal/descriptive building block. Later slices must not
retune it opportunistically. This is not a production-feature freeze.

**Frozen benchmark spec.** A named evaluation comparison (reference vs
candidate columns) is held fixed so later experiments cannot move the
goalposts. This does not mean the candidate is accepted.

**Diagnostic-only.** A useful research result that must not be treated as
an accepted downstream predictive feature.

**Production feature.** A column actually included in `FEATURE_COLUMNS`
/ `ALL_FEATURE_COLUMNS` and the Step 4B production model path.

The Slice 9 holdout (`FROZEN_DEVELOPMENT_END` and related constants)
remains reserved. Later slices evaluate on the development frame only.

## Status matrix

| Slice | Concept | Code | In HEAD | Result | Methodological freeze | Tuning freeze | Benchmark spec freeze | Production | Diagnostic-only | Consumed by | Inconsistency |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13 | Farming target (duration-adjusted last-hit residual B) | `farming_performance_target.py` | yes | A | yes: candidate B formula | no | no | no | target diagnostics; B reused as a frozen *definition* | 14–17, 21–23 | none |
| 14 | Historical player farming state | `player_farming_state.py` | yes | A | yes: causal B construction | yes: `FROZEN_SHRINKAGE_K=5` | no | no | state is research-only | 15–16, 17, 21–23 | none |
| 15 | PRE_DRAFT farming team comparison | `player_farming_comparison.py` | yes (with 16) | construction accepted; no A/B/C win-model gate | yes: `mean_farming_shrunk_b_diff` formula | no (uses Slice 14 k) | yes: `SLICE15_FROZEN_SPECS` | no | comparison plumbing | 16 | name says frozen; it is a benchmark spec |
| 16 | Walk-forward farming vs Elo | `player_farming_benchmark.py` | yes | B (weak/mixed; retain state, do not promote) | no new definition | no | yes: `SLICE16_FROZEN_SPECS is SLICE15_FROZEN_SPECS` | no | yes | 20 records the B result | none |
| 17 | Combat target (position-adjusted damage share C) | `combat_performance_target.py` | yes | A | yes: `FROZEN_COMBAT_CANDIDATE` | no | no | no | target diagnostics; C reused as a frozen *definition* | 18–23 | none |
| 18 | Historical player combat state | `player_combat_state.py` | yes | A | yes: causal C construction | yes: `FROZEN_COMBAT_SHRINKAGE_K=20` | no | no | state is research-only | 19–23 | none |
| 19 | PRE_DRAFT combat team comparison | `player_combat_comparison.py` | **no** (working tree; HEAD exports were committed without the implementation) | construction accepted; no A/B/C win-model gate | yes: `mean_combat_shrunk_c_diff` formula | no (uses Slice 18 k) | yes: `SLICE19_FROZEN_SPECS` | no | comparison plumbing | 20 | see below |
| 20 | Walk-forward combat vs Elo | `player_combat_benchmark.py` | **no** (working tree) | B (weak/mixed; retain state, do not promote) | no new definition | no | yes: `SLICE20_FROZEN_SPECS is SLICE19_FROZEN_SPECS` | no | yes | 21 cites this as prior evidence | uncommitted, parallel to 19 |
| 21 | Hero×position resource/combat *profiles* | `hero_performance_profile.py` | yes | A | yes: farming/combat profile target + hero×position key | no (`hero_shrinkage_k_frozen=False` here) | no | no | yes (no fit score) | 22 | none |
| 22 | LPO hero×position requirement states | `hero_requirement_state.py` | yes | A | yes: LPO construction | yes: `FROZEN_HERO_FARM_SHRINKAGE_K=2`, `FROZEN_HERO_COMBAT_SHRINKAGE_K=2` | no | no | yes (no fit score) | 23 | none |
| 23 | Player × hero behavioral compatibility | `player_hero_compatibility.py` | yes | **B — suggestive but unstable** | no fit-score freeze | no | no | no | **yes; required** | none | none remaining |

## Slice 19 / 20 commit-state

`SLICE19_FROZEN_SPECS` is the Elo vs Elo+`mean_combat_shrunk_c_diff`
evaluation spec (combat analogue of `SLICE15_FROZEN_SPECS`). It is not
a different historical experiment and slice numbering was not reused.

Commit `15e83a7` (Slices 17–18) exported `SLICE19_*` and
`player_combat_comparison` from `training/__init__.py` but left the
implementation, `feature_sets.py` definitions, availability, and tests
unstaged because Slice 19 was in a parallel working tree. HEAD therefore
imported names that did not exist until the working-tree Slice 19 files
were added. Slice 20 is the walk-forward on that spec, still uncommitted,
same pattern as Slice 16 on Slice 15.

Do not renumber. Treat the working-tree Slice 19/20 files as the
canonical combat comparison/benchmark, not a replacement experiment.

## What later slices may treat as established

Reuse without retuning:

- Farming target B and `FROZEN_SHRINKAGE_K=5`
- Combat target C and `FROZEN_COMBAT_SHRINKAGE_K=20`
- Farming comparison formula `mean_farming_shrunk_b_diff`
- Combat comparison formula `mean_combat_shrunk_c_diff`
- Hero×position profile keys/targets from Slice 21
- LPO hero requirement states and hero k=2 from Slice 22
- Slice 9 holdout protocol and `SLICE9_FROZEN_SPECS`
- Slice 16 result: farming vs Elo is B (do not promote)
- Slice 20 result: combat vs Elo is B (do not promote)
- Slice 23 result: compatibility is B diagnostic-only (no production fit)

Do not treat as production or as a successful fit feature:

- Any Slice 13–23 research column
- Slice 23 compatibility terms
- `PLAYER_*_FEATURE_COLUMNS` names (evaluation plumbing, not `FEATURE_COLUMNS`)

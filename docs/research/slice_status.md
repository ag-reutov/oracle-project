# Research status: Slices 13–29

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
| 24 | Current-meta Hero × Position state | `hero_position_meta_state.py` | **no** (working tree) | **C — do not freeze** | no | no | no | no | **yes; required** | none | none |
| 25 | Causal Player × Position hero-pool state | `player_hero_pool_state.py` | **no** (working tree) | **C — do not freeze** | no | no | no | no | **yes; required** | none | prior A under estimator-specific Laplace support invalidated; common-support scoring shows expanding P×R×H worse than P×H on LL/Brier |
| 26 | Causal sequential draft-state dataset | `sequential_draft_state.py` | yes | **A — freeze construction** | yes: `before_event_t` boundary + ordered prefix state + successful-ban semantics | no | no | no | state is research-only substrate | 27 | none |
| 27 | Incremental draft-value (Elo vs Elo+checkpoint picks) | `sequential_draft_benchmark.py` | **no** (working tree) | **C — do not freeze** | no | no | no | no | **yes; required** | none | Pattern D: side-aware hero main effects worsen Elo at every checkpoint; bans worsen further |
| 28 | Causal next-pick draft-policy | `next_pick_policy.py` | **no** (working tree) | **C — do not freeze** | no | no | no | no | **yes; required** | none | Pattern D: no tested extension (prefix, bans, team, version) beats baseline_b (side + pick-index popularity). Sparse linear OVR SGD representation limitation at current data scale; does not prove absence of conditional draft structure. |
| 29 | Data-scale feasibility / learning-curve audit | `data_scale_diagnostics.py` | **no** (working tree) | **DATA + REPRESENTATION LIMITED** | no | no | no | no | **yes; required** | none | Track A: ΔLL +0.45→+0.08 with N (data-scale); C re-selected via frozen S27 fold-internal procedure (C_heroes always 0.1 in practice). Track B: frozen S28 recipe (alpha=1/(C*n)) gap stays ~+5.7–+6.1 — does not rescue as implemented; not proof every sparse linear prefix model is doomed. Expansion ~1.6–2.4k worthwhile, not claimed enough for Track A to beat Elo. |

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
- Slice 24 result: current-meta H×P is C (do not freeze a time-varying layer)
- Slice 25 result: causal P×R×H hero-pool availability is **C** under
  common-support scoring (`C_T` + fixed epsilon mixture). Do not freeze
  expanding P×R×H, last-5, or hierarchical backoff. Keep as diagnostic
  evidence that role-conditioned next-choice lift was an artifact of
  estimator-specific support normalization.
- Slice 26 result: sequential draft-state construction is **A**. Freeze
  `S_(M,t)` = state **before** event `sequence == t` (events with
  `sequence < t`), ordered event prefixes, successful-vs-unsuccessful
  ban semantics (`was_successful is not False` for availability), and
  the terminal boundary after the last event. This is a dataset/state
  freeze only — not a next-pick model and not a production feature.
- Slice 27 result: Elo + side-aware checkpoint pick main effects is **C**.
  Do not freeze the picked-hero draft block. Pattern D on development
  walk-forward: every positive-pick checkpoint raises log loss vs Elo;
  adding successful bans worsens further. Keep as diagnostic evidence
  that raw hero identity main effects are not a stable live-draft
  outcome signal beyond Elo. Do not respond by adding interactions
  inside Slice 27.
- Slice 28 result: **C — do not freeze**. Pattern D. SGDClassifier(loss='log_loss') OVR logistic; multinomial logistic abandoned pre-benchmark for computational reasons. Best baseline: baseline_b (side + pick-index popularity, LL 4.385). No tested extension beats it: baseline_c +0.650, team_tendency +0.481, prefix_picks +6.052, team_identity +5.180. Extreme SGD underperformance is a sparse linear OVR representation limitation at this data scale, not proof of absence of conditional draft structure. No frozen components.
- Slice 29 result: **DATA + REPRESENTATION LIMITED**. Track A (S27 Elo vs Elo+heroes): pooled ΔLL +0.45→+0.08 as N grows; train/eval gap shrinks. Strong data-scale evidence, but the curve has **not** demonstrated that more data will make heroes beat Elo — do not extrapolate a crossing. Method note: C is re-selected at each N via frozen Slice 27 fold-internal grid `(0.1,1,10)`; not a pure fixed-C curve. In confirmation, `C_heroes=0.1` for all 20 fold×fraction points; Elo-only `C` varied. Track B (S28 baseline_b vs prefix SGD): pooled ΔLL stays ~+5.7–+6.1 and does not systematically close. Method note: recipe freezes `C=1.0` but `alpha=1/(C*n)` so effective regularization varies with N by construction — evidence is that **more data does not rescue Slice 28 as implemented**, not that every sparse linear draft-prefix model is representation-limited. Same-regime expansion inventory ~1.6–2.4k pending T1/T2 matches (excl. TI holdout): worthwhile, **not** claimed sufficient to solve Track A. Recommendation: expand contemporary data; redesign next-pick representation/training before revisiting Slice 28.
  benchmark (`draft prefix -> next successful pick hero`). Classification
  and any freeze will be recorded after development OOS evidence.

Do not treat as production or as a successful fit feature:

- Any Slice 13–29 research column
- Slice 23 compatibility terms
- Slice 24 current-meta H×P columns
- Slice 25 pool-state / next-choice diagnostic columns
- Slice 26 sequential draft-state fields (research substrate only)
- Slice 27 checkpoint pick/ban indicator columns
- Slice 28 next-pick policy features / coefficients (research-only unless
  a later confirmation freezes a supported policy-state definition)
- `PLAYER_*_FEATURE_COLUMNS` names (evaluation plumbing, not `FEATURE_COLUMNS`)

# Figure and table inventory

Every display is sourced from frozen reviewed evidence. Counts of decisions,
conditions, horizons, anchors, or episodes are descriptive repeats; the
effective independent unit is the source state.

## Main figures

| Label | Content | Canonical asset | Effective source contract | Scope language required in caption |
|---|---|---|---|---|
| Figure 1 | Exact-state branch schematic plus raw/source-aware `R x B` flow | schematic in `manuscript.tex`; reviewed `figure_risk_benefit_flow.pdf` | 20 sources, 273 decisions; disagreement 12 sources; each direction 6 | Detour is privileged; one `glass_recovery` family; conditions are nested repeats |
| Figure 2 | Oracle-gap attribution | reviewed `figure_oracle_gap_waterfall.pdf` | 20 exposed-development sources, 273 decisions | Oracle combinations are diagnostic; no method gate pass |
| Figure 3 | Condition-level source/option support | reviewed `figure_source_support.pdf` | glass 20 sources; offpath 17; noglass 17; all strict D/R support 17 sources | all three conditions are one `glass_recovery` family; offpath/noglass are controls; 23 and 51/60 concentration visible |

All three reviewed assets live under
`results/iclr27/risk_value_decoupling_dc48ff317dad_20260829T154351Z/` and are
SHA-pinned in that directory's manifest. The LaTeX manuscript links them
directly; it does not rewrite frozen PDFs.

## Main tables

| Label | Content | Canonical evidence | Effective source contract |
|---|---|---|---|
| Table 1 | Complete three-pair same-risk witness set | `same_risk_different_decision_witnesses.csv` | exactly 3 pairs, 2 sources; exact D/R witness 1 source |
| Table 2 | Risk reference, learned methods, oracle hybrids, and Oracle | Phase 2.5B/B-R metrics plus `risk_only_policy_regret.csv` and `oracle_hybrid_waterfall.csv` | 20 exposed-development sources, 273 decisions |
| Table 3 | Statewise-to-sequential boundary | `sequential_realization_gap.csv` and pinned sequential artifacts | stage-specific 4, 9, or 8 sources; repeated units stated separately |

## Appendix tables

| Label | Content | Effective source contract |
|---|---|---|
| A1 | Frozen corpus and provenance | 20 sources / 273 decisions / 819 option rows |
| A2 | Raw and equal-source `R x B` accounting | 20 sources / 273 decisions |
| A3 | Full witness hashes, outcomes, utilities, and feature summaries | 3 pairs / 2 sources / 1 D-R source |
| A4 | Complete 11-row source-macro policy-regret table | 20 sources / 273 decisions |
| A5 | Learned selector method-gate audit | 20 source-OOF blocks |
| A6 | Condition support | glass 20; offpath 17; noglass 17, inside one family |
| A7 | Expanded sequential boundary | P2 4; P3.1 9; P3.2 8 sources |
| A8 | Supported versus unsupported claim boundary | evidence-specific source counts above |

## Caption audit rules

- Do not label `glass`, `offpath`, and `noglass` as families.
- Do not report 273 decisions without 20 independent sources nearby.
- Do not report three witnesses without the two-source limit and one-source
  Detour/Retreat limit.
- Do not show an Oracle hybrid without saying it is diagnostic only.
- Do not show P3.2 zero false interventions without explaining Base collapse.
- Do not include non-glass v1 in a scientific table or denominator.

# Supplementary simulation protocol for the IPM content governance submission

Protocol prepared on 2026-09-16 before executing the new experiment matrix. This is a locally timestamped prospective extension, not external preregistration. The earlier frozen archive remains unchanged. Its historical instruction to add no further ABM runs applied to that earlier analysis, not to this separately versioned extension authorized by the author.

## Purpose and evidentiary scope

Audit whether mechanism attribution and simulated intervention comparisons depend on assigned exposure, intervention coverage, attenuation, and model dynamics. The target call is Trustworthy Content Governance and Safe Generative AI in the Age of Large Language Models. These simulations concern the evaluation of governance assumptions; they do not measure actual LLM safety, misinformation detection accuracy, real persuasion, or policy effectiveness. Message strength and susceptibility parameters are synthetic assumptions, not empirical estimates of high/low AI quality. Opinion movement is not intrinsically harm. A harmful-target interpretation requires the explicitly hypothetical assumption that the seeded target is undesirable misinformation.

Official sources checked 2026-09-16:
- https://www.elsevier.com/events/conferences/all/information-processing-and-management-conference/track-title
- https://www.elsevier.com/events/conferences/all/information-processing-and-management-conference/participate

The call is linked to IP&MC2026. The track page lists 31 July 2026 as the conference deadline; the participation page currently advertises late-breaking posters. A separate open journal submission route is not established by these pages.

## Frozen inputs and verification

Snapshot the original model and estimator source without modification. Replay at least one archived row in each topology and two seeding regimes against recorded seeds and outputs. Confirm all three outcome definitions from code, including target-oriented signed shift and conversion conditional on initial opposition. Check the hub node-selection logic against explicit node labels. Retain original implementation and a corrected node-ID implementation as distinct versions if a discrepancy is confirmed. Do not blend corrected and frozen results.

## E1 Independent schedule by dose experiment

N=1000, T=50; topology BA/ER/WS/SBM; direct cumulative dose per capita 0.05/0.50/0.95; schedule single_shot/burst/continuous; strength 0.06/0.14/0.18; reversion 0/0.05; no disclosure or inoculation. Thirty independent replication blocks. Total 6,480 new ABM runs. Full Cartesian crossing removes schedule-dose binding. Assess complete-U factor allocations separately by reversion setting; derive forward and reversed bundled views from the full response grid without counting them as additional simulations. Evaluate equal-dose schedule contributions and raw endpoint contrasts. Strength is a numeric input, not an empirical AI category.

## E2 Governance assumption stress test

N=1000, T=50, strength=0.14, no reversion. Cross four topologies, dose 0.05/0.50, schedule single_shot/continuous and 30 replication blocks.

Fifteen main policy configurations:
1. No intervention.
2-5. Disclosure lambda 0.30/0.70, with labels retained/lost on second-hand exposure (four configurations).
6-13. Inoculation random/hub, target budget 0.05/0.20, compliance 0.50/1.00, kappa=0.20 (eight configurations).
14-15. Disclosure lambda=0.70 with retained labels plus random/hub inoculation at budget=0.05, compliance=1.00.

Main matrix 7,200 runs. If the hub index issue is confirmed, use corrected node-ID selection for the main new matrix and additionally execute all five hub policies using the frozen selector (2,400 sensitivity runs). Zero-compliance and zero-budget boundary tests are mandatory for the corrected implementation.

Primary comparisons are paired against no intervention within identical exposure cells, matched-budget random-versus-hub, disclosure label-retention versus label-loss, and the dependence of these differences on dose and schedule. Report all policies; do not select a winner after seeing outcomes. Disclosure and inoculation have different resource units: do not call their raw comparison cost effectiveness or an equal-cost ranking. Population-averaged outcome differences, intervention targeting counts, and exposure summaries are retained.

## E3 Independent Friedkin-Johnsen-inspired dynamics

A separately implemented bounded opinion process with endogenous peer averaging every step. Four topologies; doses 0.05/0.50/0.95; three schedules; message coupling 0.10/0.30; stubbornness 0.20/0.80; N=1000, T=50; R=30. 4,320 exposed simulations plus 240 topology-by-stubbornness-by-replication no-seeding controls, total 4,560. Use explicit node ordering in the sparse adjacency. For isolates, use a self-loop in the averaging operator. Record the exact update equation in MODEL_SPECIFICATION.md. Analyze intervention-induced change relative to each paired no-seeding trajectory, not spontaneous peer drift. This is an external model-family check, not a reproduction of the original ABM or empirical validation.

## E4 Known-truth estimator validation with correlated noise

Truth on a balanced 2^4 grid: mu=0.8A+0.5B+0.6AB+0.3C. Shapley allocations A=.82, B=.43, C=.09, D=0, total=1.34. Noise settings: independent homoskedastic; independent heteroskedastic; heteroskedastic common block shock; heteroskedastic structured factor-correlated block shock. R=10/30/60. 1,000 datasets per setting (12,000 datasets); retain dataset-level estimates for all factors. The first 500 datasets per setting receive 399 distinct-original-pair bootstrap draws and jackknife-normal intervals. Report bias, RMSE, negative estimate frequency, theoretical in-sample bias, coverage and its binomial Monte Carlo SE. No selection of an inference method based only on favorable results.

## Randomization and analysis

New seeds use the namespace ipm-content-governance-20260916-v1 and SHA-256. Exclude intervention/dose/strength labels from selected within-block seed keys to preserve pairing, and include experiment, topology and replication to separate independent blocks. Distinct random streams are used for initialization, network, direct seeding, sharing, noise and intervention targeting. Stream alignment can diverge after intervention-dependent branching; pairing does not imply every downstream event remains identical. Record every seed.

Use independent replication blocks as the uncertainty unit, never the nominal number of agents or cells. For raw mean differences, report paired t intervals as Monte Carlo precision summaries over 30 simulation blocks; these do not measure empirical uncertainty or control familywise error. For complete-U allocations retain signed estimates and use 1,000 distinct-pair block bootstrap percentile ranges as sensitivity summaries, not calibrated confidence intervals. Suppress ratios with nonpositive denominators. New equal-weight factorial designs define new estimands: do not merge their allocations with the old benchmark without an explicit denominator definition.

Execution will retain run-level results, configuration matrix, seed manifests, source hashes, environment versions, tests, integrity checks, plots, and English insertion text with measured results. Small implementation smoke tests are separate from the fixed full matrix. Any substantive protocol change after initial outcomes must be recorded in an addendum rather than overwriting this plan.

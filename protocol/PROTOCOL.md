# PhyRoute revised evaluation protocol

Frozen after exploratory analysis on 22 September 2026, before the revised outcome comparisons. This is not a preregistration or an independent first look at the original experiment. The two original branch exchanges and the audit comparisons were already observed. Code repairs may follow freezing, but changes to methods, data allocation or endpoints must be recorded in `AMENDMENTS.md` before dependent evaluation.

## Question and interpretation

When does a neural voltage surrogate add value over current LinDistFlow screening with exact fallback after a feeder changes, and how does verifier mismatch affect that comparison? Compare unsafe approvals, false rejections and compute jointly. Do not select a winner using unsafe approvals alone. Findings apply to the stated balanced, synthetic 33-bus model; no field-safety guarantee is claimed.

## Fixed data and topology allocation

- Three independent training runs, seeds 4101, 4201 and 4301. Each run draws 30,000 base-topology training scenarios, 10,000 separate probe-training scenarios and 10,000 separate threshold-calibration scenarios. The revised run does not reuse the original checkpoint. Data seed derivation is deterministic and recorded in each output manifest.
- The original generator's load/PV/slack-voltage ranges remain fixed. Three test seeds, 9101, 9201 and 9301, each produce 10,000 original-range injections. The same injections are paired across the original topology and all 59 branch exchanges. This gives 1,800,000 original-range topology-scenario labels. Three further 10,000-scenario PV-growth tests on the original topology give 1,830,000 labels total. Reusing labels across models or mismatch variants does not increase this count.
- The topology inventory enumerates all connected radial nonbaseline single branch exchanges. Fifteen are development topologies, including both previously explored exchanges; 44 are held out. The other 13 development exchanges are selected by the SHA256 rule recorded in `topologies.json`, without using outcomes. All 59 are evaluated; the 44 held-out exchanges are the primary generalization set.
- Train an additional physics-augmented residual MLP for each training seed on 3,000 scenarios from each of the 15 development topologies plus 3,000 from the original topology, 48,000 total. Inputs are the 71 operating features and current LinDistFlow voltage extrema. Targets are exact voltage extrema minus the LinDistFlow extrema. It therefore receives current-physics information, unlike the deliberately stale baseline. Architecture 64x64, 250 epochs, otherwise the original Adam setup. No held-out-topology labels enter fitting or threshold choice.
- Inputs are shared across methods for paired comparisons. Training runs and test draws are independent, but rows sharing an operating point or topology are not treated as independent deployments.

## Methods

Retain five-member 32x32 ensembles with independently initialized members, the original training settings and documented stopping histories. Use the margin of the ensemble-mean voltage extrema as the revised ensemble verdict; record any difference from the original mean-of-member-margins implementation. Keep member-margin spread as a descriptive ensemble score.

Compare: ensemble spread, raw neural margin, spread-normalized margin, violation-classifier confidence, learned error-to-margin probe, physics-model disagreement first (PhyRoute-D), smaller absolute physics/neural margin, approval-only signed physics routing, LinDistFlow prediction with absolute-margin fallback, LinDistFlow approval-only fallback, conservative joint approval with exact fallback, physics-augmented residual prediction with margin fallback, and always-exact evaluation. A known-topology-change fallback uses the exact solver on every changed topology and the neural prediction on the original feeder. Never describe local learned probes as full reproductions of Calibrate-Then-Delegate.

The error probe is fitted only on the dedicated probe-training split. The violation classifier uses training labels. All threshold choices use the separate calibration split. Method formulas are fixed before examining revised results.

Add a standard split-conformal absolute-residual interval around the neural margin. For alpha 0.001, 0.01 and 0.05 use the finite-sample order statistic on the independent calibration residuals; defer whenever the interval crosses the decision boundary. Report its exchangeability scope and its empirical failure under topology shift. No general shifted-risk guarantee is claimed.

## Physics mismatch

Hold the true feeder, true injections and reference labels fixed. Change only the information supplied to the LinDistFlow verifier and physics-augmented predictor:

1. Current topology, correct impedances.
2. Stale original topology and its original impedances.
3. Current topology with coherent R and X scaling by 0.8, 0.9, 0.95, 1.05, 1.1 or 1.2.
4. Current topology with fixed independent per-branch R/X factors uniformly distributed on [0.9, 1.1], with deterministic seed 77001 and topology-specific derivation.

This is nine verifier variants. Random mismatch is one recorded parameter realization per topology; do not claim an expectation over all possible parameter errors. Frozen thresholds are always learned with the nominal original-topology verifier and are not recalibrated for these variants. Measurement noise and a second feeder are outside this frozen primary study, and must be labeled separate extensions if added.

## Primary and secondary endpoints

Primary: fixed calibration-budget thresholds at 5%, 10% and 20%, evaluated across all 44 held-out exchanges, three training seeds and three test seeds. Lead with the 10% calibration setting but report all three. Record the realized escalation, unsafe count, false-alarm count, true violations, accepted count and sample count for each row. Report unsafe/all, unsafe/accepted, unsafe/violations, false alarms/feasible, accuracy and acceptance coverage.

Give macro summaries by topology, seed ranges and worst-topology values. Compare methods on paired rows; do not present a binomial interval on pooled dependent outcomes as uncertainty across networks. A topology bootstrap is conditional on this benchmark and its fixed training runs, not a population-of-grids guarantee. Report seed variation directly.

Secondary: test-score quantile budgets 5%, 10%, 20% and 30%. These are retrospective capacity-matched diagnostics, not deployed thresholds. Any first-zero result is explicitly retrospective. No threshold is chosen from test labels for the primary endpoints. No arbitrary engineering risk target is introduced after seeing outcomes.

## Numerical and compute validation

The numerical reference must converge with finite voltages; abort and report failed conditions rather than silently labeling them safe or dropping them. Test nonconvergence and invalid inputs. Cross-check five predetermined scenarios for each branch exchange against pandapower. Record numerical differences and tolerances.

Measure complete routes on varied inputs for representative nominal and changed topologies, including neural/physics inference, selection and the exact fallback. Compare always exact, physics-only, neural-margin, PhyRoute-D and the physics-augmented model. Use single BLAS threads, warm-up and repeated calls, and report hardware/software plus median and p95 times. The three training runs need not all be benchmarked for latency if this is explicitly stated.

## Reproducibility and reporting

Save protocol/config/source hashes, package versions, checkpoint arrays without executable pickle, training histories, warnings, topology assignments, numerical outputs and figure-generation code. Checkpoint reuse must validate configuration. Keep the original experiment unchanged in its sibling directory. Report all selected conditions, including unfavorable results. Do not claim causal harm, recovered energy, statutory compliance or performance on unmeasured feeders.

The paper will include a truthful AI-assistance statement covering idea development, code, audit and drafting. Author identities and affiliations are supplied by the human author. Public repository/DOI and conference-submission status are recorded only after verified external actions.

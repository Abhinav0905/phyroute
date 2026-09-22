# Independent review of the revised experiment

The revised code addresses the main design defects found in the original package. The review found no train/calibration leakage, hidden substitution of the true network into a mismatched verifier, or inconsistent validity masking in the generated comparisons. The reported error rates and escalation rates deliberately use different denominators; the paper must keep that distinction visible.

## Verified experiment structure

The three original-topology training runs use separate train, probe and calibration streams. The error probe is fitted on its dedicated 10,000-row split. Thresholds and conformal radii use the separate calibration data. Exact row comparisons found no overlap between those three splits within each training run. The independent audit reconstructed every calibration threshold and conformal radius from model weights and the saved calibration inputs.

The canonical ensemble verdict is the margin of the mean predicted voltage extrema. Member-margin spread remains the uncertainty score. The old mean-of-member-margins quantity is retained for diagnosis and is not substituted into revised decisions.

The residual model uses exactly the original topology and the 15 declared development exchanges, 3,000 training rows each. Its checkpoint records 48,000 rows, all intended topology IDs, the correct inventory hash and the current source hash. No held-out topology appears in its training records. For every verifier variant, both its input extrema and its additive linear prediction use that variant; the true network is not restored internally.

All frozen routing thresholds are computed using the nominal original-topology verifier. The test-score quantiles appear only in the separately labeled retrospective diagnostics. Conformal radii use the finite-sample order statistic of absolute calibration residuals. Their exchangeability assumptions do not extend to the shifted topology tests.

## Numerical resolution and denominator audit

The study generates 1,830,000 unique topology-scenario labels. Of these, 1,827,685 are numerically resolved and 2,315 remain unresolved. All unresolved cases occur on `close34_open01`; the counts are 802, 734 and 779 for the three test seeds. Newton-Raphson resolves a further 135 cases after the strict sweep fails. An unresolved numerical solve is not proof that the operating point lacks an AC solution.

The same validity mask applies to every model and verifier variant in a condition. Unresolved margins remain NaN. The truth array explicitly intersects the voltage-violation test with the validity mask, and metric counts are then restricted to resolved rows. No NaN comparison becomes a reported safe label. An escalated unresolved case is withheld, and unresolved cases that receive a cheap approval are counted separately.

Unsafe/all, unsafe/accepted, unsafe/violations, false alarms/feasible and acceptance coverage are conditional on solver-resolved draws. Escalation uses every generated input because an unresolved input can still incur a solve. The parent analysis intentionally replaces its displayed escalation field with `realized_escalation_all` while retaining the resolved-only value separately. The independent audit checks this replacement algebraically and against the summary aggregation. These metrics do not bound risk on the unresolved subset.

## Reproducibility checks

The audit checks the frozen protocol/configuration hashes and all nine core source hashes against run `8cb0242b95738614d4bcb048900dc07da99e04dfb46149a77b77a0ff2bacd2fc`. The model checkpoints load without pickle and pass configuration and array-integrity validation. Training histories have consistent iteration, loss and validation-history lengths.

All 183 physics caches are checked for content hashes, validity-mask consistency, reference counts, nine verifier variants and agreement between stored voltage extrema and margins. The complete output inventory is 549 evaluation CSVs with 549 rows each, or 301,401 rows. The audit checks every CSV hash, provenance identity, method/variant/mode inventory, count identity, derived rate and frozen threshold. The always-exact control has zero observed unsafe approvals and false rejections on resolved labels.

A separate inference implementation reconstructs predictions from model arrays. All 549 rows are recomputed for each of five selected condition cells: original topology, PV growth, both previously explored exchanges and the topology containing unresolved draws. Every verifier variant and each frozen/retrospective operating mode is included. This is 2,745 independently replayed comparison rows; the remaining rows receive hash, inventory, denominator and algebra checks.

The audit also reconstructs every summary family from the original CSVs: nominal held-out, mismatch, conformal, retrospective matched-budget, original/PV and no-routing results. It checks the topology means, worst-topology means and training-seed ranges. Unique reference counts are computed without multiplying shared labels by the number of trained models.

Latency is evaluated on the preset original and RC-A topologies, one trained seed and batches of 1, 64 and 512. The timed path includes prediction, input gathering, gate selection and strict-sweep fallback. It excludes checkpoint loading, topology setup and Newton-Raphson retries. Timing order rotates across methods. The independent audit checks all 30 workload cells and derives medians, p95s and means from the saved call-level measurements. These measurements do not establish latency across all 59 exchanges or across unresolved workloads.

Machine-readable findings are in `validation/revised_integrity.json`; the command is:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python validation/audit_revised.py --require-complete
```

## Remaining interpretation limits

The residual MLP has 48,000 training labels spread across 16 topologies. The stale ensemble has 30,000 original-topology labels, and its learned error probe uses a separate 10,000-row split. A difference between them cannot isolate the effect of architecture or physics features: training coverage and label budget also differ.

The 44 held-out exchanges share one underlying 33-bus feeder. Three training seeds and three test seeds improve repeatability within this benchmark; they do not establish performance across a population of distribution networks. Shared input rows, topology variants and repeated physics-only methods are dependent observations. The reported seed ranges and topology means are descriptive, not confidence intervals for arbitrary grids.

The standalone residual-checkpoint reuse function verifies its configuration but does not itself reject a changed source or inventory hash. Both hashes match in this completed run and are independently checked by the audit. A future experiment that modifies code or topology allocations must validate those provenance fields or use a new checkpoint location.

The original and revised archives, submission status, authorship details and public DOI remain separate from this numerical review. This audit made no external publication or submission.

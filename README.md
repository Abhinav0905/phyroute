# PhyRoute: Voltage Screening under Feeder Reconfiguration

**v0.2.0 software release.** Code, numerical outputs and manuscript sources for comparing neural, physics-only and hybrid voltage screening after a distribution feeder changes. Author: Kumar Abhinav. Repository: [Abhinav0905/phyroute](https://github.com/Abhinav0905/phyroute).

The companion manuscript remains an author-review draft. Author affiliation and email still require confirmation, and the manuscript has not been submitted to the conference. Software release and archival deposit do not establish peer review or acceptance.

The revised experiment asks when a neural surrogate adds value over an inexpensive physical approximation with exact fallback. It also tests what happens when the verifier's network model is wrong. The results support a tradeoff between unsafe approvals, false rejections and computation. They do not establish a safety guarantee.

## Main results

These are macro averages over **44 held-out branch exchanges**, three training/calibration runs and three independent test draws. Each routing threshold was fixed on the original-topology calibration split to target 10% escalation. The verifier uses the correct current topology and impedances.

| Method | Realized escalation | Unsafe approvals | False rejections |
|---|---:|---:|---:|
| Neural-margin routing | 10.222% | 14.8359% | 2.6839% |
| PhyRoute-D | 25.185% | 0.0606% | 0.0786% |
| LinDistFlow prediction + exact fallback | 8.658% | 0.0615% | 0.1678% |
| Physics-augmented residual MLP + exact fallback | 8.398% | 0.0065% | 0.0248% |

Escalation uses all generated inputs. Unsafe approvals use numerically resolved inputs; false rejections use resolved feasible inputs. These columns therefore have different denominators. Source: [nominal_primary.csv](results/revised/analysis/nominal_primary.csv).

Physics-only screening reaches nearly the same unsafe-approval rate as PhyRoute-D with about one third of its exact calls, while PhyRoute-D rejects fewer feasible points. The residual MLP improves both error measures in this comparison, but it has **48,000 training labels from 16 topologies**, compared with 30,000 original-topology labels for the stale ensemble. Training coverage, label budget and features all differ. The experiment cannot attribute its improvement to architecture or physics features alone.

Verifier quality matters. With the stale original topology, unsafe-approval macro averages rise to 15.441% for PhyRoute-D, 15.439% for physics-only screening and 14.860% for the residual MLP. Independence from the neural model does not make a verifier correct. See [mismatch_primary.csv](results/revised/analysis/mismatch_primary.csv).

Across the three training/calibration runs, nominal unsafe-rate macro averages range from 0.0567% to 0.0637% for PhyRoute-D, 0.0578% to 0.0646% for physics-only screening and 0.0056% to 0.0074% for the residual MLP. Each run averages the same 44 held-out exchanges and three test draws. These are descriptive seed ranges, **not confidence intervals** for other grids. All exchanges share one balanced, synthetic 33-bus feeder.

## Fixed experiment

- Training seeds: **4101, 4201, 4301**. Each run uses 30,000 original-topology training cases, 10,000 separate error-probe fitting cases and 10,000 separate threshold-calibration cases. The ensemble contains five 32x32 MLPs. Its decision uses the margin of the mean predicted voltage extrema; spread is computed across member margins.
- Topologies: the original feeder plus **59 connected radial single branch exchanges**. Fifteen exchanges are development cases, including the two explored in the earlier experiment. The other 44 are held out. The assignment rule and IDs are saved in [topologies.json](protocol/topologies.json).
- Test seeds: **9101, 9201, 9301**, each with 10,000 operating points paired across all 60 topologies. A further 10,000 PV-growth cases per test seed use the original topology. Total: **1,830,000 generated topology-scenario cases**. Reusing labels across models or verifier variants does not increase this number.
- Nine verifier variants: correct current network, stale original network, coherent resistance/reactance scaling by 0.8, 0.9, 0.95, 1.05, 1.1 or 1.2, and one fixed topology-specific realization of independent branch resistance/reactance perturbations within +/-10%. The physical reference stays unchanged.
- Primary thresholds target 5%, 10% and 20% escalation on the independent nominal calibration split. Test-quantile budgets of 5%, 10%, 20% and 30% are separately labeled retrospective diagnostics. Split-conformal margin intervals use alpha 0.001, 0.01 and 0.05; their exchangeability assumptions do not cover topology shift.

The full comparison contains **549 CSV files and 301,401 rows**. It includes learned routing scores, PhyRoute-D, physics-only fallback, signed approval screening, conservative joint approval, the residual MLP, conformal intervals, always-exact evaluation and a known-topology-change fallback. The [protocol](protocol/PROTOCOL.md) and [amendments](protocol/AMENDMENTS.md) record choices made before dependent outcome comparisons. This was a revision after an exploratory audit, not a preregistration.

## Reference resolution and checks

Strict backward/forward sweep is the first reference solver. Failed cases receive a Newton-Raphson attempt with numerical residual checks. Of the generated cases, **1,827,685 resolve and 2,315 remain unresolved**. Newton-Raphson recovers 135 cases after the sweep fails.

All unresolved cases occur on `close34_open01`: 802, 734 and 779 across the three test seeds. Their positions and solver diagnostics are retained. Every method uses the same validity mask within a condition; unresolved cases are excluded from reported error denominators, and unverified cheap approvals are counted separately. An unresolved solve is not proof of physical infeasibility.

**54 tests passed.** The independent audit checks output inventories, hashes, thresholds, denominators and summary calculations, and separately replays 2,745 selected comparison rows. See [revised_review.md](validation/revised_review.md). Some neural fits reached their epoch cap; all stopping histories and warnings are preserved.

## Reproduce serially

The recorded environment used Python 3.11.14. Package versions are pinned in [requirements-lock.txt](requirements-lock.txt); runtime and source hashes are in [run_manifest.json](results/revised/run_manifest.json).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
python scripts/run_pipeline.py
python scripts/generate_paper_assets.py
python validation/audit_revised.py --require-complete
```

`run_pipeline.py` trains or validates cached checkpoints, prepares all reference/proxy data, evaluates every seed pair, aggregates the results and measures complete-route latency. Run it serially within one output tree so shared cache writes do not overlap. The runner sets one BLAS thread. Latency measurements cover the original feeder and the previously explored RC-A exchange, not every topology or unresolved workload.

To resume individual phases, run the following in order:

```bash
for seed in 4101 4201 4301; do
  python scripts/run_revised.py train --seed "$seed"
done
for seed in 9101 9201 9301; do
  python scripts/run_revised.py prepare --test-seed "$seed"
done
for train in 4101 4201 4301; do
  for test in 9101 9201 9301; do
    python scripts/run_revised.py evaluate --train-seed "$train" --test-seed "$test"
  done
done
python scripts/analyze_revised.py
python scripts/measure_revised_latency.py
```

Caches are verified before reuse. A cached replay does not retrain its models. For fresh training, use a clean work copy without the generated `results/` tree and keep the archived results separately. Changing frozen source or configuration requires a separate output tree and a documented new run.

## Files and manuscript

| Path | Contents |
|---|---|
| `phyroute/revised.py`, `scripts/run_revised.py` | Revised methods, calibration and evaluation |
| `phyroute/reference.py`, `phyroute/powerflow.py` | Reference resolution, failures and validity masks |
| `phyroute/checkpoint.py`, `phyroute/topology_model.py` | Non-pickle model storage and development-topology residual model |
| `configs/revised.yaml`, `protocol/` | Frozen design, topology assignment and amendments |
| `results/revised/models/`, `results/models/` | Base and residual checkpoints, histories and calibration metadata |
| `results/revised/evaluation/`, `results/revised/analysis/` | Per-condition comparisons and derived summaries |
| `results/revised/data/`, `results/revised/physics/`, `results/revised/predictions/` | Regenerable numeric caches |
| `validation/`, `logs/` | Tests, independent audits and execution records |
| `paper/`, `figures/revised/` | SVProc LaTeX source, bibliography, generated tables and figures |

The manuscript uses the **official Springer SVProc LaTeX class** supplied for the conference. The final author-review PDF path is `output/pdf/PhyRoute_Revised_ICDLAI2026.pdf`. Typesetting is a separate step after `generate_paper_assets.py`, using `paper/build_pdf.py`; instructions are in [paper/RENDERING.md](paper/RENDERING.md). The PDF and LaTeX source are the manuscript deliverables for this revision.

**Legacy files:** `scripts/run_all.py` and `configs/default.yaml` belong to the original v0.1.0 experiment. `paper/build_content.py` and `paper/make_docx.js` implement its earlier Word workflow. They are retained in the full local workspace for provenance and omitted from the compact release archive. They must not be used to regenerate the revised results or manuscript. The revised commands above and the current [paper instructions](paper/README.md) are authoritative.

## Release and license

The [GitHub v0.2.0 release](https://github.com/Abhinav0905/phyroute/releases/tag/v0.2.0) is published and archived on Zenodo at [doi:10.5281/zenodo.22905467](https://doi.org/10.5281/zenodo.22905467). [CITATION.cff](CITATION.cff) provides citation metadata. See [PUBLICATION_RECEIPT.md](PUBLICATION_RECEIPT.md) for the verified publication records.

The tag and Zenodo archive preserve the release snapshot at commit `0411bf66cfc53cbabf5e31b1d2b9d0d4ea944325`. The `main` branch adds the DOI to the manuscript availability text and release documentation after publication; those edits are separate from the tagged snapshot. [RELEASE_NOTES.md](RELEASE_NOTES.md) describes the compact package, `release_manifest.json` and separately regenerable caches. `scripts/package_review.py` creates and verifies the archive after the PDF review and experiment audit pass. Conference submission remains a separate step after manuscript review.

Code is licensed under Apache-2.0; see [LICENSE](LICENSE). Publisher template files retain their own notices. AI assistance contributed to ideation, implementation, auditing and drafting; the manuscript records that assistance for human review.

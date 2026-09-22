# v0.2.0 software release

Title: *PhyRoute: Voltage Screening under Feeder Reconfiguration*. Author: Kumar Abhinav. Release date: 22 September 2026. Repository: [Abhinav0905/phyroute](https://github.com/Abhinav0905/phyroute).

Published records: [GitHub v0.2.0](https://github.com/Abhinav0905/phyroute/releases/tag/v0.2.0) and [Zenodo DOI 10.5281/zenodo.22905467](https://doi.org/10.5281/zenodo.22905467). The tag and archival record preserve source commit `0411bf66cfc53cbabf5e31b1d2b9d0d4ea944325`. See [PUBLICATION_RECEIPT.md](PUBLICATION_RECEIPT.md) for verification details.

The subsequent `main` update adds the minted DOI to the manuscript availability paragraph and release documentation. It is separate from the archived v0.2.0 snapshot.

The manifest on `main` describes the updated tree. Use `python scripts/package_review.py --manifest-only` after reviewed documentation changes to refresh it without replacing the published release ZIP. The immutable tag retains its original manifest.

This software release includes a companion manuscript for author review. Author affiliation and email still require confirmation. The manuscript has not been submitted to the conference. A software release or archival deposit does not establish peer review or acceptance.

## What changed

The revised experiment replaces the original two-switch comparison with all 59 radial single branch exchanges, a declared 15/44 development/held-out allocation and three fresh training/calibration runs crossed with three independent test draws. It adds the missing physics-only cascade, signed screening, an independently calibrated conformal baseline and a physics-augmented residual MLP. Nine verifier variants test current, stale and perturbed network models while keeping reference labels fixed.

The study now separates probe fitting from threshold calibration. The ensemble decision uses the margin of mean voltage extrema. Model arrays are stored without executable pickle, stopping histories and warnings are preserved, and cache identities include configuration and source hashes.

The reference pipeline retains every generated case and distinguishes a resolved voltage violation from numerical failure. Of 1,830,000 generated cases, 1,827,685 resolve and 2,315 remain unresolved; Newton-Raphson recovers 135 cases after sweep failure. Primary error rates are conditional on resolved cases. Escalation rates include every generated case.

The completed run contains 549 comparison files and 301,401 rows. The test suite passes 54 tests. Independent audits verify hashes, inventories, thresholds, denominators and summaries, with 2,745 selected comparison rows reconstructed from model arrays. The README reports the resulting safety, rejection and compute tradeoffs rather than retaining the original blanket claim that PhyRoute-D is the only effective recovery rule.

## Release archive workflow

Run `python scripts/package_review.py` after the final PDF has been rendered and checked. The script requires a passing visual-review record for the current PDF hash and a passing complete experiment audit. It builds `output/phyroute-v0.2.0.zip`. Uploading that archive and creating a GitHub or Zenodo release are separate operations.

1. **Compact code and paper archive.** Include package source, revised scripts, tests, configuration, protocol and amendments, topology inventory, pinned requirements, license notices, citation metadata, this README, the final author-review PDF and its LaTeX source/bibliography/required figures. Include all base and residual model arrays, their metadata, histories and calibration thresholds; all 549 evaluation CSVs and companion manifests; aggregate summaries; latency call records; audit scripts/reports; execution logs; and the run/environment manifests.
2. **Regenerable raw caches.** Numeric `.npz` files under `results/revised/data/`, `results/revised/physics/` and `results/revised/predictions/` remain in the full local workspace and are omitted from the compact archive. Their existing JSON manifests, solver resolution diagnostics and cache hashes are included. Regenerate the missing arrays with the serial pipeline before running the complete independent audit. The packaging script does not build a second raw-cache archive.
3. **Archive manifest and verification.** The script writes root-level `release_manifest.json`, with the release version, package status, source run hash, `included_files` and `omitted_regenerable_npz`. Each file entry records its relative path, byte count and SHA-256. The manifest is included in the ZIP but excluded from its own digest list. The script checks ZIP CRCs, member count and every listed member's SHA-256, then writes the archive digest to `output/phyroute-v0.2.0.zip.sha256`.

The completed experiment's run hash is `8cb0242b95738614d4bcb048900dc07da99e04dfb46149a77b77a0ff2bacd2fc`. Preserve this provenance; do not relabel older exploratory results as revised outcomes. Changes to source or configuration require a new documented run. A reproduction that reads packaged checkpoints is a cached-model replay; fresh-training reproduction starts in a separate clean work copy without generated results.

The script excludes virtual environments, Python bytecode, temporary files, smoke-test work trees and local operating-system metadata. It retains validation records and the inspected final PDF, but excludes the temporary `paper/build/` directory and unrelated output files. Required publisher LaTeX files and their original archive retain their existing notices.

## Legacy and operational records

`scripts/run_all.py`, `configs/default.yaml`, `paper/build_content.py` and `paper/make_docx.js` remain identifiable as v0.1.0 material in the full local workspace. The compact archive excludes them. They do not regenerate the revised manuscript or experiment. The source files fingerprinted by the completed revised run remain at their recorded paths.

One exploratory parallel invocation encountered a shared temporary-file collision while creating an identical PV cache. A serial retry passed the content-hash checks. Preserve that failure log and its successful retry as an operational record. The documented pipeline runs serially and prepares shared caches before the crossed evaluations.

The manuscript deliverables are the official SVProc LaTeX source and `output/pdf/PhyRoute_Revised_ICDLAI2026.pdf`. Citation metadata identifies the repository, version, release date and verified Zenodo DOI. Unconfirmed affiliation and email remain omitted. The manuscript has not been submitted through CMT; conference-submission status is independent of the published software release and archival record.

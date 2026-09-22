# Publication receipt

Verified on 22 September 2026.

| Record | Verified location or identifier |
| --- | --- |
| Repository | [Abhinav0905/phyroute](https://github.com/Abhinav0905/phyroute) |
| Software release | [v0.2.0](https://github.com/Abhinav0905/phyroute/releases/tag/v0.2.0) |
| Release source commit | `0411bf66cfc53cbabf5e31b1d2b9d0d4ea944325` |
| Version DOI | [10.5281/zenodo.22905467](https://doi.org/10.5281/zenodo.22905467) |
| Concept DOI | [10.5281/zenodo.22905466](https://doi.org/10.5281/zenodo.22905466) |
| Public archival record | [Zenodo record 22905467](https://zenodo.org/records/22905467) |

Zenodo's GitHub integration is enabled for this repository. Its release webhook is active, and Zenodo shows v0.2.0 as published. Future eligible GitHub releases can be archived through that integration. Existing authenticated sessions were used; the supplied access token was not used or stored.

## Archive verification

The downloaded Zenodo source archive passed ZIP CRC and recorded size/checksum checks. Every one of the 1,467 manifest-listed files matched its recorded size and SHA-256. Including the manifest, the archive contains 1,468 files. The GitHub release attachment was downloaded separately and matched the locally verified package checksum. The two compressed archives have different containers; their recorded file contents agree.

- GitHub attached `phyroute-v0.2.0.zip` SHA-256: `5e583e347d3a0d533654239a272788b1c8cb75e3b9390220ced71fc9a5e071b6`.
- Zenodo `Abhinav0905/phyroute-v0.2.0.zip` SHA-256: `19627797fd959e7119f370de1d682b21af20c6034c2a9e8ea9c1950978f225ab`.
- Frozen experiment run hash: `8cb0242b95738614d4bcb048900dc07da99e04dfb46149a77b77a0ff2bacd2fc`.

Machine-readable evidence is in [validation/published_archive_verification.json](validation/published_archive_verification.json). Large scenario, physics and prediction arrays are regenerable; their hashes and metadata remain in the archive. Saved model arrays and all evaluation tables are included.

## Manuscript citation update

The immutable v0.2.0 tag and Zenodo deposit contain the manuscript that existed before the DOI was assigned. A subsequent `main` commit adds the verified repository and version DOI to the availability paragraph, updates citation metadata and rebuilds the [13-page PDF](output/pdf/PhyRoute_Revised_ICDLAI2026.pdf). The scientific prose and numerical results are unchanged. All 13 rebuilt pages were visually checked; compilation produced no overfull boxes, missing characters or undefined references.

The updated PDF SHA-256 is `dd77642df915fda3e9c87d813101f39913251de05aa911b8ca5316fd47f96937`. The `release_manifest.json` on `main` describes that updated tree. The tag retains its original manifest. Run `python scripts/package_review.py --manifest-only` to refresh the current-tree manifest without replacing a published release ZIP.

The manuscript remains an author-review draft. Affiliation and correspondence details, human approval and the venue's submission checks remain in [AUTHOR_CHECKLIST.md](AUTHOR_CHECKLIST.md). No conference submission through CMT has been made. Software publication and a DOI do not establish peer review or acceptance.

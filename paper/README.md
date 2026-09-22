# PhyRoute manuscript

**PhyRoute: Voltage Screening under Feeder Reconfiguration** is the revised author-review manuscript. The current PDF has 13 A4 pages, four tables, two figures and 15 cited references. Editable LaTeX sources are included.

The delivery PDF is [`output/pdf/PhyRoute_Revised_ICDLAI2026.pdf`](../output/pdf/PhyRoute_Revised_ICDLAI2026.pdf). Author details and submission checks are tracked in [`AUTHOR_CHECKLIST.md`](../AUTHOR_CHECKLIST.md). The manuscript accompanies the published [GitHub v0.2.0 release](https://github.com/Abhinav0905/phyroute/releases/tag/v0.2.0) and [Zenodo archive](https://doi.org/10.5281/zenodo.22905467). It has not been submitted to the conference through CMT. Software release and archival deposit do not establish peer review or acceptance.

The v0.2.0 archive preserves the tagged source snapshot. The `main` manuscript adds the DOI to its data-and-code availability paragraph after publication. See [PUBLICATION_RECEIPT.md](../PUBLICATION_RECEIPT.md) for the source commit and publication records.

## Build

Run from the `phyroute-revised` repository root, with Python, Tectonic and Poppler available:

```bash
python scripts/generate_paper_assets.py
python paper/build_pdf.py paper/main_template.tex --render-pages
```

The first command regenerates the three numerical tables and `paper/generated_tables/numeric_ledger.json` from completed analysis files. The fourth table defines the routing methods in the manuscript source. The second command compiles the paper with its checked-in figures and bibliography. It writes `paper/build/main_template.pdf`, compiler logs, extracted text, diagnostics and page images. Review every rendered page after changing content, then copy the approved PDF to the delivery path.

| File | Purpose |
|---|---|
| `main_template.tex` | Root document using the official Springer SVProc class |
| `metadata.tex` | Title, authors and affiliation/email details |
| `manuscript_body.tex` | Abstract, results, discussion, conclusion and declarations |
| `introduction_related.tex`, `methods_draft.tex` | Included introduction and methods sections |
| `references.bib`, `references_verified.md` | Citation metadata and verification boundaries |
| `generated_tables/` | Numerical tables and their source-value ledger |
| `../figures/revised/` | The two publication figures in PDF and PNG form |
| `RENDERING.md` | Template provenance, build outputs and layout decisions |

The revised manuscript is supplied as PDF and editable LaTeX. No revised DOCX is included. The full local workspace preserves the original experiment and its legacy `build_content.py` and `make_docx.js` Word builders. The compact release archive omits those builders and the separate original experiment. They are not the revised paper's build entry points.

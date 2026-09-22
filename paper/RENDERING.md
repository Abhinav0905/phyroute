# Manuscript rendering

The revised manuscript uses the Springer **SVProc** package supplied on the [ICDLAI downloads page](https://icdlai.in/downloads/), retrieved on 22 September 2026. The compact review archive includes `vendor/icdlai2026/latex-package.zip` and its extracted LaTeX package, with the class, bibliography styles, sample and instructions. The full local workspace also retains the separate Word and publisher-PDF downloads.

The current author-review manuscript has **13 A4 pages, four tables, two figures and 15 cited references**. Its editable sources are listed in [`README.md`](README.md). The delivery PDF is [`../output/pdf/PhyRoute_Revised_ICDLAI2026.pdf`](../output/pdf/PhyRoute_Revised_ICDLAI2026.pdf). All 13 pages passed visual inspection; records are in [`pdf_visual_review.json`](../validation/pdf_visual_review.json) and [`paper_build.json`](../validation/paper_build.json). No revised Word document is included.

## Rebuild and inspect

From the repository root:

```bash
python scripts/generate_paper_assets.py
python paper/build_pdf.py paper/main_template.tex --render-pages
```

The asset command reads `results/revised/analysis/summary.json` and `latency.json`; it regenerates three numerical tables and a source-value ledger. The method-definition table is part of the LaTeX source. The build command uses Tectonic and the archived SVProc files. It may download missing TeX packages on its first run. Python runs the build helper, while Poppler supplies PDF metadata, text extraction and page rendering.

Outputs are written to `paper/build/`:

- `main_template.pdf`: compiled manuscript.
- `main_template.log` and compiler output files: typesetting diagnostics.
- `build_diagnostics.json`: page count, page size, overflow and missing-reference checks.
- `main_template.txt` and `pages/`: extracted text and rendered pages for inspection.

After every manuscript change, check every page for clipped text, figure readability, broken tables, missing glyphs and misplaced floats. Resolve overfull boxes and undefined citations before replacing the delivery PDF. Compilation alone does not establish visual quality or scientific approval.

## Layout decisions

`main_template.tex` selects `a4paper,norunningheads` and suppresses page numbers. It preserves SVProc's 12.2 cm by 19.3 cm text block and publisher typography. Figures fit that text width, equations remain native LaTeX and bibliography entries use `spmpsci`. Article identifiers use `eid` to preserve their digits without inserted thousands separators.

The conference's supplied template and some website formatting instructions disagree. The manuscript follows the supplied publisher package. Items requiring an author decision or venue clarification are listed in [`AUTHOR_CHECKLIST.md`](../AUTHOR_CHECKLIST.md).

## Full-workspace download checksums

This ledger covers the original downloads in the full workspace. Only `latex-package.zip` and its extracted package are included in the compact review archive; the Word archive and separate PDFs below are omitted there.

| File | SHA-256 |
|---|---|
| `latex-package.zip` | `51a9a27a80d02ac45218abc83f465557ae96cc315adaebcaffe60de3fcdc7e9b` |
| `word-templates.zip` | `3a9f0434fd5071edf0db38582a013857414a8978aaa957b7d4f25fa364db5219` |
| `sample-latex-paper-1.pdf` | `5a5f5456ec1d331009af92a1c9bd4b843d8d380ff0d2148b3e2feff848588d6c` |
| `latex-instructions-for-authors-1.pdf` | `acf79dc348988e901ad49bafa423b81efb62d4363026c3f0e7b295f20d8e05b5` |
| `proceedings-guidelines-for-authors-Springer-new.pdf` | `080ef3db94833c39e2ad54f27cfa471e897c8d767adc4c77e868a12d857c3ef4` |

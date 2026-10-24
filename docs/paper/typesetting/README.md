# Rebuilding the bilingual papers

The English and Japanese Markdown manuscripts are the editable content sources.
The typesetting template and scripts convert them to editable LaTeX and PDFs.
No network access or repository changes are required to rebuild the documents.

## Files

- `../manuscript_en.md` and `../manuscript_ja.md`: manuscript sources.
- `paper-template.tex`: shared page layout and font settings.
- `format.lua`: typesetting-only page-break and long-identifier handling.
- `build_papers.py`: reproducible Markdown → LaTeX → PDF build.
- `SLA_paper_EN.tex` and `SLA_paper_JA.tex`: generated, editable LaTeX sources.
- `check_pdfs.py`: structural checks and PNG rendering for visual review.

## Dependencies

- Python 3, Pandoc 3.1 or later, and XeLaTeX (TeX Live 2025 was used).
- LaTeX packages: fontspec, amsmath, amssymb, unicode-math, geometry, graphicx,
  xcolor, booktabs, longtable, array, calc, fancyhdr, titlesec, enumitem,
  needspace, fvextra, xurl, and hyperref.
- Fonts: Latin Modern Roman, Latin Modern Sans, Latin Modern Mono,
  Latin Modern Math, Noto Serif CJK JP, and Noto Sans CJK JP.
- For QA only: Poppler (`pdftoppm`, `pdffonts`), Python `pypdf`, and `pdfplumber`.

## Build

From the directory containing the manuscripts:

```sh
python3 typesetting/build_papers.py
python3 typesetting/check_pdfs.py
```

To build just one language, pass `--lang EN` or `--lang JA` to `build_papers.py`.
The build script creates `SLA_paper_EN.pdf` and `SLA_paper_JA.pdf` beside the
Markdown sources, and generated `.tex` sources in `typesetting/`.
Intermediates, build logs, page images, and a structural QA report are written
to `tmp/`. The script can initialize a local XeLaTeX format cache if the provided
Debian runtime does not have one; it does not install software or edit system
configuration.

Inspect every rendered page after changing content or layout. Structural checks
alone cannot establish mathematical accuracy, citation validity, or good layout.
Both papers are explicitly marked as drafts, without a conference identity or
claim of acceptance or peer review.

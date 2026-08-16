# Journal version (JBI)

This directory holds the **journal extension** of the paper, targeting the
*Journal of Biomedical Informatics*. All journal-extension writing goes here.

## `paper/` is FROZEN — do not edit it

`paper/main.tex` is the **conference paper**. It may be submitted as-is to a
conference or workshop, so it must stay exactly as it is for the remainder of
the project. Do not edit it, reformat it, or "fix" its numbers — including the
withdrawn EHR-contradiction column, which is discussed in
`sections/negation_contradiction.tex` and corrected *here*, not there.

`paper_journal/main.tex` began as a byte-identical copy of `paper/main.tex`
(verified at creation, 2026-08-17). Every journal change is made to this copy.

## Layout

```
paper_journal/
├── main.tex      # journal manuscript (started as a copy of the frozen paper)
├── plots/        # copied from paper/plots/ so this compiles standalone
├── sections/     # drafted sections, not yet wired into main.tex
└── README.md
```

## Sections drafted but NOT yet inserted

| File | Status |
|---|---|
| `sections/negation_contradiction.tex` | complete draft; replaces the conference paper's EHR-contradiction treatment. Its header comment lists four changes it forces elsewhere in `main.tex` (results-table column, abstract, limitations, H1 criterion 3). |

Sections live separately until reviewed, then get `\input{}` into `main.tex` or
pasted in. Keeping them out until then means a large in-place edit never
happens before the prose has been read.

## Still to write

- SHAP explainability section (results exist: `experiments/results/final_eval/shap_*`)
- Full journal rewrite (Step 6)

## Compiling

Neither version has ever been compiled — there is no LaTeX toolchain on this
machine (`pdflatex`/`xelatex`/`lualatex`/`latexmk`/`tectonic` all absent).
Table widths are estimated, not TeX-verified. Use Overleaf: upload `main.tex`
plus `plots/`, compile with pdfLaTeX, then check the log for `Overfull \hbox`.

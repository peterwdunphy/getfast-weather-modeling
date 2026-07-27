# GetFast Weather-Adjusted Predictor: Modeling and Results

This is the results companion to `getfast-weather-research` (the literature review).
It keeps the review intact (motivation, sports-science synthesis, strategic decisions)
and rewrites the technical sections from *proposals* into *results*: the WBGT estimator,
the estimated heat curve, distance scaling, per-runner tolerance and acclimation, the
deep-learning track, validation, and the shipped predictor.

## Build

```bash
pdflatex getfast-weather-modeling.tex
bibtex   getfast-weather-modeling
pdflatex getfast-weather-modeling.tex
pdflatex getfast-weather-modeling.tex
# or, in one step:
latexmk -pdf getfast-weather-modeling.tex
```

Requires a LaTeX distribution with `tikz`, `natbib`, `booktabs`, `hyperref`, `microtype`
(a standard TeX Live install). No LaTeX toolchain is installed on `bb`; compile on a
machine that has one (the same setup that builds the research repo).

## Contents

- `getfast-weather-modeling.tex` -- the document (lit review + Methods + Results).
- `references.bib` -- shared bibliography (copied from the research repo).
- `figures/` -- reused review figures plus model-result figures.

## Figures

- The seven result figures (`fig_heat_response`, `fig_g_vs_literature`,
  `fig_distance_scaling`, `fig_validation`, `fig_dl_response`, `fig_acclimation`,
  `fig_tolerance`) are built fresh by `build_paper_figures.py` in one consistent,
  colorblind-safe style. It reads the result artifacts under `/weather/results/`,
  `/weather/data/` (the OOF prediction parquets and the DL percentile npz), and the
  clean activity table. Rebuild with `python build_paper_figures.py`.
- Three review figures (`fig_marathon_starts_trend`, `fig_functional_forms_compared`,
  `fig_acclimation_timeline`) are carried over from `getfast-weather-research`.
- Three process diagrams are TikZ, drawn inline in the `.tex`.

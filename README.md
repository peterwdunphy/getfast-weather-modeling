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

- The four result figures (`fig_mape_ladder`, `fig_error_by_temp`, `fig_dl_vs_coach`,
  `fig_gamma`) are built by `build_paper_figures.py` in one consistent, colorblind-safe
  style. They report the held-out analysis in notebooks 09/10 (population and personalized
  deep-learning heat features added to a weather-blind XGBoost). Values are the authoritative
  numbers from notebook 10; the DL curve reads `/weather/data/heat_percentile_averaged.npz`.
  Rebuild with `python build_paper_figures.py`.
- Three review figures (`fig_marathon_starts_trend`, `fig_functional_forms_compared`,
  `fig_acclimation_timeline`) are carried over from `getfast-weather-research`.
- Two process diagrams (the experiment pipeline and the deep-learning architecture) are
  TikZ, drawn inline in the `.tex` and wrapped in `\resizebox` so they fit the page width.

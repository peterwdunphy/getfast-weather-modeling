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

## Figure provenance

Model-result figures are exported by scripts under
`/weather/getfast-weather/deep_learning_model/` (notebooks 01-09, `build_*` scripts,
`results/`). Review figures (`fig_marathon_starts_trend`, `fig_functional_forms_compared`,
`fig_acclimation_timeline`) are carried over from `getfast-weather-research`.

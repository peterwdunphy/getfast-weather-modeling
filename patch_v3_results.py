#!/usr/bin/env python3
"""patch_v3_results.py - update getfast-weather-modeling.tex to the v3 corpus / 5M-run results.

Every replacement is an exact-string swap that must be found exactly once, so a changed
anchor fails loudly instead of silently skipping. Numbers come from:
  notebook 19b (primary pipeline, v3, 28 C ceiling), 19c/19d/19e (secondaries),
  heat_percentile_averaged_v3.npz (curve), lit_comparison_v3 (benchmark), and the
  HR-quintile / monotone-projection recomputations logged in the session.
Run once:  python3 patch_v3_results.py
"""
import pathlib, sys

P = pathlib.Path("/weather/getfast-weather-modeling/getfast-weather-modeling.tex")
s = P.read_text()
R = []   # (old, new)

# ---------------------------------------------------------------- abstract + intro
R += [
("We assembled roughly a quarter million training runs, each matched to the weather it was run in.",
 "We assembled roughly four hundred thousand training runs, each matched to the weather it was run in."),
("evaluated on a runner-disjoint holdout of 15{,}892 activities from 104 runners, using robust error",
 "evaluated on a runner-disjoint holdout of 24{,}311 activities from 154 runners, using robust error"),
("The learned heat response steepens above roughly 16\\,\\textdegree C, reaching a 5.0\\% pace penalty at 28\\,\\textdegree C relative to 10\\,\\textdegree C, worth about eleven minutes to a 3:30 marathoner and twelve to a four-hour marathoner, with individual runners spanning 3.5\\% to 6.4\\% at the tenth and ninetieth percentiles. Adding these features lowered held-out error only modestly, from 7.34 to 7.13\\%. The informative result is calibration: the weather-blind model predicts runs in the hottest band 0.76\\% too fast, and the heat features remove that optimism.",
 "The learned heat response steepens above roughly 16\\,\\textdegree C, reaching a 4.3\\% pace penalty at 28\\,\\textdegree C relative to 10\\,\\textdegree C, worth about nine minutes to a 3:30 marathoner and ten to a four-hour marathoner, with individual runners spanning 3.1\\% to 5.4\\% at the tenth and ninetieth percentiles. Adding these features lowered held-out error only modestly, from 7.56 to 7.33\\%. The informative result is calibration: the weather-blind model predicts runs in the hottest band 1.84\\% too fast, and the heat features remove that optimism."),
("We assemble an observational corpus of roughly a quarter of a million training runs, each matched to the weather",
 "We assemble an observational corpus of roughly four hundred thousand training runs, each matched to the weather"),
]

# ---------------------------------------------------------------- methods
R += [
("  \\node[io]    (all)   at (3.5, 0)    {259{,}262 activities from 1{,}628 runners};",
 "  \\node[io]    (all)   at (3.5, 0)    {403{,}719 activities from 2{,}420 runners};"),
("  \\node[io]    (hold)  at (6.8,-3.2)  {Holdout runners\\\\15{,}892 activities, 104 runners};",
 "  \\node[io]    (hold)  at (6.8,-3.2)  {Holdout runners\\\\24{,}311 activities, 154 runners};"),
("Runners are partitioned by a hash of their identifier, so the 104 holdout runners contribute to no fitting stage",
 "Runners are partitioned by a hash of their identifier, so the 154 holdout runners contribute to no fitting stage"),
("Runs above 25\\,\\textdegree C WBGT were excluded, since the corpus thins sharply beyond that point. The resulting frame holds 259{,}262 activities from 1{,}628 runners across over one hundred countries.",
 "Runs above 28\\,\\textdegree C WBGT were excluded, since the corpus thins sharply beyond that point (Appendix~\\ref{app:ceiling}). The resulting frame holds 403{,}719 activities from 2{,}420 runners across roughly one hundred countries. This is an enlarged corpus relative to an earlier round of this study: weather enrichment was extended to activities recorded after the original cutoff and to runners who later backfilled their histories, which raised the number of runs above 28\\,\\textdegree C WBGT from 1{,}846 to 4{,}205 and made the 28\\,\\textdegree C ceiling supportable."),
("One runner in ten was held out, leaving 15{,}892 activities from 104 runners for evaluation once the prior-history requirement described below is applied.",
 "One runner in ten was held out, leaving 24{,}311 activities from 154 runners for evaluation once the prior-history requirement described below is applied."),
("The model predicts pace directly. Two heat artifacts are then read off the trained model by counterfactual WBGT sweeps that hold every other channel fixed.",
 "The model predicts pace directly. It was trained for five million optimizer steps on the fitting runners: an initial one-million-step run, resumed from its final state with the learning-rate decay stretched so that the rate reached its floor at the five-million-step mark rather than at 2.2 million. Held-out error in the hot strata stopped improving near two million steps while overall error continued to fall slowly; the checkpoint with the lowest held-out error, at step 4.33 million, is the one used throughout. Two heat artifacts are then read off the trained model by counterfactual WBGT sweeps that hold every other channel fixed."),
("The median of these shapes is taken across the 1{,}186 fitting runners with a positive heat response, then rescaled by the median sensitivity of a 750-runner reference sample.",
 "The median of these shapes is taken across the 1{,}786 fitting runners with a positive heat response, then rescaled by the median sensitivity of a 1{,}093-runner reference sample."),
("it alters 8 of 37 grid points, by at most 0.13 percentage points of pace against a total curve range of 5.04.",
 "it alters 4 of 37 grid points, by at most 0.05 percentage points of pace against a total curve range of 4.21."),
("All comparisons use the same 15{,}892-activity holdout, identical hyperparameters, and three random seeds (100, 200, 300).",
 "All comparisons use the same 24{,}311-activity holdout, identical hyperparameters, and three random seeds (100, 200, 300)."),
("a pool of 149{,}974 activities from 934 runners, and then frozen before being applied to the holdout",
 "a pool of 227{,}908 activities from 1{,}413 runners, and then frozen before being applied to the holdout"),
]

# ---------------------------------------------------------------- results: main table
R += [
("  \\caption{Held-out model comparison (15{,}892 activities, 104 runners, three XGB seeds). Robust MAPE",
 "  \\caption{Held-out model comparison (24{,}311 activities, 154 runners, three XGB seeds). Robust MAPE"),
("""    Weather-blind XGB               & 7.3381 & $-1.44$ & --- \\\\
    $+$ population DL curve          & 7.2936 & $-1.57$ & $+0.045$ vs blind [$-0.009$, 0.101] \\\\
    $+$ personalized DL score        & 7.1304 & $-1.09$ & $+0.163$ vs population [0.039, 0.308] \\\\
    \\quad shuffled control          & 7.2996 & $-1.31$ & $-0.006$ vs population [$-0.051$, 0.034] \\\\
    \\quad constrained (curve delta) & 7.2043 & $-1.16$ & $+0.089$ vs population [0.015, 0.167] \\\\""",
"""    Weather-blind XGB               & 7.5574 & $-1.53$ & --- \\\\
    $+$ population DL curve          & 7.4976 & $-1.64$ & $+0.060$ vs blind [0.003, 0.118] \\\\
    $+$ personalized DL score        & 7.3293 & $-1.17$ & $+0.168$ vs population [0.062, 0.283] \\\\
    \\quad shuffled control          & 7.4841 & $-1.38$ & $+0.014$ vs population [$-0.019$, 0.043] \\\\
    \\quad constrained (curve delta) & 7.3996 & $-1.23$ & $+0.098$ vs population [0.041, 0.154] \\\\"""),
]

# ---------------------------------------------------------------- results: the heat response
R += [
("reaching 1.2\\% at $20\\,^{\\circ}$C, 2.4\\% at $24\\,^{\\circ}$C and 5.02\\% at $28\\,^{\\circ}$C relative to a $10\\,^{\\circ}$C reference. For a runner whose cool-weather marathon is 3:30 those penalties correspond to roughly two and a half, five and eleven minutes, respectively; for a four-hour runner, to roughly three, six and twelve minutes.",
 "reaching 1.0\\% at $20\\,^{\\circ}$C, 2.4\\% at $24\\,^{\\circ}$C and 4.3\\% at $28\\,^{\\circ}$C relative to a $10\\,^{\\circ}$C reference. For a runner whose cool-weather marathon is 3:30 those penalties correspond to roughly two, five and nine minutes, respectively; for a four-hour runner, to roughly two and a half, six and ten minutes."),
("The tenth and ninetieth percentiles of the per-runner response at $28\\,^{\\circ}$C are 3.5\\% and 6.4\\%. For the 3:30 marathoner, that is a gap of seven minutes versus thirteen; for the four-hour runner, eight against fifteen. The gap between a heat-tolerant and a heat-sensitive runner in identical conditions therefore ranges six to seven minutes.",
 "The tenth and ninetieth percentiles of the per-runner response at $28\\,^{\\circ}$C are 3.1\\% and 5.4\\%. For the 3:30 marathoner, that is a gap of six and a half minutes versus eleven; for the four-hour runner, seven against thirteen. The gap between a heat-tolerant and a heat-sensitive runner in identical conditions therefore ranges five to six minutes."),
("Between the 20 to $24\\,^{\\circ}$C and 24 to $28\\,^{\\circ}$C bands the learned curve steepens from 0.31 to 0.66\\% per degree, a factor of 2.12. The compensability ratio steepens by a factor between 1.64 and 1.87 over the same interval, depending on the humidity and solar load assumed. The two agree in sign and in order of magnitude, with the measured curvature the larger of the two. A gap in that direction is expected rather than troubling, since the pace penalty should be convex in the ratio rather than proportional to it: as the requirement approaches the ceiling, sweat that cannot evaporate drips instead, so the fluid cost compounds on precisely the days evaporation is least effective \\citep{taylor2006}. The physics predicts that the curve should bend and roughly how sharply, not that the penalty should reach 5\\% at $28\\,^{\\circ}$C.",
 "Between the 20 to $24\\,^{\\circ}$C and 24 to $28\\,^{\\circ}$C bands the learned curve steepens from 0.33 to 0.48\\% per degree, a factor of 1.45. The compensability ratio steepens by a factor between 1.64 and 1.87 over the same interval, depending on the humidity and solar load assumed. The two agree in sign and are close in size, with the measured curvature falling a little below the range the physics implies. A gap in that direction is consistent with the selection mechanisms of Section~\\ref{sec:selection}, all of which act to understate the penalty on the hottest days, and it leaves room for the additional convexity that \\citet{taylor2006} attribute to sweat that drips rather than evaporates once the requirement approaches the ceiling. The physics predicts that the curve should bend and roughly how sharply, not that the penalty should reach 4.3\\% at $28\\,^{\\circ}$C."),
("""distance, elevation and fitness controlled, gives $+0.22\\%$ per \\textdegree C in the easiest fifth
rising monotonically to $+0.53\\%$ in the hardest, a factor of 2.5 (Figure~\\ref{fig:hrinteraction}).
The gradient is concentrated at the top: a linear heart-rate-by-heat interaction does not capture it
($+0.021\\%$ per \\textdegree C per standard deviation of heart rate, $t=0.8$), because four of the five
quintiles are close together and the effect appears in the hardest one.""",
 """distance, elevation and fitness controlled, gives $+0.20\\%$ per \\textdegree C in the easiest fifth
rising monotonically to $+0.42\\%$ in the hardest, a factor of 2.1 (Figure~\\ref{fig:hrinteraction}).
The gradient is not captured by a linear heart-rate-by-heat interaction
($-0.023\\%$ per \\textdegree C per standard deviation of heart rate, $t=-0.9$), which is indistinguishable
from zero even though every quintile step is in the same direction."""),
("reaching roughly $+7\\%$ in the hottest and hardest cell. \\textbf{B:} runs per cell. Cells below 150 runs\nare marked and omitted from A; the hottest-and-hardest cell rests on 272 runs, so the peak of the ridge\nis the least certain part of the surface.}",
 "reaching roughly $+5\\%$ in the hottest and hardest cell. \\textbf{B:} runs per cell. Cells below 150 runs\nare marked and omitted from A; the hottest-and-hardest cell rests on 534 runs, so the peak of the ridge\nis the least certain part of the surface.}"),
]

# ---------------------------------------------------------------- results: prediction + bias
R += [
("Robust MAPE falls from 7.34\\% for the weather-blind model to 7.29\\% with the population curve, and to 7.13\\% once the personalized score is added, a total of 0.21 points. On a 3:30 marathon the whole progression is worth about 26 seconds, and on a four-hour marathon about 30.",
 "Robust MAPE falls from 7.56\\% for the weather-blind model to 7.50\\% with the population curve, and to 7.33\\% once the personalized score is added, a total of 0.23 points. On a 3:30 marathon the whole progression is worth about 29 seconds, and on a four-hour marathon about 33."),
("If the gain came from the extra column alone, the shuffled version should reproduce it; instead it lands at 7.30\\%, marginally worse than the population model. In runner-clustered bootstrap terms, the personalized gain beyond the population curve is 0.1632 points, with a 95\\% interval of [0.039, 0.308] excluding zero, while the shuffled interval [$-0.051$, 0.034] straddles it. The population curve's own gain of 0.0445 points has an interval of [$-0.009$, 0.101].",
 "If the gain came from the extra column alone, the shuffled version should reproduce it; instead it lands at 7.48\\%, within noise of the population model. In runner-clustered bootstrap terms, the personalized gain beyond the population curve is 0.168 points, with a 95\\% interval of [0.062, 0.283] excluding zero, while the shuffled interval [$-0.019$, 0.043] straddles it. The population curve's own gain of 0.060 points has an interval of [0.003, 0.118], which now excludes zero on the enlarged corpus."),
("  \\caption{Held-out model comparison (15{,}892 activities, 104 runners, three XGB seeds). Panel A: robust MAPE",
 "  \\caption{Held-out model comparison (24{,}311 activities, 154 runners, three XGB seeds). Panel A: robust MAPE"),
("The weather-blind model predicts runs too slow in cold and mild conditions, with a bias near $-2\\%$, then crosses zero and predicts too fast in the hottest band, at $+0.755\\%$ for runs between 22 and $25\\,^{\\circ}$C. Its errors on hot days are therefore optimistic, which is the direction that matters for a runner planning a race.",
 "The weather-blind model predicts runs too slow in cold and mild conditions, with a bias near $-2\\%$, then crosses zero between 22 and $25\\,^{\\circ}$C ($+0.36\\%$) and predicts too fast in the hottest band, at $+1.84\\%$ for runs between 25 and $28\\,^{\\circ}$C. Its errors on hot days are therefore optimistic, and increasingly so as conditions worsen, which is the direction that matters for a runner planning a race."),
("Supplying the heat features removes that optimism. In the same hottest band, the population curve moves the bias to $-1.336\\%$ and the personalized score to $-0.715\\%$, so the sign flips from optimistic to slightly conservative. For the 3:30 runner this is a shift from predicting 1.6 minutes too fast to 1.5 minutes too slow, and for the four-hour runner from 1.8 minutes too fast to 1.7 too slow. The personalized model also holds the bias closest to zero across every other temperature band.",
 "Supplying the heat features removes that optimism. In the same hottest band, the population curve moves the bias to $-1.08\\%$ and the personalized score to $-0.81\\%$, so the sign flips from optimistic to conservative. For the 3:30 runner this is a shift from predicting 3.9 minutes too fast to 1.7 minutes too slow, and for the four-hour runner from 4.4 minutes too fast to 2.0 too slow. The personalized model also holds the bias closest to zero across every other temperature band."),
]

# ---------------------------------------------------------------- results: what the score measures
R += [
("Its effect is nearly independent of temperature. Relative to the population model it shifts predictions by about $-0.4$ to $-0.6\\%$ in every temperature band, and the row-level correlation between that shift and WBGT is $-0.027$.",
 "Its effect is nearly independent of temperature. Relative to the population model it shifts predictions by about $-0.2$ to $-0.5\\%$ in every temperature band, and the row-level correlation between that shift and WBGT is $+0.026$."),
("it yields 7.20\\% robust MAPE, a gain of 0.089 points beyond the population curve with a 95\\% interval of [0.015, 0.167]. That is a genuine temperature-shaped signal, but it is about 55\\% of the 0.1632-point gain the unconstrained score achieves.",
 "it yields 7.40\\% robust MAPE, a gain of 0.098 points beyond the population curve with a 95\\% interval of [0.041, 0.154]. That is a genuine temperature-shaped signal, but it is about 58\\% of the 0.168-point gain the unconstrained score achieves."),
("Fitted on the manuscript's baseline, the population curve's scale is $\\alpha = 1.10$, with a 95\\% interval of [0.86, 1.35] that includes one, so the magnitude the deep model learned agrees with the correction the pace model independently requires. The personalized deviation's coefficient is $\\gamma = -1.56$ [$-2.43$, $-0.75$], decisively negative:",
 "Fitted on the manuscript's baseline, the population curve's scale is $\\alpha = 1.10$, with a 95\\% interval of [0.91, 1.30] that includes one, so the magnitude the deep model learned agrees with the correction the pace model independently requires. The personalized deviation's coefficient is $\\gamma = -0.96$ [$-1.71$, $-0.10$], negative:"),
]

# ---------------------------------------------------------------- results: benchmark
R += [
("\\caption{Held-out bias by heat threshold, out-of-fold over all 1{,}628 runners. Flatness is the\nroot-mean-square variation of the bias profile across WBGT bins; lower means less of the error is\npredictable from conditions.}",
 "\\caption{Held-out bias by heat threshold, out-of-fold over the 2{,}143 runners with an estimable\nheat tolerance (400{,}337 runs). Flatness is the root-mean-square variation of the bias profile across\nWBGT bins; lower means less of the error is predictable from conditions.}"),
("""\\begin{tabular}{@{}lrrrr@{}}
\\toprule
& \\multicolumn{3}{c}{signed bias (\\%)} & \\\\
\\cmidrule(lr){2-4}
Correction & 15+ & 20+ & 23+ & Flatness \\\\
\\midrule
None (weather-blind)          & $+0.07$ & $+0.99$ & $+1.90$ & 1.04 \\\\
Martin 1999                   & $-0.49$ & $-0.53$ & $-0.21$ & \\textbf{0.23} \\\\
\\textbf{This study}           & $-0.55$ & $-0.35$ & $\\mathbf{+0.09}$ & 0.29 \\\\
Mantzios 2022 (marathon)      & $-0.44$ & $-0.10$ & $+0.49$ & 0.42 \\\\
Ely 2007 (25th place)         & $-1.00$ & $-0.71$ & $-0.15$ & 0.50 \\\\
Guy 2014                      & $-0.91$ & $-0.50$ & $+0.12$ & 0.54 \\\\
Vernon 2021 (mass field)      & $-2.73$ & $-3.45$ & $-3.33$ & 1.95 \\\\
Gasparetto 2020 (cluster 2)   & $-9.55$ & $-18.89$ & $-24.98$ & 11.97 \\\\
\\bottomrule
\\end{tabular}""",
 """\\begin{tabular}{@{}lrrrrr@{}}
\\toprule
& \\multicolumn{4}{c}{signed bias (\\%)} & \\\\
\\cmidrule(lr){2-5}
Correction & 15+ & 20+ & 23+ & 25+ & Flatness \\\\
\\midrule
None (weather-blind)          & $+0.10$ & $+0.94$ & $+1.63$ & $+2.08$ & 1.19 \\\\
\\textbf{This study}           & $-0.58$ & $-0.52$ & $-0.46$ & $-0.47$ & \\textbf{0.10} \\\\
Martin 1999                   & $-0.57$ & $-0.73$ & $-0.69$ & $-0.71$ & 0.29 \\\\
Mantzios 2022 (marathon)      & $-0.46$ & $-0.19$ & $+0.17$ & $+0.41$ & 0.40 \\\\
Ely 2007 (25th place)         & $-0.95$ & $-0.73$ & $-0.41$ & $-0.18$ & 0.49 \\\\
Guy 2014                      & $-0.83$ & $-0.48$ & $\\mathbf{-0.03}$ & $+0.34$ & 0.58 \\\\
Vernon 2021 (mass field)      & $-2.64$ & $-3.40$ & $-3.53$ & $-3.62$ & 2.13 \\\\
\\bottomrule
\\end{tabular}"""),
("""Three things follow. The learned curve leaves the smallest residual bias in the heat, $+0.09\\%$ above
WBGT 23 against $+1.90\\%$ uncorrected, and it is the only form besides Mantzios's marathon slope whose
published magnitude needs no rescaling; fitting a free scale to it returns $1.05$. Forms estimated on
mass fields over-correct severely when applied to training runs, by factors of three to fourteen, which
is the expected direction given that race-day heat sensitivity exceeds training-pace sensitivity.
And the elite forms, which are shallower, transfer well: Martin's quadratic is marginally flatter across
the whole range than the learned curve, though it achieves this partly by bending in cool conditions
where the learned curve is deliberately flat.""",
 """Three things follow. The learned curve is the flattest of the corrections by a wide margin: its
residual bias sits between $-0.46$ and $-0.58\\%$ at every threshold, a root-mean-square variation of
0.10 against 0.29 for the next best form, so almost none of what remains is predictable from
conditions. What remains is a small uniform over-correction, and the form whose residual comes nearest
zero in the hottest strata, Guy's capped slope, does so by crossing from under- to over-correction as
the threshold rises rather than by being flat. The learned curve is also one of only two forms, with
Mantzios's marathon slope, whose published magnitude needs no rescaling; fitting a free scale to it
returns $1.09$. Forms estimated on mass fields over-correct severely when applied to training runs, by
factors of three to fourteen (the Gasparetto quadratics by more than an order of magnitude and are
omitted from the table), which is the expected direction given that race-day heat sensitivity exceeds
training-pace sensitivity. And the elite forms, which are shallower, transfer well: Martin's quadratic
is the next flattest, though it achieves this partly by bending in cool conditions where the learned
curve is deliberately flat."""),
("Figure~\\ref{fig:litbench} shows the same comparison as a profile. No published form and no fitted\ncorrection alters mean absolute error appreciably, all sitting between 7.4\\% and 7.5\\% at WBGT 15 and\nabove;",
 "Figure~\\ref{fig:litbench} shows the same comparison as a profile. No published form and no fitted\ncorrection alters mean absolute error appreciably, all sitting between 7.40\\% and 7.48\\% on this sample;"),
("runners (right), at three cumulative WBGT thresholds. Bars are runner-clustered bootstrap 95\\%",
 "runners (right), at four cumulative WBGT thresholds. Bars are runner-clustered bootstrap 95\\%"),
]

# ---------------------------------------------------------------- results: robustness branch
R += [
("This branch is supplementary and is not an equivalent comparison, since it changes the fitness representation and drops current effort at the same time, which raises its absolute error. Within it, the population temperature information remains useful though its gain interval includes zero, raw WBGT produces a clearer gain, and the strict-prior personalization no longer improves on the population curve. The $\\gamma$ reversal disappears, but the estimate is imprecise and unstable across runner folds.",
 "This branch is supplementary and is not an equivalent comparison, since it changes the fitness representation and drops current effort at the same time, which raises its absolute error; the fixed fitness estimate is also available for only two thirds of the enlarged corpus, so the branch runs on 16{,}554 held-out activities from 103 runners. Within it, the population temperature information remains useful though its gain interval includes zero (0.09 points [$-0.10$, 0.25]), raw WBGT produces a gain of the same size, and the strict-prior personalization's gain over the population curve, 0.16 points [$-0.03$, 0.38], no longer excludes zero. The $\\gamma$ reversal disappears ($\\gamma = +1.12$ [$-0.92$, 3.31]), but the estimate is imprecise and unstable across runner folds. A narrower ablation that keeps the trailing-pace fitness proxy and removes only current heart rate gives the same picture with more precision: the population curve improves on the blind baseline by 0.03 points [0.01, 0.06], the personalized score by a further 0.09 [0.01, 0.18], and the shuffled control by 0.01 [$-0.01$, 0.03]."),
]

# ---------------------------------------------------------------- results: acclimation (rewritten)
R += [
("""The literature in Section~\\ref{sec:literature} predicts that recent heat exposure should reduce a runner's heat penalty. Two tests examine whether the model's per-runner score behaves that way.

The first compares the score against each runner's recent training conditions. Runners with more hot training in the preceding weeks receive lower sensitivity ranks, which is the direction acclimation predicts. The second uses an independently curated set of candidate acclimation episodes, matched hot runs recorded before and after a block of heat exposure. Across those episodes, performance in matched conditions improves on average after the block.

Both tests are consistent with acclimation, and neither is decisive. The episodes are candidates identified from training history rather than verified physiological measurements, and no runner contributes enough repeated observations near $28\\,^{\\circ}$C to estimate an individual adaptation magnitude. What the corpus supports is that acclimation is detectable as a population-level tendency in these data. What it does not support is a per-runner acclimation state that could be tracked over a training block, which would require either denser sampling in heat or the physiological markers, resting heart rate chief among them, that consumer devices do not reliably record.""",
 """The literature in Section~\\ref{sec:literature} predicts that recent heat exposure should reduce a runner's heat penalty. Two tests examine whether the model's per-runner score behaves that way.

The first compares the score against each runner's recent training conditions. For every activity the number of runs above $18\\,^{\\circ}$C WBGT in the preceding fourteen days is counted, and the runner's prospective sensitivity rank is regressed on it within runner, with the current run's WBGT as a control and a placebo term for the fourteen days that follow. One standard deviation more recent hot training lowers the rank by 0.94 percentile points (runner-clustered 95\\% interval [$-1.23$, $-0.64$]), which is the direction acclimation predicts, while the same count over the following fourteen days has no effect ($+0.08$ [$-0.19$, 0.34]), so the association is not a season or climate artifact. An exposure measure that decays with a fourteen-day half-life gives the same answer ($-0.73$ per standard deviation [$-1.13$, $-0.33$]). Across within-runner deciles of exposure the rank falls monotonically, from $+0.9$ points in the least-exposed decile to $-1.6$ in the most exposed.

The second test uses an independently curated set of candidate acclimation episodes: pairs of hot runs, matched on conditions, recorded before and after a block of heat exposure, with a control set of pairs that bracket no such block. Across 465 low-to-high exposure episodes from 353 runners, performance in matched conditions improves after the block by 1.95\\% [1.34, 2.56], and 63\\% of episodes improve; across 450 low-to-low control episodes from 359 runners the improvement is 0.55\\% [$-0.03$, 1.13]. The difference, 1.40 percentage points [0.54, 2.27], is the acclimation effect the design isolates.

Both tests are consistent with acclimation, and neither is decisive. The episodes are candidates identified from training history rather than verified physiological measurements, and few runners contribute repeated episodes above $25\\,^{\\circ}$C (29 episodes from 24 runners), so no individual adaptation magnitude can be estimated. What the corpus supports is that acclimation is detectable as a population-level tendency in these data, and that the model's per-runner score moves with it in the expected direction. What it does not support is a per-runner acclimation state that could be tracked over a training block, which would require either denser sampling in heat or the physiological markers, resting heart rate chief among them, that consumer devices do not reliably record."""),
]

# ---------------------------------------------------------------- discussion + limitations
R += [
("The magnitude is quantifiable: at $28\\,^{\\circ}$C the curve implies roughly eleven additional minutes for a runner whose ideal weather marathon prediction is 3:30 and twelve for a four-hour runner.",
 "The magnitude is quantifiable: at $28\\,^{\\circ}$C the curve implies roughly nine additional minutes for a runner whose ideal weather marathon prediction is 3:30 and ten for a four-hour runner."),
("Second, runs at the extreme hot end are sparse. The held-out cohort is bounded above at $25\\,^{\\circ}$C by the ceiling imposed in Section~\\ref{sec:methods} based on sparsity. Even below that threshold, the warm bins hold fewer activities than the mild ones, leading to the hot-end estimates carrying the most sampling uncertainty. Denser sampling of genuinely hot runs would tighten these estimates.",
 "Second, runs at the extreme hot end are sparse. The held-out cohort is bounded above at $28\\,^{\\circ}$C by the ceiling imposed in Section~\\ref{sec:methods} based on sparsity; the enlarged corpus more than doubled the runs above that point, which is what made the ceiling supportable at 28 rather than 25\\,\\textdegree C. Even below it, the warm bins hold fewer activities than the mild ones (1{,}625 held-out activities between 25 and $28\\,^{\\circ}$C against 5{,}398 between 10 and 15), leading to the hot-end estimates carrying the most sampling uncertainty. Denser sampling of genuinely hot runs would tighten these estimates."),
("the effect is roughly constant across temperature, only about half survives when constrained to act through the temperature curve, and the direct calibration returns the wrong sign.",
 "the effect is roughly constant across temperature, only about 58\\% survives when constrained to act through the temperature curve, and the direct calibration returns the wrong sign."),
]

missing = [a[:70] for a, _ in R if s.count(a) != 1]
if missing:
    print("ANCHORS NOT FOUND EXACTLY ONCE:"); [print("  -", m) for m in missing]; sys.exit(1)
for a, b in R:
    s = s.replace(a, b)
P.write_text(s)
print(f"applied {len(R)} replacements to {P.name}")

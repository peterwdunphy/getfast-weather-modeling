#!/usr/bin/env python3
"""mechanism_numbers.py - section 7 of journal/numbers.tex: heart rate, cadence and stride length in heat, and the
decomposition of the gap between the training curve and the race-day scale.

Reads five aggregate JSON files (no run-level data) and prints one \\newcommand per number with its source field.
Read-only; writes nothing (stdout only, kept as mechanism_numbers.log).

  estimates.json   pace_hr_cadence_heat_20261110/proposed_figures.py   training-run models behind fig_mech_*
  followups.json   pace_hr_cadence_heat_20261110/followups.py          gap decomposition and robustness (fig_followup_*)
  race.json        cadence_decay_20261029                              race-distance efforts split by what cadence did
  benchmarks.json      pace_hr_cadence_heat_20261110/benchmarks.py      the same quantities in the units of earlier work
  speed_adjusted.json  pace_hr_cadence_heat_20261110/speed_adjusted.py  cadence change net of the slowing it accompanies

Run: /home/bb/test_env/bin/python3 journal/mechanism_numbers.py > journal/mechanism_numbers.log
"""
import json

SRC = "/weather/data/fade/pace_hr_cadence_heat_20261110"
E = json.load(open(f"{SRC}/estimates.json"))
F = json.load(open(f"{SRC}/followups.json"))
R = json.load(open("/weather/data/fade/cadence_decay_20261029/race.json"))["d_section21_v3"]
BM = json.load(open(f"{SRC}/benchmarks.json"))
SA = json.load(open(f"{SRC}/speed_adjusted.json"))


# ---- formatting helpers (same conventions as the rest of numbers.tex) ------------------------------------------------
def num(v, nd=2):
    """A number for running text: LaTeX minus sign in math mode for negatives, fixed decimals."""
    s = f"{abs(v):.{nd}f}"
    return f"$-{s}$" if round(v, nd) < 0 else s


def ci(b, lo, hi, nd=2):
    """Estimate [lower, upper]."""
    return f"{num(b, nd)} [{num(lo, nd)}, {num(hi, nd)}]"


def bse(b, se, nd=2):
    """Estimate with a normal 95% interval from a coefficient and its standard error."""
    return ci(b, b - 1.96 * se, b + 1.96 * se, nd)


def big(n):
    """Integer with a thin-space-free thousands separator that LaTeX sets correctly: 363{,}005."""
    return f"{int(n):,}".replace(",", "{,}")


def cmd(name, value, why):
    """Print one macro line with its provenance as a trailing comment."""
    print(f"\\newcommand{{\\{name}}}{{{value}}}   % {why}")


def est(d, nd=2):
    """Format a stored {'coef', 'ci'} estimate."""
    return ci(d["coef"], d["ci"][0], d["ci"][1], nd)


# WBGT bin positions in the 16-bin training models (2 C bins, reference 8-10 C at index 5)
I20, I24, I28 = 11, 13, 15

# ---- 7a. training runs: pace, heart rate and cadence by WBGT (estimates.json, fig1) ---------------------------------
print("% 7a. Training runs: pace, heart rate and cadence against WBGT (estimates.json fig1; reference WBGT 8-10 C)")
f1 = E["fig1"]
cmd("mechRuns", big(E["n_runs"]), "estimates.json n_runs")
cmd("mechRunners", big(E["n_runners"]), "estimates.json n_runners")
cmd("mechRunsHot", big(E["bin_counts"]["[28.0, inf)"]), "bin_counts: runs at WBGT 28 C and over")
for name, key, why in (("mechPace", "pace", "pace, % slower"),
                       ("mechPaceAtHR", "pace_at_hr", "pace at matched heart rate, % slower"),
                       ("mechHR", "hr_at_pace", "heart rate at matched pace, bpm"),
                       ("mechCad", "cad_at_pace", "cadence at matched pace, %")):
    for tag, i, lab in (("Twenty", I20, "20-22 C"), ("TwentyFour", I24, "24-26 C"), ("Hot", I28, "28 C and over")):
        cmd(f"{name}{tag}", bse(f1[key]["b"][i], f1[key]["se"][i]), f"fig1.{key} at {lab}: {why}")

# ---- 7b. heat cost by band of the run's %HRmax (estimates.json fig2; followups.json C1) -----------------------------
print("% 7b. Pace cost by band of the run's mean heart rate (fig2), and two other band definitions (followups C1)")
for tag, key in (("Easy", "easy"), ("Mod", "moderate"), ("Hard", "hard")):
    b = E["fig2"][key]
    cmd(f"band{tag}Hot", bse(b["b"][I28], b["se"][I28]), f"fig2.{key} at 28 C and over, % slower")
    cmd(f"band{tag}TwentyFour", bse(b["b"][I24], b["se"][I24]), f"fig2.{key} at 24-26 C, % slower")
    cmd(f"band{tag}Runs", big(b["n"]), f"fig2.{key} runs")
C1 = F["C1_effort_bands"]
for tag, key, why in (("Adj", "eff_b", "heat-adjusted %HRmax bands"), ("Own", "eff_c", "own-HR thirds within 4 C WBGT band")):
    for lv, lab in ((0, "Easy"), (1, "Mod"), (2, "Hard")):
        cmd(f"band{tag}{lab}Hot", est(C1[key][lv]["at_ge28"]), f"C1.{key}[{lv}] at 28 C and over: {why}")
cmd("bandMovedPct", num(C1["moved_between_bands_by_adjustment_pct"], 1), "C1: % of runs that change band after the heat adjustment")

# ---- 7c. speed = cadence x stride (estimates.json fig4) ------------------------------------------------------------
print("% 7c. Speed difference split into cadence and stride length (fig4; coefficients add exactly)")
f4 = E["fig4"]
for tag, i, lab in (("Twenty", I20, "20-22 C"), ("Hot", I28, "28 C and over")):
    cmd(f"splitSpeed{tag}", bse(f4["lspd"]["b"][i], f4["lspd"]["se"][i]), f"fig4.lspd at {lab}, %")
    cmd(f"splitCad{tag}", bse(f4["lcad"]["b"][i], f4["lcad"]["se"][i]), f"fig4.lcad at {lab}, %")
    cmd(f"splitStride{tag}", bse(f4["lstr"]["b"][i], f4["lstr"]["se"][i]), f"fig4.lstr at {lab}, %")
    # share of the speed difference carried by cadence (ratio of two point estimates; no interval)
    cmd(f"splitCadShare{tag}", f"{100 * f4['lcad']['b'][i] / f4['lspd']['b'][i]:.0f}", f"fig4 lcad / lspd at {lab}, %")

# ---- 7d. second half against first half of the same run (estimates.json fig5) --------------------------------------
print("% 7d. Second half minus first half of sampled runs, WBGT 28 C and over against 10-15 C (fig5; last bin)")
f5 = E["fig5"]
cmd("driftRuns", big(f5["spd_decay"]["n"]), "fig5.spd_decay n")
cmd("driftRunners", big(f5["spd_decay"]["runners"]), "fig5.spd_decay runners")
cmd("driftHR", bse(f5["hr_drift"]["b"][-1], f5["hr_drift"]["se"][-1]), "fig5.hr_drift at 28 C and over, bpm")
cmd("driftSpeed", bse(f5["spd_decay"]["b"][-1], f5["spd_decay"]["se"][-1]), "fig5.spd_decay at 28 C and over, %")
cmd("driftCad", bse(f5["cad_decay"]["b"][-1], f5["cad_decay"]["se"][-1]), "fig5.cad_decay at 28 C and over, %")

# ---- 7e. race-distance efforts by what cadence did (race.json d_section21_v3) --------------------------------------
print("% 7e. Race-distance efforts at WBGT 18 C and over: loss as a multiple of the typical loss (race.json d_section21_v3)")
for name, key in (("rhoAll", "all hot with both terms"),
                  ("rhoRaceHeld", "in-race decay held (cad_drift >= -1%)"),
                  ("rhoRaceFell", "in-race decay (cad_drift < -1%)"),
                  ("rhoCoolHeld", "raw: held vs own cool races (>= -1%)"),
                  ("rhoCoolDown", "raw: down vs own cool races (< -1%)"),
                  ("rhoTrainHeld", "cadence held for the speed (give >= -1%)"),
                  ("rhoTrainDown", "cadence down for the speed (give < -1%)")):
    v = R[key]
    cmd(name, ci(v["rho"], v["ci95"][0], v["ci95"][1]), f"'{key}': rho, runner bootstrap")
    cmd(f"{name}N", big(v["efforts"]), f"'{key}': efforts")

# ---- 7f. the gap between the training curve and the race-day scale (followups.json A_gap) --------------------------
print("% 7f. Gap between the all-training scale and the official race-day scale, on the curve's scale (followups A_gap)")
A = F["A_gap"]
cmd("gapTrainAll", est(A["all_training"]), "A_gap.all_training: scale of all ordinary training runs")
for dist, tag in (("marathon", "Mar"), ("half", "Half")):
    e, s = A["efforts"][dist], A["gap_shares"][dist]
    cmd(f"gapPlat{tag}", est(e["scale_finish_time_within_runner"]), f"{dist}: platform runners, finish time within runner x distance")
    cmd(f"gapPlat{tag}N", big(e["n_within"]), f"{dist}: efforts in that fit")
    cmd(f"gapPlat{tag}Runners", big(e["runners_within"]), f"{dist}: runners in that fit")
    cmd(f"gapPred{tag}", est(e["scale_loss_vs_prediction_pooled"]), f"{dist}: loss against the fitness prediction, pooled, above 10 C")
    cmd(f"gapPredSlope{tag}", est(e["prediction_slope_on_curve_within_runner"]), f"{dist}: fitness prediction on the curve, within runner")
    cmd(f"gapPlatMinusOff{tag}", est(s["platform_minus_official"]), f"{dist}: platform scale minus official scale")
    cmd(f"gapShareDur{tag}", f"{100 * s['duration_share']:.0f}", f"{dist}: share of the gap from run duration, %")
    cmd(f"gapShareAcc{tag}", f"{100 * s['acclimation_difference_share']:.0f}", f"{dist}: share from the exposure-index difference, %")
    cmd(f"gapShareRace{tag}", f"{100 * s['racing_the_distance_same_cohort_share']:.0f}", f"{dist}: share from racing the distance, same cohort, %")
    cmd(f"gapSharePlat{tag}", num(100 * s["platform_to_official_share"], 0), f"{dist}: share from platform to official cohort, %")
fifth = A["efforts_by_multiplier_fifth"]
cmd("gapFifthLow", est(fifth[0]), "lowest fifth of the predicted individual multiplier: scale")
cmd("gapFifthHigh", est(fifth[4]), "highest fifth of the predicted individual multiplier: scale")
cmd("gapEffortsPooled", est(A["efforts_pooled"]), "all race-distance efforts above 10 C with a held-out prediction: scale")

# ---- 7g. platform efforts inside official editions (followups.json B) ----------------------------------------------
print("% 7g. Platform race-distance efforts on the date, distance and start line of an official edition (followups B)")
B = F["B_platform_in_editions"]
cmd("edlEfforts", big(B["in_effort_table"]), "B.in_effort_table")
cmd("edlEditions", big(B["editions"]), "B.editions")
cmd("edlRunners", big(B["runners"]), "B.runners")
cmd("edlSlope", est(B["slope_loss_on_official_effect"]), "B: platform loss on the official edition effect, edition-clustered")
cmd("edlPOne", f"{B['p_slope_equals_1']:.2f}", "B: p for slope = 1")
cmd("edlPZero", f"{B['p_slope_equals_0']:.2f}", "B: p for slope = 0")

# ---- 7h. the exposure index on race-distance efforts: interaction or level (followups.json D) ----------------------
print("% 7h. Exposure index on race-distance efforts: proportional term against a level term (followups D)")
D = F["D_score_on_race_efforts"]
cmd("formInt", est(D["form"]["pooled"]["interaction_only"]["tc_x_score"]), "D.form.pooled: cost x index, no level term")
cmd("formIntFree", est(D["form"]["pooled"]["with_level_terms"]["tc_x_score"]), "D.form.pooled: cost x index with level terms")
cmd("formLevel", est(D["form"]["pooled"]["with_level_terms"]["score_level"]), "D.form.pooled: level term of the index, %")
cmd("formIntFreeMar", est(D["form"]["marathon"]["with_level_terms"]["tc_x_score"]), "D.form.marathon: cost x index with level terms")
cmd("formIndexCostCorr", f"{D['corr_score_typical_cost']:.2f}", "D: correlation of the index with the typical cost")
cmd("splitPred", est(D["split"]["fitness_prediction"]), "D.split: one unit of index on the fitness prediction, %")
cmd("splitFinish", est(D["split"]["finish_time"]), "D.split: one unit of index on the finish time, %")
cmd("splitLoss", est(D["split"]["loss_vs_prediction"]), "D.split: one unit of index on the loss, %")
cmd("splitN", big(D["split"]["finish_time"]["n"]), "D.split: efforts")
cmd("splitRunners", big(D["split"]["finish_time"]["runners"]), "D.split: runners")

# ---- 7i. relative-speed baseline check (followups.json C2) --------------------------------------------------------
print("% 7i. Hottest minus coolest WBGT band at the same relative speed, two baselines (followups C2)")
C2 = F["C2_relative_speed_baseline"]
for tag, key in (("ThirtyDay", "relspd30"), ("Fixed", "relspdfix")):
    h, c = C2[f"{key}:hr_res"], C2[f"{key}:cad_res"]
    cmd(f"relHR{tag}", f"{num(h['hot_minus_cool_min'])} to {num(h['hot_minus_cool_max'])}", f"C2 {key}: HR gap range across speed bins, bpm")
    cmd(f"relCad{tag}", f"{num(c['hot_minus_cool_min'])} to {num(c['hot_minus_cool_max'])}", f"C2 {key}: cadence gap range across speed bins, %")
cmd("relCorr", f"{C2['corr_of_the_two_relative_speeds']:.2f}", "C2: correlation of the two relative speeds")

# ---- 7j. race-distance efforts within runner: cadence and stride length (race.json a_within_runner) ---------------
# These are within-runner slopes per unit of the typical race-day heat cost (so a loss slope of 1 is the typical cost),
# from the earlier cadence analysis; the JSON stores estimates with runner-bootstrap 95% intervals.
print("% 7j. Race-distance efforts within runner, per unit of typical heat cost (race.json a_within_runner)")
RW = json.load(open("/weather/data/fade/cadence_decay_20261029/race.json"))["a_within_runner"]


def rw(key, scale=1.0, nd=2):
    """Format a stored {'est', 'ci95'} estimate, optionally rescaled (100 turns a share into a percentage)."""
    v = RW[key]
    return ci(scale * v["est"], scale * v["ci95"][0], scale * v["ci95"][1], nd)


cmd("raceWithinN", big(RW["efforts"]), "a_within_runner.efforts")
cmd("raceWithinRunners", big(RW["runners"]), "a_within_runner.runners")
cmd("raceSpeed", rw("lspeed per unit typical cost"), "log speed per unit typical cost, %")
cmd("raceCad", rw("lcad per unit typical cost"), "log cadence per unit typical cost, %")
cmd("raceStride", rw("lstride per unit typical cost"), "log stride length per unit typical cost, %")
cmd("raceCadShare", rw("share of the heat speed loss that is cadence (rest stride)", 100, 0), "cadence share of the heat speed loss, %")
cmd("raceCadDrift", rw("cad_drift per unit typical cost"), "within-effort cadence change per unit typical cost, %")
# heat slope of the loss once the effort's mean cadence is in the model (1 = no change); descriptive, not causal
m = RW["heat slope of loss | lcad"]
cmd("raceCondSlope", ci(m["conditional"], *m["conditional_ci95"]), "heat slope of loss conditional on the effort's cadence")
cmd("raceCondShare", ci(100 * m["share_carried"], 100 * m["share_ci95"][0], 100 * m["share_ci95"][1], 0), "share of the heat slope that moves with cadence, %")

# ---- 7k. inputs of the gap decomposition that the text quotes (followups.json A_gap) -------------------------------
print("% 7k. Exposure index of hot race-distance efforts and hot training runs, and sensitivity fits (followups A_gap)")
for dist, tag in (("marathon", "Mar"), ("half", "Half")):
    sc, e, s = A["effort_scores"][dist], A["efforts"][dist], A["gap_shares"][dist]
    cmd(f"gapIndexHot{tag}", num(sc["mean_score_hot_efforts"]), f"{dist}: mean exposure index on hot efforts")
    cmd(f"gapHotEfforts{tag}", big(sc["n_hot"]), f"{dist}: hot efforts (WBGT 18 C and over)")
    cmd(f"gapPredN{tag}", big(e["n_above_10C"]), f"{dist}: efforts above 10 C in the pooled loss fit")
    cmd(f"gapPlatPred{tag}", est(e["scale_finish_time_within_runner_plus_prediction"]), f"{dist}: finish-time scale with the fitness prediction as covariate")
    cmd(f"gapPlatTrend{tag}", est(e["scale_finish_time_within_runner_plus_trend"]), f"{dist}: finish-time scale with a runner time trend")
    cmd(f"gapUnits{tag}", num(s["gap_in_scale_units"]), f"{dist}: official scale minus all-training scale")
cmd("splitCurveFinish", est(D["split"]["finish_time"]["curve_coef"]), "D.split: finish time on the curve, within runner x distance")
cmd("splitCurvePred", est(D["split"]["fitness_prediction"]["curve_coef"]), "D.split: fitness prediction on the curve, same model")
cmd("edlSDOfficial", num(B["sd_official_effect_pct"]), "B: SD of the official edition effect across linked editions, %")
cmd("edlSDPlatform", num(B["sd_platform_loss_pct"]), "B: SD of the platform loss across linked efforts, %")


# =====================================================================================================================
# Sections 7l and 7m (10 November 2026): the same quantities in the units used by earlier work, and the cadence results
# with the accompanying slowing held fixed. Both source files hold aggregates only.
# =====================================================================================================================
def blh(d, nd=2, scale=1.0):
    """Format a stored {'b', 'lo', 'hi'} regression estimate (scale=100 turns a share into a percentage)."""
    return ci(scale * d["b"], scale * d["lo"], scale * d["hi"], nd)


def mlh(d, nd=2, scale=1.0):
    """Format a stored {'mean', 'lo', 'hi'} group mean with its runner-clustered interval."""
    return ci(scale * d["mean"], scale * d["lo"], scale * d["hi"], nd)


def eci(d, nd=2, scale=1.0):
    """Format a stored {'est', 'ci95'} bootstrap estimate."""
    return ci(scale * d["est"], scale * d["ci95"][0], scale * d["ci95"][1], nd)


def rci(d, nd=2):
    """Format a stored {'rho', 'ci95'} multiple of the typical heat cost."""
    return ci(d["rho"], d["ci95"][0], d["ci95"][1], nd)


# ---- 7l. benchmarks in the units of earlier work (benchmarks.json) -------------------------------------------------
print("% 7l. The training-run results in the units of earlier work (benchmarks.json): cadence share of a speed")
print("%     difference, heart-rate drift in bpm and %, and decoupling of pace from heart rate")
a = BM["A_cadence_share_cool"]
cmd("benchCoolRuns", big(a["speed_only"]["n"]), "A: runs at WBGT 5-15 C")
cmd("benchCoolRunners", big(a["speed_only"]["runners"]), "A: runners in that fit")
# share of a within-runner speed difference between cool runs that is cadence (the rest is stride length)
cmd("benchCadShare", blh(a["speed_only"], 0, 100), "A.speed_only: cadence share of a cool-day speed difference, %")
cmd("benchCadShareFit", blh(a["plus_fitness"], 0, 100), "A.plus_fitness: same with the fitness control, %")
for tag, key in (("Slow", "slowest"), ("Mid", "middle"), ("Fast", "fastest")):
    cmd(f"benchCadShare{tag}", blh(a["by_own_speed_third"][key], 0, 100), f"A.by_own_speed_third.{key}: cadence share, %")
cmd("benchCadMedian", f"{a['cohort']['median_cadence_spm']:.0f}", "A.cohort: median cadence, steps per minute")
cmd("benchPaceMedian", num(a["cohort"]["median_pace_min_km"]), "A.cohort: median pace, min/km")
# what cadence did in heat against what the same slowing does to cadence on a cool day
for tag, key in (("Mild", "mild_20_22"), ("Hot", "severe_28_plus")):
    h = a["heat_vs_benchmark"][key]
    cmd(f"benchSpeed{tag}", num(h["speed_loss"]), f"A.heat_vs_benchmark.{key}: speed difference, %")
    cmd(f"benchCadObs{tag}", num(h["cadence_observed"]), f"A.heat_vs_benchmark.{key}: cadence difference observed, %")
    cmd(f"benchCadOrd{tag}", num(h["cadence_if_ordinary_slowing"]), f"A.heat_vs_benchmark.{key}: cadence difference the cool-day share implies, %")
b = BM["B_cadence_share_within_run"]
cmd("benchHalfShareCool", blh(b["cool_under_15"], 0, 100), "B.cool_under_15: cadence share of second-half speed change, training, %")
cmd("benchHalfShareHot", blh(b["hot_24_plus"], 0, 100), "B.hot_24_plus: same at WBGT 24 C and over, %")
c = BM["C_hr_drift_units"]
cmd("benchHRPerSpeed", blh(c["bpm_per_pct_speed_change"]), "C: bpm of drift per 1% of second-half speed change")
cmd("benchDriftRuns", big(c["n"]), "C: runs with both halves sampled")
cmd("benchDriftRunners", big(c["runners"]), "C: runners")
# five WBGT bands; macro names cannot carry digits, so the bands are named
BANDS = (("Cold", "<10", "below 10 C"), ("Cool", "10-15", "10-15 C"), ("Warm", "20-24", "20-24 C"),
         ("Hot", "24-28", "24-28 C"), ("Severe", ">=28", "28 C and over"))
for tag, key, lab in BANDS:
    v = c["bands"][key]
    cmd(f"hrDriftBpm{tag}", mlh(v["hr_drift_bpm"], 1), f"C.bands {lab}: second half minus first half, bpm")
    cmd(f"hrDriftPct{tag}", mlh(v["hr_drift_pct"], 1), f"C.bands {lab}: same as % of first-half heart rate")
    cmd(f"hrDriftPerH{tag}", mlh(v["hr_drift_bpm_per_h"], 1), f"C.bands {lab}: bpm per hour of run")
    cmd(f"hrDriftFixed{tag}", num(v["hr_drift_bpm_at_fixed_speed"], 1), f"C.bands {lab}: bpm with the second-half speed change removed")
    cmd(f"hrDriftN{tag}", big(v["n"]), f"C.bands {lab}: runs")
    st = c["steady_45_90min"][key]
    cmd(f"hrSteadyBpm{tag}", mlh(st["hr_drift_bpm"], 1), f"C.steady_45_90min {lab}: bpm, runs of 45-90 min at even pace")
    cmd(f"hrSteadyN{tag}", big(st["hr_drift_bpm"]["n"]), f"C.steady_45_90min {lab}: runs")
d = BM["D_decoupling"]
for tag, key, lab in BANDS:
    v = d["bands_raw_pct"][key]
    cmd(f"decoup{tag}", mlh(v, 1), f"D.bands_raw_pct {lab}: decoupling of speed per beat, second half against first, %")
    cmd(f"decoupOverFive{tag}", f"{100 * v['share_over_5pct']:.0f}", f"D {lab}: share of runs with decoupling over 5%, %")
for tag, key, lab in BANDS:
    if key == "10-15":
        continue  # the reference band of the within-runner contrast
    cmd(f"decoupWithin{tag}", blh(d["vs_10_15_within_runner"][key], 1), f"D.vs_10_15_within_runner {lab}: points of decoupling against 10-15 C")
e = BM["E_per_degree"]
cmd("perDegHR", blh(e["hr_bpm_per_C_at_matched_pace"]), "E: bpm per C of WBGT above 15, matched pace")
cmd("perDegHRPct", blh(e["hr_pct_per_C_at_matched_pace"]), "E: % of heart rate per C above 15, matched pace")
cmd("perDegCad", blh(e["cadence_pct_per_C_at_matched_pace"], 3), "E: % of cadence per C above 15, matched pace")

# ---- 7m. cadence with the accompanying slowing held fixed (speed_adjusted.json) ------------------------------------
print("% 7m. Cadence change net of the slowing it accompanies (speed_adjusted.json): race-distance efforts split by")
print("%     cadence, the cool-day comparison, race efforts by WBGT band, and training runs at a fixed speed change")
sm, A2 = SA["sample"], SA["A_race_effort_contrast"]
cmd("spadjHotN", big(sm["hot_18_plus"]), "sample: hot efforts (WBGT 18 C and over) with both cadence terms")
cmd("spadjHotRunners", big(sm["hot_runners"]), "sample: runners with a hot effort")
cmd("spadjCoolN", big(sm["cool_under_15"]), "sample: cool efforts (WBGT under 15 C)")
cmd("spadjCoolRunners", big(sm["cool_runners"]), "sample: runners with a cool effort")
cmd("spadjContrast", eci(A2["as_first_estimated"]["fell_minus_held"]), "A: fell minus held, multiples of the typical cost")
# what else differs between the two groups: the efforts whose cadence fell also slowed far more in the second half
for tag, key in (("Held", "held"), ("Fell", "fell")):
    g = A2["group_profile"][key]
    cmd(f"spadjSlow{tag}", mlh(g["second_half_slowing_pct"], 1), f"A.group_profile.{key}: second-half slowing, %")
    cmd(f"spadjCad{tag}", mlh(g["cadence_change_pct"], 1), f"A.group_profile.{key}: second-half cadence change, %")
    cmd(f"spadjLoss{tag}", mlh(g["loss_pct"], 1), f"A.group_profile.{key}: loss against the fitness prediction, %")
    cmd(f"spadjWbgt{tag}", num(g["wbgt"]["mean"], 1), f"A.group_profile.{key}: mean WBGT, C")
th = A2["within_thirds_of_slowing"]
for tag, key in (("Low", "slowed_least"), ("Mid", "middle"), ("High", "slowed_most")):
    cmd(f"spadjThird{tag}", eci(th[key]["fell_minus_held"]), f"A.within_thirds.{key}: fell minus held inside a third of slowing")
cmd("spadjThirdPooled", eci(th["fell_minus_held_within_thirds"]), "A.within_thirds: fell minus held, pooled within thirds")
aj = A2["adjusted_for_slowing"]
cmd("spadjCoolRel", blh(aj["cool_relation"]["cadence_change_per_1pct_slowing"], 3), "A.adjusted.cool_relation: % cadence per 1% of second-half slowing, cool efforts")
cmd("spadjCorrHot", num(aj["cool_relation"]["raw_correlation_hot"]), "A.adjusted: correlation of cadence change with slowing, hot efforts")
cmd("spadjCorrCool", num(aj["cool_relation"]["raw_correlation_cool"]), "A.adjusted: same, cool efforts")
cmd("spadjAdjContrast", eci(aj["linear"]["same_cut_minus_1pct"]["fell_minus_held"]), "A.adjusted.linear: fell minus held on cadence net of slowing")
cmd("spadjAdjContrastQuad", eci(aj["quadratic_plus_climb"]["same_cut_minus_1pct"]["fell_minus_held"]), "A.adjusted.quadratic_plus_climb: same, flexible adjustment")
cmd("spadjMoved", big(aj["linear"]["moved_group_vs_first_estimate"]), "A.adjusted.linear: efforts that change group")
jm = A2["joint_model"]
cmd("spadjJointCad", eci(jm["both"]["per_1pct_cadence_change"], 3), "A.joint_model.both: multiple of typical cost per 1% of cadence change")
cmd("spadjJointSlow", eci(jm["both"]["per_1pct_slowing"], 3), "A.joint_model.both: per 1% of second-half slowing")
cmd("spadjCadOnly", eci(jm["cadence_only"]["per_1pct_cadence_change"], 3), "A.joint_model.cadence_only")
cmd("spadjSlowOnly", eci(jm["slowing_only"]["per_1pct_slowing"], 3), "A.joint_model.slowing_only")
# the same split on cool days: a cadence fall goes with a larger loss there too, so only the difference is about heat
cc = A2["cool_effort_comparison"]
cmd("spadjGapHot", eci(cc["hot"]["loss_gap_pct"]), "A.cool_effort_comparison.hot: loss, fell minus held, %")
cmd("spadjGapCool", eci(cc["cool"]["loss_gap_pct"]), "A.cool_effort_comparison.cool: loss, fell minus held, %")
cmd("spadjCoolFellN", big(cc["cool"]["efforts_fell"]), "cool efforts whose cadence fell by more than 1%")
cmd("spadjCoolHeldN", big(cc["cool"]["efforts_held"]), "cool efforts whose cadence held")
cmd("spadjCoolLossFell", num(cc["cool"]["loss_fell_pct"]), "cool efforts, cadence fell: mean loss, %")
cmd("spadjCoolLossHeld", num(cc["cool"]["loss_held_pct"]), "cool efforts, cadence held: mean loss, %")
cmd("spadjGapDiff", eci(cc["hot_gap_minus_cool_gap_pct"]), "hot gap minus cool gap, % of finish time")
cmd("spadjGapDiffRho", eci(cc["heat_specific_gap_in_rho_units"]), "the same in multiples of the typical cost")
cmd("spadjGapDiffShare", f"{100 * cc['hot_gap_minus_cool_gap_pct']['share_boot_above_0']:.0f}", "share of bootstrap draws above zero, %")
nt = A2["rho_net_of_cool_day_loss"]
cmd("spadjNetFell", ci(nt["fell"]["rho_net_of_cool"], *nt["fell"]["ci95"]), "rho net of the same group's cool-day loss: cadence fell")
cmd("spadjNetHeld", ci(nt["held"]["rho_net_of_cool"], *nt["held"]["ci95"]), "rho net of the same group's cool-day loss: cadence held")
cmd("spadjNetDiff", eci(nt["fell_minus_held"]), "difference of the two")
# race-distance efforts by WBGT band, within runner, against 10-15 C
B2 = SA["B_race_efforts_by_wbgt"]
w = B2["within_runner_vs_10_15"]
cmd("raceBandN", big(w["ps"]["n"]), "B.within_runner_vs_10_15.ps: efforts")
cmd("raceBandRunners", big(w["ps"]["runners"]), "B.within_runner_vs_10_15.ps: runners")
RB = (("Mild", "15-18", "15-18 C"), ("Warm", "18-22", "18-22 C"), ("Hot", "22-26", "22-26 C"), ("Severe", ">=26", "26 C and over"))
for tag, key, lab in RB:
    cmd(f"raceBandSlow{tag}", blh(w["ps"][key], 1), f"B.within.ps {lab}: second-half slowing, points against 10-15 C")
    cmd(f"raceBandCad{tag}", blh(w["cad_drift"][key], 1), f"B.within.cad_drift {lab}: second-half cadence change, points")
    cmd(f"raceBandHR{tag}", blh(w["hr_drift"][key], 1), f"B.within.hr_drift {lab}: heart-rate drift, % points")
    cmd(f"raceBandDec{tag}", blh(w["dec"][key], 1), f"B.within.dec {lab}: decoupling, points")
    cmd(f"raceBandCadFixed{tag}", blh(B2["cadence_change_at_fixed_slowing_vs_10_15"][key]), f"B.cadence_change_at_fixed_slowing {lab}: cadence change at the same slowing, points")
cmd("raceBandCadPerSlow", blh(B2["cadence_change_at_fixed_slowing_vs_10_15"]["per_1pct_slowing"], 3), "B: % cadence per 1% of second-half slowing, all efforts")
rv = B2["raw_by_band"]
cmd("raceBandSevereN", big(rv[">=26"]["second_half_slowing_pct"]["n"]), "B.raw_by_band: efforts at 26 C and over")
cmd("raceBandSevereRunners", big(rv[">=26"]["second_half_slowing_pct"]["runners"]), "B.raw_by_band: runners at 26 C and over")
for tag, key, lab in (("Cold", "<10", "below 10 C"), ("Cool", "10-15", "10-15 C")) + RB:
    cmd(f"raceRawSlow{tag}", mlh(rv[key]["second_half_slowing_pct"], 1), f"B.raw_by_band {lab}: second-half slowing, %")
    cmd(f"raceRawHR{tag}", mlh(rv[key]["hr_drift_pct"], 1), f"B.raw_by_band {lab}: heart-rate drift, %")
    cmd(f"raceRawDec{tag}", mlh(rv[key]["decoupling_pct"], 1), f"B.raw_by_band {lab}: decoupling, %")
    cmd(f"raceRawOverFive{tag}", f"{100 * rv[key]['share_decoupling_over_5pct']:.0f}", f"B.raw_by_band {lab}: share with decoupling over 5%, %")
sh = B2["cadence_share_of_second_half_speed_change"]
cmd("raceHalfShareCool", blh(sh["cool_under_15"], 0, 100), "B: cadence share of the second-half speed change, cool efforts, %")
cmd("raceHalfShareHot", blh(sh["hot_18_plus"], 0, 100), "B: same, efforts at 18 C and over, %")
cmd("raceHalfShareHotter", blh(sh["hot_22_plus"], 0, 100), "B: same, efforts at 22 C and over, %")
# training runs: second half against first half at 28 C and over, observed and at a fixed speed change
C3 = SA["C_training_within_run"]
cmd("trainHalfRuns", big(C3["cad_decay"]["observed"]["n"]), "C.cad_decay: runs")
cmd("trainHalfRunners", big(C3["cad_decay"]["observed"]["runners"]), "C.cad_decay: runners")
cmd("trainCadObs", blh(C3["cad_decay"]["observed"][">=28"]), "C.cad_decay.observed at 28 C and over, % against 10-15 C")
cmd("trainCadFixed", blh(C3["cad_decay"]["at_fixed_speed_change"][">=28"]), "C.cad_decay.at_fixed_speed_change at 28 C and over, %")
cmd("trainCadFixedQuad", blh(C3["cad_decay"]["at_fixed_speed_change_quadratic"][">=28"]), "C.cad_decay, quadratic in the speed change")
cmd("trainCadFixedFree", blh(C3["cad_decay"]["at_fixed_speed_change_slope_free_in_heat"][">=28"]), "C.cad_decay, speed-change slope free to differ in heat")
cmd("trainCadPerSpeed", blh(C3["cad_decay"]["at_fixed_speed_change"]["per_1pct_speed_change"], 3), "C.cad_decay: % cadence per 1% of speed change")
cmd("trainCadRemain", f"{100 * C3['share_of_cadence_fall_not_accounted_for_by_slowing_28_plus']:.0f}", "C: share of the cadence fall that remains at a fixed speed change, %")
cmd("trainHRObs", blh(C3["hr_drift"]["observed"][">=28"]), "C.hr_drift.observed at 28 C and over, bpm against 10-15 C")
cmd("trainHRFixed", blh(C3["hr_drift"]["at_fixed_speed_change"][">=28"]), "C.hr_drift.at_fixed_speed_change at 28 C and over, bpm")
cmd("trainHRPerSpeed", blh(C3["hr_drift"]["at_fixed_speed_change"]["per_1pct_speed_change"], 3), "C.hr_drift: bpm per 1% of speed change")
cmd("trainSpdObs", blh(C3["spd_decay"]["observed"][">=28"]), "C.spd_decay.observed at 28 C and over, %")

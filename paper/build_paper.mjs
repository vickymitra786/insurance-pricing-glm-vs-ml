// Builds out/pricing.docx. Every result is read from results/pricing.json (written by run_all.py);
// tests/check_paper.py then checks every number in the paper against that file.
// Author: Vivek Mitra
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun, Packer, PageNumber, Paragraph,
  Table, TableCell, TableRow, TextRun, WidthType,
} from "docx";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const R = JSON.parse(readFileSync(path.join(root, "results", "pricing.json"), "utf8"));
const parity = JSON.parse(readFileSync(path.join(root, "results", "pricing.parity.json"), "utf8"));
const fig = name => readFileSync(path.join(root, "out", "figures", name));

const P = R.point, C = R.ci95, D = R.data, L = R.double_lift, S = R.segments, rob = R.robustness;
const n = v => Math.round(v).toLocaleString("en");
const d = (v, k = 2) => v.toFixed(k);
const ci = (a, k = 2, s = 1) => `[${d(s * a[0], k)}, ${d(s * a[1], k)}]`;
const dev = v => d(100 * v, 3);
const eur = v => v.toLocaleString("en", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const cuts = rob.map(r => r.dev_cut_pct);
const wins = rob.filter(r => r.dev_cut_pct > 0).length;
const sevWorse = R.severity.dev_glm >= R.severity.dev_flat;
const imp = Object.entries(R.importance).sort((a, b) => b[1].gbm - a[1].gbm);
const cal = m => { const i = R.worst_tenth[m], c = R.calibration[m]; return { t: i + 1, p: d(100 * c.pred[i]), a: d(100 * c.actual[i]) }; };
const FN = { DrivAge: "Driver age", BonusMalus: "Bonus-malus", VehAge: "Vehicle age", VehPower: "Vehicle power",
  Density: "Population density", Area: "Area", VehBrand: "Vehicle brand", VehGas: "Fuel", Region: "Region" };

// ---------- small builders
const FONT = "Calibri";
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: 21, ...o });
const sub = t => run(t, { subScript: true, italics: true });          // real subscripts, not underscores
const it = t => run(t, { italics: true });
function para(parts, o = {}) {
  const children = (Array.isArray(parts) ? parts : [parts]).map(x => (typeof x === "string" ? run(x) : x));
  return new Paragraph({ children, spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...o });
}
const h1 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 120 } });
const h2 = t => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 180, after: 80 } });
const eq = parts => new Paragraph({ children: parts, alignment: AlignmentType.CENTER, spacing: { after: 120 } });
const bullet = parts => para(parts, { bullet: { level: 0 }, alignment: AlignmentType.LEFT });
const caption = t => new Paragraph({ children: [run(t, { size: 18, italics: true })], spacing: { after: 200 }, alignment: AlignmentType.LEFT });
const image = (name, w, h) => new Paragraph({ children: [new ImageRun({ type: "png", data: fig(name), transformation: { width: w, height: h } })], alignment: AlignmentType.CENTER });
const line = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(head, rows, widths) {
  const cell = (t, i, bold, top) => new TableCell({
    width: { size: widths[i], type: WidthType.PERCENTAGE },
    borders: { top: top ? line : none, bottom: bold ? line : none, left: none, right: none },
    children: [new Paragraph({ children: [run(String(t), { size: 18, bold })], alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT })],
  });
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [new TableRow({ children: head.map((t, i) => cell(t, i, true, true)) }),
      ...rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, i, false, false)) }))],
  });
}

// ---------- content
const title = [
  new Paragraph({ children: [run("Does machine learning beat the actuarial GLM?", { size: 34, bold: true })], alignment: AlignmentType.CENTER, spacing: { after: 60 } }),
  new Paragraph({ children: [run("An honest benchmark on French motor insurance", { size: 26 })], alignment: AlignmentType.CENTER, spacing: { after: 200 } }),
  new Paragraph({ children: [run("Vivek Mitra, Independent researcher, vivekmitra.com", { size: 22 })], alignment: AlignmentType.CENTER, spacing: { after: 60 } }),
  new Paragraph({ children: [run("November 2026", { size: 20, color: "555555" })], alignment: AlignmentType.CENTER, spacing: { after: 280 } }),
];

const abstract = [
  h1("Abstract"),
  para(`Gradient-boosted trees are increasingly proposed as an alternative to the Poisson generalised linear model (GLM) that insurers use to price motor claim frequency. We compare the two on ${n(D.policies)} French motor third-party liability policies (freMTPL2, CASdatasets). The GLM uses the published actuarial banding; the boosting model uses the same exposure offset and the raw rating factors. Both are fitted on about 80% of policies and scored on ${n(D.test)} held-out policies, with policies that share identical rating factors kept on the same side of the split. Boosting lowers the held-out Poisson deviance from ${dev(P.dev_glm)} to ${dev(P.dev_gbm)} (×10⁻²), a ${d(P.dev_cut_pct)}% reduction (95% paired bootstrap interval ${ci(C.dev_cut_pct)}%), and raises the Gini index by ${d(P.gini_gain, 3)} ${ci(C.gini_gain, 3)}, from ${d(P.gini_glm, 3)} to ${d(P.gini_gbm, 3)}. It has the lower deviance on all ${rob.length} further splits. In a double-lift chart, actual claims are closer to the boosting prediction in ${L.ml_closer_tenths} of 10 groups. The segments the GLM misses most are pairs of factors, such as new vehicles by fuel type. A Gamma GLM for claim size ${sevWorse ? "did not beat" : "only narrowly beat"} a flat average on held-out claims. Both models run in a web browser and match Python to within ${parity.max_rel_diff.toExponential(1)}.`),
];

const intro = [
  h1("1. Introduction"),
  para("Most motor insurers set prices with a generalised linear model: a Poisson model for how often a policy claims, a Gamma model for how much each claim costs, and a table of multiplicative relativities that underwriters, regulators and pricing committees can read line by line (McCullagh and Nelder, 1989; Ohlsson and Johansson, 2010; Goldburd et al., 2020). Tree-based machine learning has been studied as a way to improve these tariffs (Henckaerts et al., 2021)."),
  para("This paper asks a narrow question. On a large, public motor portfolio, does a gradient-boosting model with a Poisson objective predict claim frequency better than the published benchmark GLM for that portfolio? If so, by how much, where, and how sure can we be? We use one fixed, seeded held-out test set, report every comparison with a paired bootstrap interval, repeat the comparison on five more splits, and keep identical rating profiles on the same side of each split so that the flexible model cannot gain from recognising repeated profiles. We call one model better only when the interval for the difference excludes zero."),
  para("The paper makes four contributions. First, a benchmark that one command rebuilds from public data. Second, the standard actuarial diagnostics, namely Lorenz curves and Gini indices (Frees et al., 2011), calibration by tenth and double-lift charts, applied to both models on the same held-out data. Third, a segment-level audit that shows where the GLM's expected claim count is furthest from actual claims and how far boosting's is. Fourth, a browser implementation of both models, checked against Python, that lets a non-specialist price a driver and compare the two models."),
];

const dataSec = [
  h1("2. Data"),
  para(`freMTPL2freq holds ${n(D.policies)} motor third-party liability policies from a French insurer (${n(D.policy_years)} policy-years in total), with the number of claims and nine rating factors: driver age, bonus-malus level, vehicle age, vehicle power class, population density of the home town, an area code from A (rural) to F (urban), vehicle brand (coded), fuel type and region (22 former French regions). freMTPL2sev holds ${n(D.claim_rows)} individual claim amounts linked by policy number, of which ${n(D.claim_records)} match a policy in freMTPL2freq. Both tables are part of the CASdatasets R package (Dutang and Charpentier) that accompanies Charpentier (2014). We take them from OpenML, datasets 41214 and 41215 (Vanschoren et al., 2013).`),
  para(`We apply the cleaning that is standard for this data (Noll et al., 2018): claim counts above 4 are set to 4, and exposure is capped at one year. Individual claims are capped at €100,000; the ${d(100 * D.train_loss_share_above_cap, 1)}% of training-data recorded cost above the cap is returned as a flat large-loss load of ${d(R.pure_premium.large_loss_load, 3)}. ${n(D.claiming_policies_without_amount)} policies with claims have no matching amount in the severity table. We therefore model claim frequency on the full claim counts, and set the cost per predicted claim from recorded amounts, so the overall premium level matches recorded cost on the training data.`),
];

const method = [
  h1("3. Method"),
  h2("3.1 Baseline: Poisson GLM"),
  para(["For policy ", it("i"), " with exposure ", it("e"), sub("i"), " (years) and claim count ", it("N"), sub("i"), ", the GLM assumes ", it("N"), sub("i"), " ~ Poisson(", it("e"), sub("i"), it("λ"), sub("i"), ") with"]),
  eq([it("log λ"), sub("i"), it(" = β"), sub("0"), it(" + Σ"), sub("j"), it(" β"), sub("j"), it(" x"), sub("ij"), run(",   so   "),
    it("log E[N"), sub("i"), it("] = log e"), sub("i"), it(" + β"), sub("0"), it(" + Σ"), sub("j"), it(" β"), sub("j"), it(" x"), sub("ij"), run(",")]),
  para(`and log exposure is an offset with coefficient fixed at one. The covariates follow the GLM of Noll et al. (2018): area as a single slope over A=1 to F=6; vehicle power as a category, capped at 9; vehicle age in three bands (0, 1–10, 11+ years); driver age in seven bands (18–20, 21–25, 26–30, 31–40, 41–50, 51–70, 71+); bonus-malus as a slope, capped at 150; log density as a slope; and vehicle brand, fuel and region as categories. With the base levels removed this gives ${R.glm_coefficients.length} coefficients, fitted by iteratively reweighted least squares in statsmodels (Seabold and Perktold, 2010). A test refits the GLM on a sample with statsmodels' formula interface, which builds its own dummy variables, and checks that the deviance and every fitted value agree.`),
  h2("3.2 Challenger: gradient boosting"),
  para(`The challenger is LightGBM (Ke et al., 2017) with a Poisson objective and log exposure as the initial score, so it models the same quantity as the GLM. It sees the nine raw factors with no banding; brand and region are native categorical features. The settings are fixed, not tuned: learning rate 0.05, 31 leaves per tree and at least 200 policies per leaf. Trees are added until the Poisson loss on a validation slice of the training data has not improved for 100 rounds; the main run kept ${R.trees} trees. The validation slice holds about 15% of the training profiles, split in the same grouped way, and boosting is fitted on the rest; the test set is never used for any choice.`),
  h2("3.3 Claim size and pure premium"),
  para(`We fit a Gamma GLM with a log link to the average capped claim size per claiming policy, weighted by its number of claims, on the same covariates. On the ${n(R.severity.claims_policies_test)} held-out claiming policies its Gamma deviance was ${d(R.severity.dev_glm, 4)}, against ${d(R.severity.dev_flat, 4)} for the flat average claim size of the training data. ${sevWorse ? "Because the model did not beat the flat average, both prices use the flat average." : "Because the gain was negligible, both prices use the flat average."} The pure premium per policy-year is then predicted frequency times a cost per predicted claim, set on the training data so that predicted and recorded capped cost agree in total (€${eur(R.pure_premium.cost_per_claim.glm)} for the GLM, €${eur(R.pure_premium.cost_per_claim.gbm)} for boosting). The web demo also multiplies by the large-loss load. All of the difference between the two prices therefore comes from claim frequency.`),
  h2("3.4 Split and evaluation"),
  para(`The ${n(D.policies)} policies have ${n(D.rating_profiles)} distinct combinations of the nine rating factors. The split assigns each combination, with all its policies, to the test set with probability 20%, using a fixed seed (${R.seed}). This gives ${n(D.train)} training and ${n(D.test)} test policies (${n(D.test_policy_years)} policy-years, ${n(D.test_claims)} claims). All results are on the test policies.`),
  bullet([run("Poisson deviance. ", { bold: true }), "The mean over policies of the unit deviance 2[μ − N + N log(N/μ)], with μ the expected claim count, reported ×10⁻². Lower is better. A model with one overall rate, fitted on the training data, gives the no-model reference."]),
  bullet([run("Gini index. ", { bold: true }), "From the ordered Lorenz curve: policies sorted from lowest to highest predicted frequency, cumulative share of exposure against cumulative share of claims (or of capped claim cost). Gini is one minus twice the area under the curve (Frees et al., 2011)."]),
  bullet([run("Calibration by tenth. ", { bold: true }), "Policies sorted by predicted frequency into ten groups of equal exposure, actual against predicted frequency."]),
  bullet([run("Double lift. ", { bold: true }), "Policies sorted by the ratio of boosting's predicted frequency to the GLM's into ten groups of equal exposure. In each group we compare the actual frequency with each model's prediction (Goldburd et al., 2020)."]),
  bullet([run("Segments. ", { bold: true }), `Actual over expected claims for every value of each factor and for every pair of the eight non-regional factors (factors banded as in Table 3), keeping cells where the GLM expects at least 30 claims (${n(S.cells)} cells).`]),
  bullet([run("Explanations. ", { bold: true }), "TreeSHAP values (Lundberg et al., 2020) for boosting and, for the additive GLM, each factor's term minus its average over the test policies. Both are on the log-frequency scale, on all test policies."]),
  para(`Intervals are 95% percentile intervals from ${n(R.bootstrap)} bootstrap resamples of the test policies (Efron and Tibshirani, 1993). Each resample scores both models on the same policies, so every difference has a paired interval. The models are not refitted in each resample, so the intervals reflect test-set noise for these fitted models. Five further splits (seeds 1 to 5) refit both models and show variation from the training data as well.`),
];

const resultsTable = table(
  ["Metric (held-out test set)", "No model", "GLM", "Boosting", "Boosting − GLM [95% CI]"],
  [
    ["Poisson deviance ×10⁻²", dev(P.dev_null), dev(P.dev_glm), dev(P.dev_gbm), `−${dev(P.dev_gap)} ${ci(C.dev_gap, 3, -100).replace(/\[(.*), (.*)\]/, "[$2, $1]").replace(/-/g, "−")}`],
    ["Deviance reduction vs GLM", "", "", "", `${d(P.dev_cut_pct)}% ${ci(C.dev_cut_pct)}`],
    ["Gini, claim counts", "", d(P.gini_glm, 3), d(P.gini_gbm, 3), `+${d(P.gini_gain, 3)} ${ci(C.gini_gain, 3)}`],
    ["Gini, capped claim cost", "", d(P.gini_pp_glm, 3), d(P.gini_pp_gbm, 3), `+${d(P.gini_pp_gain, 3)} ${ci(C.gini_pp_gain, 3)}`],
    ["Capped pure premium per policy-year (actual €" + d(R.pure_premium.actual_test, 2) + ")", "", "€" + d(R.pure_premium.glm_test, 2), "€" + d(R.pure_premium.gbm_test, 2), ""],
  ], [34, 11, 12, 12, 31]);

const robTable = table(["Split", "GLM deviance ×10⁻²", "Boosting deviance ×10⁻²", "Reduction", "Trees"],
  [[`Main (seed ${R.seed})`, dev(P.dev_glm), dev(P.dev_gbm), d(P.dev_cut_pct) + "%", String(R.trees)],
    ...rob.map(r => [`Seed ${r.seed}`, dev(r.dev_glm), dev(r.dev_gbm), d(r.dev_cut_pct) + "%", String(r.trees)])], [28, 20, 22, 15, 15]);

const segRows = S.top_glm_misses.slice(0, 8).map(r => [r.segment, n(r.exposure), n(r.claims), d(r.ae_glm), d(r.ae_gbm)]);
const segTable = table(["Segment", "Policy-years", "Claims", "Actual ÷ GLM", "Actual ÷ boosting"], segRows, [40, 15, 12, 16, 17]);
const impTable = table(["Factor", "GLM mean |effect|", "Boosting mean |SHAP|"],
  imp.map(([f, v]) => [FN[f], d(v.glm, 3), d(v.gbm, 3)]), [40, 30, 30]);
const g = cal("glm"), b = cal("gbm");

const results = [
  h1("4. Results"),
  h2("4.1 Headline comparison"),
  para(`Table 1 gives the main result. Boosting lowers the held-out Poisson deviance by ${d(P.dev_cut_pct)}% relative to the GLM, with a 95% paired bootstrap interval of ${ci(C.dev_cut_pct)}%, which excludes zero. Against the no-model reference, the GLM removes ${d(P.explained_glm_pct)}% of the deviance and boosting ${d(P.explained_gbm_pct)}%. The Gini index on claim counts rises by ${d(P.gini_gain, 3)} ${ci(C.gini_gain, 3)}. On capped claim cost, which is much noisier, it rises by ${d(P.gini_pp_gain, 3)} ${ci(C.gini_pp_gain, 3)}; both intervals exclude zero. On the test set both models predict a capped pure premium of €${d(R.pure_premium.glm_test, 2)} (GLM) and €${d(R.pure_premium.gbm_test, 2)} (boosting) per policy-year, against €${d(R.pure_premium.actual_test, 2)} actual.`),
  caption(`Table 1. Held-out results. Intervals: ${n(R.bootstrap)} paired bootstrap resamples of the test policies.`),
  resultsTable,
  para(""),
  para(`Table 2 repeats the comparison on five more grouped splits, refitting both models each time. Boosting has the lower deviance on ${wins} of ${rob.length}, with reductions from ${d(Math.min(...cuts))}% to ${d(Math.max(...cuts))}%.`),
  caption("Table 2. The same comparison on six grouped splits."),
  robTable,
  h2("4.2 Ranking and calibration"),
  image("lorenz.png", 620, 272),
  caption("Figure 1. Ordered Lorenz curves on the test set. The further a curve bows below the diagonal, the better the model separates low and high risks."),
  para(`Figure 1 shows the Lorenz curves and Figure 2 calibration by tenth of predicted frequency. The largest calibration gap for the GLM is in tenth ${g.t}, where it predicted ${g.p} claims per 100 policy-years against ${g.a} actual; for boosting it is tenth ${b.t}, with ${b.p} predicted against ${b.a} actual.`),
  image("calibration.png", 620, 263),
  caption("Figure 2. Calibration by tenth of exposure, sorted by each model's own prediction."),
  h2("4.3 Double lift: where the predictions disagree, which is closer?"),
  para(`The two models often disagree. Half of the test policies have a boosting ÷ GLM frequency ratio between ${d(R.price_gap.quartiles[0])} and ${d(R.price_gap.quartiles[2])}, and ${d(100 * R.price_gap.share_over_20pct, 1)}% of test exposure has predictions more than 20% apart. Figure 3 sorts the test policies by this ratio. In the tenth where boosting predicts least relative to the GLM (mean ratio ${d(L.ratio[0])}), actual claims were ${d(L.ae_glm[0])} times the GLM's expected count and ${d(L.ae_gbm[0])} times boosting's; in the tenth where it predicts most (mean ratio ${d(L.ratio[9])}), the figures were ${d(L.ae_glm[9])} and ${d(L.ae_gbm[9])}. The actual frequency is closer to the boosting prediction in ${L.ml_closer_tenths} of the 10 groups.`),
  image("double_lift.png", 620, 282),
  caption("Figure 3. Double lift chart. Error bars: 95% Poisson interval on the actual claim count."),
  h2("4.4 Segments the GLM misprices"),
  para(`Of ${n(S.cells)} one-factor and two-factor segments, boosting's expected count was closer to actual claims in ${d(100 * S.ml_closer_share, 1)}%. ${S.flagged_glm} segments were more than two standard errors from the GLM's expectation, against ${S.flagged_gbm} for boosting; for a perfectly calibrated model about ${d(S.flagged_by_chance, 1)} would be expected by chance. The segments overlap and were not chosen in advance, so these counts are descriptive, not formal tests. Table 3 lists the cells furthest from the GLM's expectation, measured in standard errors; ${S.top_glm_misses.slice(0, 8).filter(r => r.segment.includes(", ")).length} of the 8 are pairs of factors, which the GLM has no interaction terms to price.`),
  caption("Table 3. The eight test segments furthest from the GLM's expected claim count (in standard errors). Bands: bonus-malus 50, 51-60, 61-80, 81-100, 101+; vehicle age 0, 1-3, 4-7, 8-12, 13+."),
  segTable,
  h2("4.5 What each model learned"),
  para("Figure 4 plots the average effect of each numeric factor on claim frequency under both models, as a multiplier against an average policy. The curves have similar shapes for bonus-malus, driver age, vehicle age and density. The GLM's curves are steps or straight lines on the log scale by construction; boosting's follow the data. One-factor averages cannot show interactions; the segments in Table 3 do. Table 4 ranks the factors by the average size of their effect. Boosting gives area almost no weight; density, which it also sees, carries related information in finer detail."),
  image("effects.png", 620, 413),
  caption("Figure 4. Average effect of each factor (multiplier on claim frequency). GLM: its own term for the factor, minus the term's average. Boosting: mean TreeSHAP value by factor value. Bins with fewer than 30 policies are left out."),
  caption("Table 4. Average absolute effect on log claim frequency, all test policies."),
  impTable,
];

const discussion = [
  h1("5. Discussion"),
  para(`On this portfolio, boosting predicts claim frequency better than the benchmark GLM by a modest margin, and every interval for the difference excludes zero. The margin looks small as a share of deviance because individual claim counts are mostly noise: even boosting removes only ${d(P.explained_gbm_pct)}% of the no-model deviance, and the GLM already removes ${d(P.explained_glm_pct)}%. As risk ranking, the Gini gain is ${d(P.gini_gain, 3)} ${ci(C.gini_gain, 3)}, and the two models' predictions differ by more than 20% on ${d(100 * R.price_gap.share_over_20pct, 1)}% of exposure. In the double lift, actual claims were closer to boosting in ${L.ml_closer_tenths} of 10 groups.`),
  para("For an insurer, these results support running boosting as a challenger next to the GLM rather than replacing it: read the double lift, and treat the segments where boosting is closer, mostly pairs such as vehicle age with fuel, as candidates for new GLM terms (Henckaerts et al., 2021). We did not test such an extended GLM here. Both models price a policy in a web browser from a small JSON file, and both can show the factors behind a price."),
];

const limits = [
  h1("6. Limitations"),
  bullet("One portfolio, one country and roughly one year of data. Results may differ for other lines, markets or periods, and there is no out-of-time test."),
  bullet(`The data have known quirks: ${n(D.claiming_policies_without_amount)} claiming policies without amounts, and repeated rating profiles. We handle the second with a grouped split, but cannot recover true policyholder identities.`),
  bullet("The GLM is the published benchmark specification without interaction terms. A GLM with interactions might narrow the gap; we did not test one."),
  bullet("Boosting settings were fixed, not tuned; tuning might change the margin. Bootstrap intervals hold the fitted models fixed; the extra splits address training variation."),
  bullet("Claim size uses a flat average because a Gamma GLM did not beat it on held-out claims. A different severity model or more claims data could change this."),
  bullet("The pure premium excludes expenses, reinsurance, profit and competition, so it is not a market price."),
];

const repro = [
  h1("7. Reproducibility"),
  para(["Code: ", run("github.com/vickymitra786/insurance-pricing-glm-vs-ml", { bold: true }), ` (GNU AGPL-3.0). One command, python run_all.py, downloads both tables from OpenML, fits and scores the models, runs the tests, draws the figures and builds this paper. Seeds are fixed throughout. Software: Python 3.13, NumPy, pandas, statsmodels 0.15.0, LightGBM 4.7.0. Three tests run every time: the GLM is compared with statsmodels' formula interface; the browser code prices ${n(parity.cases)} held-out policies and its largest relative difference from Python is ${parity.max_rel_diff.toExponential(1)}; and a script checks every number in this paper against the results file. An interactive version runs at vivekmitra.com/pricing.html.`]),
];

const refs = [
  "Charpentier, A. (ed.) (2014). Computational Actuarial Science with R. Chapman and Hall/CRC.",
  "Dutang, C. and Charpentier, A. CASdatasets: Insurance Datasets. R package. https://github.com/dutangc/CASdatasets. Tables freMTPL2freq and freMTPL2sev as published on OpenML, datasets 41214 and 41215 (version 1), accessed 21 November 2026.",
  "Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman and Hall.",
  "Frees, E. W., Meyers, G. and Cummings, A. D. (2011). Summarizing insurance scores using a Gini index. Journal of the American Statistical Association, 106(495), 1085–1098.",
  "Goldburd, M., Khare, A., Tevet, D. and Guller, D. (2020). Generalized Linear Models for Insurance Rating, 2nd ed. CAS Monograph Series No. 5. Casualty Actuarial Society.",
  "Henckaerts, R., Côté, M.-P., Antonio, K. and Verbelen, R. (2021). Boosting insights in insurance tariff plans with tree-based machine learning methods. North American Actuarial Journal, 25(2), 255–285.",
  "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30.",
  "Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. and Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2, 56–67.",
  "McCullagh, P. and Nelder, J. A. (1989). Generalized Linear Models, 2nd ed. Chapman and Hall.",
  "Noll, A., Salzmann, R. and Wüthrich, M. V. (2018). Case study: French motor third-party liability claims. SSRN 3164764.",
  "Ohlsson, E. and Johansson, B. (2010). Non-Life Insurance Pricing with Generalized Linear Models. Springer.",
  "Seabold, S. and Perktold, J. (2010). statsmodels: Econometric and statistical modeling with Python. Proceedings of the 9th Python in Science Conference.",
  "Vanschoren, J., van Rijn, J. N., Bischl, B. and Torgo, L. (2013). OpenML: networked science in machine learning. SIGKDD Explorations, 15(2), 49–60.",
].map(t => new Paragraph({ children: [run(t, { size: 19 })], spacing: { after: 80 }, indent: { left: 360, hanging: 360 } }));

const doc = new Document({
  creator: "Vivek Mitra", title: "Does machine learning beat the actuarial GLM?", description: "An honest benchmark on French motor insurance",
  styles: { default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 26, bold: true, color: "1F4E8C" } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 22, bold: true } }] },
  sections: [{
    properties: { page: { margin: { top: 1300, bottom: 1300, left: 1300, right: 1300 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run("Vivek Mitra · vivekmitra.com · ", { size: 16, color: "777777" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "777777" })] })] }) },
    children: [...title, ...abstract, ...intro, ...dataSec, ...method, ...results, ...discussion, ...limits, ...repro, h1("References"), ...refs],
  }],
});
writeFileSync(path.join(root, "out", "pricing.docx"), await Packer.toBuffer(doc));
console.log("  paper written: out/pricing.docx");

"""Fit both models on the training policies, score them on held-out policies, write every published number.

Outputs: results/pricing.json (all results), results/pricing.model.json (what the browser needs to
price a driver) and out/parity_cases.json (Python prices the browser must reproduce).
Author: Vivek Mitra
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
import gbm
import glm

SEED = 2026                       # the one fixed split behind every headline number
OTHER_SEEDS = [1, 2, 3, 4, 5]     # five more splits, to show the result is not luck
BOOT = 1000                       # bootstrap resamples of the test policies
TYPICAL = dict(DrivAge=45, BonusMalus=50, VehAge=6, VehPower=6, Density=500, Area="C",
               VehBrand="B1", VehGas="Regular", Region="R24")


# ---------------------------------------------------------------- metrics
def poisson_dev(y, mu):
    """Poisson unit deviance per policy (y = claims, mu = expected claims). Lower is better."""
    return 2 * (mu - y + np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0))


def gamma_dev(y, mu):
    return 2 * ((y - mu) / mu - np.log(y / mu))


def lorenz(order, y, w, c=1.0):
    """Ordered Lorenz curve: policies from lowest to highest predicted risk; x = exposure share, y = loss share.
    c holds bootstrap multiplicities (how often each policy was drawn)."""
    cw = np.concatenate([[0], np.cumsum((w * c)[order])])
    cy = np.concatenate([[0], np.cumsum((y * c)[order])])
    return cw / cw[-1], cy / cy[-1]


def gini(order, y, w, c=1.0):
    x, l = lorenz(order, y, w, c)
    return 1 - 2 * np.trapezoid(l, x)


def tenths(key, w):
    """Group 0..9 by key, each group holding a tenth of the exposure."""
    order = np.argsort(key, kind="stable")
    g = np.empty(len(key), int)
    g[order] = np.minimum((np.cumsum(w[order]) / w.sum() * 10).astype(int), 9)
    return g


def by_tenth(g, *cols):
    return [np.bincount(g, weights=c, minlength=10) for c in cols]


# ---------------------------------------------------------------- segments
def bands(df):
    bm, va, vp = df["BonusMalus"], df["VehAge"], df["VehPower"]
    return {
        "Driver age": glm.driv_band(df["DrivAge"].to_numpy()),
        "Bonus-malus": np.select([bm == 50, bm <= 60, bm <= 80, bm <= 100], ["50", "51-60", "61-80", "81-100"], "101+"),
        "Vehicle age": np.select([va == 0, va <= 3, va <= 7, va <= 12], ["0", "1-3", "4-7", "8-12"], "13+"),
        "Vehicle power": np.select([vp <= 5, vp <= 7, vp <= 9], ["4-5", "6-7", "8-9"], "10+"),
        "Area": df["Area"].to_numpy(), "Fuel": df["VehGas"].to_numpy(),
        "Brand": df["VehBrand"].to_numpy(), "Region": df["Region"].to_numpy(),
    }


def segments(test, e_glm, e_gbm, min_expected=30):
    """Actual vs expected claims in every one-factor segment and every pair of factors (region alone only)."""
    b = bands(test)
    pairs = [(k,) for k in b] + [(p, q) for i, p in enumerate(list(b)[:-1]) for q in list(b)[i + 1:-1]]
    rows = []
    for names in pairs:
        label = pd.Series([", ".join(f"{n} {v}" for n, v in zip(names, vals)) for vals in zip(*(b[n] for n in names))])
        s = pd.DataFrame({"seg": label, "w": test["Exposure"], "y": test["ClaimNb"], "a": e_glm, "b": e_gbm}).groupby("seg").sum()
        rows.append(s[s["a"] >= min_expected])
    s = pd.concat(rows)
    s["ae_glm"], s["ae_gbm"] = s["y"] / s["a"], s["y"] / s["b"]
    s["z_glm"], s["z_gbm"] = (s["y"] - s["a"]) / np.sqrt(s["a"]), (s["y"] - s["b"]) / np.sqrt(s["b"])
    s["ml_closer"] = (s["ae_gbm"] - 1).abs() < (s["ae_glm"] - 1).abs()
    s = s.reset_index().rename(columns={"seg": "segment", "w": "exposure", "y": "claims"})
    top = lambda z: s.loc[s[z].abs().sort_values(ascending=False).index[:12]].to_dict("records")
    return {"cells": len(s), "ml_closer_share": s["ml_closer"].mean(),
            "flagged_glm": int((s["z_glm"].abs() > 2).sum()), "flagged_gbm": int((s["z_gbm"].abs() > 2).sum()),
            "flagged_by_chance": 0.0455 * len(s),   # share of a normal beyond 2 standard errors
            "top_glm_misses": top("z_glm"), "top_gbm_misses": top("z_gbm")}


# ---------------------------------------------------------------- explanations
def glm_effects(beta, X):
    """For an additive GLM the exact SHAP value of a factor is its term minus the term's average."""
    factor = {"Area": "Area", "BonusMalus": "BonusMalus", "logDensity": "Density", "Diesel": "VehGas"}
    out = {}
    for j, col in enumerate(glm.COLUMNS[1:], start=1):
        f = factor.get(col, col.split("=")[0])
        out[f] = out.get(f, 0) + X[:, j] * beta[j]
    return {f: v - v.mean() for f, v in out.items()}


def effect_curve(x, phi, grid):
    """Average contribution at each grid value (bins with fewer than 30 policies are left empty)."""
    k = np.clip(np.searchsorted(grid, x), 0, len(grid) - 1)
    n = np.bincount(k, minlength=len(grid))
    s = np.bincount(k, weights=phi, minlength=len(grid))
    return [s[i] / n[i] if n[i] >= 30 else None for i in range(len(grid))]


# ---------------------------------------------------------------- main
def fit_and_score(df, seed):
    train_i, test_i = data.split(df, seed)
    train, test = df.iloc[train_i].reset_index(drop=True), df.iloc[test_i].reset_index(drop=True)
    fit_i, valid_i = data.split(train, seed + 1, 0.15)
    g = glm.fit_frequency(train)
    b = gbm.fit(train.iloc[fit_i], train.iloc[valid_i])
    return train, test, g, b


def run(root: Path) -> dict:
    data.download(root / "data")
    df = data.load(root / "data")
    print(f"  {len(df):,} policies")
    train, test, g, b = fit_and_score(df, SEED)
    beta = g.params.to_numpy()
    print(f"  boosting stopped at {b.best_iteration} trees")

    w, y, loss = test["Exposure"].to_numpy(), test["ClaimNb"].to_numpy(), test["LossCap"].to_numpy()
    rate0 = train["ClaimNb"].sum() / train["Exposure"].sum()
    r_glm, r_gbm = np.exp(glm.design(test) @ beta), gbm.rate(b, test)
    d0, d1, d2 = poisson_dev(y, rate0 * w), poisson_dev(y, r_glm * w), poisson_dev(y, r_gbm * w)

    # pure premium = claim frequency x recorded cost per predicted claim (fitted on training data)
    cost = {m: train["LossCap"].sum() / (train["Exposure"] * r).sum()
            for m, r in (("glm", np.exp(glm.design(train) @ beta)), ("gbm", gbm.rate(b, train)))}
    pp_glm, pp_gbm = r_glm * cost["glm"], r_gbm * cost["gbm"]
    load = train["Loss"].sum() / train["LossCap"].sum()

    # severity: does a Gamma GLM beat the flat average claim size on held-out claims?
    s = glm.fit_severity(train)
    claimed = test["NAmt"] > 0
    ys, ns = (test["LossCap"] / test["NAmt"])[claimed], test["NAmt"][claimed]
    flat = train["LossCap"].sum() / train["NAmt"].sum()
    severity = {"claims_policies_test": int(claimed.sum()), "flat_mean": flat,
                "dev_flat": float((ns * gamma_dev(ys, flat)).sum() / ns.sum()),
                "dev_glm": float((ns * gamma_dev(ys, np.exp(glm.design(test[claimed]) @ s.params.to_numpy()))).sum() / ns.sum())}

    # point estimates and bootstrap intervals
    # premium = frequency x a constant, so one sort order per model serves both claim counts and claim cost
    o_glm, o_gbm = np.argsort(r_glm, kind="stable"), np.argsort(r_gbm, kind="stable")

    def stats(c):
        dg, db = (c * d1).sum(), (c * d2).sum()
        out = {"dev_null": (c * d0).sum() / len(y), "dev_glm": dg / len(y), "dev_gbm": db / len(y),
               "dev_gap": (dg - db) / len(y), "dev_cut_pct": 100 * (dg - db) / dg,
               "gini_glm": gini(o_glm, y, w, c), "gini_gbm": gini(o_gbm, y, w, c),
               "gini_pp_glm": gini(o_glm, loss, w, c), "gini_pp_gbm": gini(o_gbm, loss, w, c)}
        out["gini_gain"] = out["gini_gbm"] - out["gini_glm"]
        out["gini_pp_gain"] = out["gini_pp_gbm"] - out["gini_pp_glm"]
        return out

    point = stats(np.ones(len(y)))
    rng = np.random.default_rng(SEED)
    draws = pd.DataFrame([stats(np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y))) for _ in range(BOOT)])
    ci = {k: [draws[k].quantile(0.025), draws[k].quantile(0.975)] for k in draws}
    point["explained_glm_pct"] = 100 * (1 - point["dev_glm"] / point["dev_null"])   # share of no-model deviance removed
    point["explained_gbm_pct"] = 100 * (1 - point["dev_gbm"] / point["dev_null"])
    print(f"  deviance x100: none {100*point['dev_null']:.3f}, GLM {100*point['dev_glm']:.3f}, boosting {100*point['dev_gbm']:.3f}"
          f"; cut {point['dev_cut_pct']:.2f}% [{ci['dev_cut_pct'][0]:.2f}, {ci['dev_cut_pct'][1]:.2f}]")

    # charts: Lorenz, calibration, double lift
    grid = np.linspace(0, 1, 101)
    curves = {k: np.interp(grid, *lorenz(order, t, w)).tolist()
              for k, order, t in (("freq_glm", o_glm, y), ("freq_gbm", o_gbm, y), ("pp_glm", o_glm, loss), ("pp_gbm", o_gbm, loss))}
    calib = {}
    for m, r in (("glm", r_glm), ("gbm", r_gbm)):
        ew, ey, ee = by_tenth(tenths(r, w), w, y, r * w)
        calib[m] = {"actual": ey / ew, "pred": ee / ew, "claims": ey}
    ratio = r_gbm / r_glm
    ew, ey, e1, e2, el, ep1, ep2, er = by_tenth(tenths(ratio, w), w, y, r_glm * w, r_gbm * w, loss, pp_glm * w, pp_gbm * w, ratio * w)
    lift = {"ratio": er / ew, "claims": ey, "freq_actual": ey / ew, "freq_glm": e1 / ew, "freq_gbm": e2 / ew,
            "freq_lo": np.maximum(ey - 1.96 * np.sqrt(ey), 0) / ew, "freq_hi": (ey + 1.96 * np.sqrt(ey)) / ew,
            "ae_glm": ey / e1, "ae_gbm": ey / e2, "pp_actual": el / ew, "pp_glm": ep1 / ew, "pp_gbm": ep2 / ew}
    worst = {m: int(np.argmax(np.abs(c["actual"] / c["pred"] - 1))) for m, c in calib.items()}   # least calibrated tenth
    lift["ml_closer_tenths"] = int((np.abs(lift["ae_gbm"] - 1) < np.abs(lift["ae_glm"] - 1)).sum())

    # explanations: exact TreeSHAP (from LightGBM) and the GLM's own additive terms, on every test policy
    phi_gbm = b.predict(gbm.matrix(test), pred_contrib=True, num_iteration=b.best_iteration)[:, :-1]
    phi_glm = glm_effects(beta, glm.design(test))
    grids = {"DrivAge": np.arange(18, 91), "BonusMalus": np.arange(50, 151, 5), "VehAge": np.arange(0, 26),
             "VehPower": np.arange(4, 16), "Density": np.unique(np.round(np.logspace(0, np.log10(27000), 40))),
             "Area": np.arange(1, 7)}
    Xs = gbm.matrix(test)
    effects = {}
    for f, gr in grids.items():
        j = data.FACTORS.index(f)
        effects[f] = {"x": gr.tolist(), "glm": effect_curve(Xs[:, j], phi_glm[f], gr),
                      "gbm": effect_curve(Xs[:, j], phi_gbm[:, j], gr)}
    importance = {f: {"glm": float(np.abs(phi_glm[f]).mean()), "gbm": float(np.abs(phi_gbm[:, j]).mean())}
                  for j, f in enumerate(data.FACTORS)}

    # the same comparison on five other splits
    robust = []
    for sd in OTHER_SEEDS:
        tr, te, g2, b2 = fit_and_score(df, sd)
        mu1 = np.exp(glm.design(te) @ g2.params.to_numpy()) * te["Exposure"]
        mu2 = gbm.rate(b2, te) * te["Exposure"]
        dv1, dv2 = poisson_dev(te["ClaimNb"], mu1).mean(), poisson_dev(te["ClaimNb"], mu2).mean()
        robust.append({"seed": sd, "dev_glm": dv1, "dev_gbm": dv2, "dev_cut_pct": 100 * (dv1 - dv2) / dv1, "trees": b2.best_iteration})
        print(f"  split {sd}: cut {robust[-1]['dev_cut_pct']:.2f}%")

    results = {
        "seed": SEED, "bootstrap": BOOT,
        "data": {"policies": len(df), "train": len(train), "test": len(test), "policy_years": df["Exposure"].sum(),
                 "claims": df["ClaimNb"].sum(), "claim_rows": data.ROWS[41215], "claim_records": int(df["NAmt"].sum()),
                 "rating_profiles": int(df.groupby(data.FACTORS).ngroups),
                 "claiming_policies_without_amount": int(((df["ClaimNb"] > 0) & (df["NAmt"] == 0)).sum()),
                 "test_policy_years": w.sum(), "test_claims": y.sum(), "claim_cap": data.CLAIM_CAP,
                 "train_loss_share_above_cap": 1 - train["LossCap"].sum() / train["Loss"].sum()},   # same data as the load
        "trees": b.best_iteration, "point": point, "ci95": ci, "severity": severity,
        "pure_premium": {"cost_per_claim": cost, "large_loss_load": load, "actual_test": loss.sum() / w.sum(),
                         "glm_test": (pp_glm * w).sum() / w.sum(), "gbm_test": (pp_gbm * w).sum() / w.sum()},
        "lorenz": curves, "calibration": calib, "worst_tenth": worst, "double_lift": lift,
        "price_gap": {"quartiles": np.quantile(ratio, [0.25, 0.5, 0.75]), "share_over_20pct": w[np.abs(np.log(ratio)) > np.log(1.2)].sum() / w.sum()},
        "segments": segments(test, r_glm * w, r_gbm * w), "effects": effects, "importance": importance,
        "robustness": robust,
        "glm_coefficients": [{"name": n, "coef": c, "se": e} for n, c, e in zip(glm.COLUMNS, beta, g.bse.to_numpy())],
    }
    model = {"glm": dict(zip(glm.COLUMNS, beta)), "driv_edges": glm.DRIV_EDGES, "driv_labels": glm.DRIV_LABELS,
             "areas": glm.AREAS, "brands": gbm.BRANDS, "regions": gbm.REGIONS, "factors": data.FACTORS,
             "trees": gbm.export(b), "cost_per_claim": cost, "large_loss_load": load, "typical": TYPICAL}
    cases = test.sample(2000, random_state=SEED + 1)
    parity = [{"profile": prof, "glm": float(a), "gbm": float(c)} for prof, a, c in
              zip(cases[data.FACTORS].to_dict("records"), np.exp(glm.design(cases) @ beta), gbm.rate(b, cases))]

    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [clean(x) for x in v]
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return float(f"{v:.8g}")
        return v

    (root / "out").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "pricing.json").write_text(json.dumps(clean(results), separators=(",", ":")))
    (root / "results" / "pricing.model.json").write_text(json.dumps(clean(model), separators=(",", ":")))
    (root / "out" / "parity_cases.json").write_text(json.dumps(parity))
    return results

// Pricing functions for a web page: GLM and gradient-boosting claim frequency.
// The block below is copied verbatim into site/js/pricing.js; tests/parity.mjs fails if the two differ.
// Author: Vivek Mitra

// ---------------------------------------------------------------- the models (kept identical to web/pricing_model.js in the code repository; its parity test checks)

/** GLM: add up the coefficient of every column that applies to this driver, then exp(). */
export function glmRate(m, p) {
  const driv = m.driv_labels[m.driv_edges.filter(e => p.DrivAge >= e).length - 1];
  const vehAge = p.VehAge < 1 ? "0" : p.VehAge <= 10 ? "1-10" : "11+";
  const x = {
    Intercept: 1,
    Area: m.areas.indexOf(p.Area) + 1,
    [`VehPower=${Math.min(p.VehPower, 9)}`]: 1,
    [`VehAge=${vehAge}`]: 1,
    [`DrivAge=${driv}`]: 1,
    BonusMalus: Math.min(p.BonusMalus, 150),
    logDensity: Math.log(p.Density),
    [`VehBrand=${p.VehBrand}`]: 1,
    Diesel: p.VehGas === "Diesel" ? 1 : 0,
    [`Region=${p.Region}`]: 1,
  };
  let s = 0;
  for (const [col, v] of Object.entries(x)) s += (m.glm[col] ?? 0) * v;   // base levels have no coefficient
  return Math.exp(s);                                                       // claims per policy-year
}

/** Boosting: walk each tree from its root to a leaf, add up the leaf values, then exp(). */
export function gbmRate(m, p) {
  const x = m.factors.map(f =>
    f === "Area" ? m.areas.indexOf(p.Area) + 1 :
    f === "VehBrand" ? m.brands.indexOf(p.VehBrand) :
    f === "Region" ? m.regions.indexOf(p.Region) :
    f === "VehGas" ? (p.VehGas === "Diesel" ? 1 : 0) : p[f]);
  let s = 0;
  for (const t of m.trees) {
    let n = t.f.length ? 0 : -1;                       // a one-leaf tree has no splits
    while (n >= 0) {
      const v = x[t.f[n]], thr = t.t[n];
      n = (Array.isArray(thr) ? thr.includes(v) : v <= thr) ? t.l[n] : t.r[n];
    }
    s += t.v[-1 - n];
  }
  return Math.exp(s);
}

/** Yearly pure premium = claim frequency x recorded cost per claim x large-loss load. */
export const premium = (m, model, rate) => rate * m.cost_per_claim[model] * m.large_loss_load;

/** Reasons: how much the price changes when one factor is moved from the typical driver to this driver. */
export function reasons(m, p, rateFn) {
  const r = rateFn(m, p);
  return m.factors.map(f => ({ f, x: r / rateFn(m, { ...p, [f]: m.typical[f] }) }))
    .filter(d => Math.abs(Math.log(d.x)) > 0.005)
    .sort((a, b) => Math.abs(Math.log(b.x)) - Math.abs(Math.log(a.x)));
}

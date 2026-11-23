// The browser pricing functions (web/pricing_model.js) must give the same prices as Python.
// Checks 2,000 held-out policies, both models, and writes results/pricing.parity.json.
// If the website folder is next to this repository, its copy of the functions must match exactly.
// Author: Vivek Mitra
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const web = path.join(root, "web", "pricing_model.js");
const { glmRate, gbmRate } = await import(pathToFileURL(web).href);
const model = JSON.parse(readFileSync(path.join(root, "results", "pricing.model.json"), "utf8"));
const cases = JSON.parse(readFileSync(path.join(root, "out", "parity_cases.json"), "utf8"));

let worst = 0;
for (const c of cases) {
  worst = Math.max(worst, Math.abs(glmRate(model, c.profile) / c.glm - 1), Math.abs(gbmRate(model, c.profile) / c.gbm - 1));
}
const report = { cases: cases.length, max_rel_diff: worst, tolerance: 1e-6 };
writeFileSync(path.join(root, "results", "pricing.parity.json"), JSON.stringify(report));
console.log(report);
if (!(worst < 1e-6)) { console.error("browser and Python prices differ"); process.exit(1); }

// the website keeps a verbatim copy of the model block (from its marker line to the end of web/pricing_model.js)
const site = path.resolve(root, "..", "..", "site", "js", "pricing.js");
if (existsSync(site)) {
  const text = f => readFileSync(f, "utf8").replace(/\r\n/g, "\n");
  const mine = text(web), block = mine.slice(mine.indexOf("// ------"));
  if (!text(site).includes(block)) { console.error("site/js/pricing.js differs from web/pricing_model.js"); process.exit(1); }
  console.log("site/js/pricing.js has the same pricing functions");
}

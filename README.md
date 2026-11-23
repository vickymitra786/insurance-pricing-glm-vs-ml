# Does machine learning beat the actuarial GLM?

An honest benchmark on French motor insurance. The standard actuarial model for claim frequency, a
Poisson GLM, against gradient boosting with a Poisson objective. Both are fitted on 80% of 678,013 real
policies and scored on held-out policies, with bootstrap intervals and five repeat splits.

Live demo: [vivekmitra.com/pricing.html](https://vivekmitra.com/pricing.html) ·
Paper: [results/pricing.pdf](results/pricing.pdf) (also at [vivekmitra.com/papers/pricing.pdf](https://vivekmitra.com/papers/pricing.pdf))

## Result

On 136,090 held-out policies (seed 2026), with policies that share identical rating factors kept on the same side of the split:

| Metric (held-out test set) | No model | Poisson GLM | Gradient boosting | Boosting vs GLM, 95% interval |
|---|---|---|---|---|
| Poisson deviance ×10⁻² (lower is better) | 32.911 | 31.317 | 30.223 | 3.49% lower [3.13%, 3.82%] |
| Gini, claim counts | 0 | 0.275 | 0.332 | +0.057 [0.049, 0.066] |
| Gini, capped claim cost | 0 | 0.269 | 0.305 | +0.036 [0.009, 0.060] |
| Double lift: price-gap groups where the model was closer to actual claims | | 0 of 10 | 10 of 10 | |
| Five more random splits: deviance cut | | | | 3.18% to 3.68%; boosting lower on 5 of 5 |

Boosting beats the GLM by a clear but modest margin, so it is kept as the challenger. A Gamma GLM for
claim size did **not** beat a flat average on held-out claims (Gamma deviance 1.3153 vs 1.3120), so both
prices use the flat average cost per claim. The numbers above are copied from `results/pricing.json`,
which `run_all.py` writes.

## Run it

Needs Python 3.13, Node 18+ and, for the PDF only, Microsoft Word on Windows.

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

`run_all.py` downloads the data, fits and scores both models, runs the three tests, draws the figures and builds the paper.
Everything it writes stays inside this folder: `results/` (committed) and `out/` (working files, not committed).
Expect about 10 minutes. Seeds are fixed, so every number is the same on every run.

## Data

`freMTPL2freq` (678,013 policies) and `freMTPL2sev` (26,639 claim amounts) from the CASdatasets R package
(C. Dutang and A. Charpentier, <https://github.com/dutangc/CASdatasets>), the companion to A. Charpentier (ed.),
*Computational Actuarial Science with R*, CRC Press, 2014. `run_all.py` downloads them from OpenML,
datasets [41214](https://www.openml.org/d/41214) and [41215](https://www.openml.org/d/41215), into `data/`.
To fetch them by hand, save these two files into `data/`:

- <https://data.openml.org/datasets/0004/41214/dataset_41214.pq>
- <https://data.openml.org/datasets/0004/41215/dataset_41215.pq>

## What is in here

| File | What it does |
|---|---|
| `src/data.py` | Download, clean (claims capped at 4, exposure at one year, single claims at €100k) and split by rating profile |
| `src/glm.py` | The Poisson GLM with the banding of Noll, Salzmann and Wüthrich (2018); the Gamma severity GLM |
| `src/gbm.py` | LightGBM with a Poisson objective and log(exposure) offset; exports the trees to JSON |
| `src/evaluate.py` | Test-set metrics, bootstrap intervals, Lorenz, calibration, double lift, segments, SHAP, repeat splits |
| `src/figures.py` | Paper figures |
| `web/pricing_model.js` | The two pricing functions the web page runs (GLM and trees), as plain JavaScript |
| `results/` | `pricing.json` (every published number), `pricing.model.json` (coefficients and trees), `pricing.parity.json`, `pricing.pdf` |
| `tests/test_glm.py` | The GLM must match an independent statsmodels formula fit |
| `tests/parity.mjs` | The browser code must give the same prices as Python (2,000 held-out policies) |
| `tests/check_paper.py` | Every number in the paper must trace to the results file; headline claims must match it |
| `paper/` | Builds the Word paper from the results and converts it to PDF |

## Licence

Copyright (C) 2026 Vivek Mitra. Licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).

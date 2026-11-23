"""The hand-built GLM design must give the same fit as statsmodels' own formula interface.

The formula fit builds its dummy variables independently, so matching deviance and predictions
shows the banding, base levels and offset in src/glm.py are right.
Author: Vivek Mitra
"""
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import glm   # noqa: E402

df = data.load(ROOT / "data").sample(100_000, random_state=1).reset_index(drop=True)
ours = glm.fit_frequency(df)

f = df.assign(AreaN=df["Area"].map({a: i + 1 for i, a in enumerate(glm.AREAS)}),
              Power=np.minimum(df["VehPower"], 9), VehAgeB=glm.vehage_band(df["VehAge"].to_numpy()),
              DrivB=glm.driv_band(df["DrivAge"].to_numpy()), BM=np.minimum(df["BonusMalus"], 150),
              logD=np.log(df["Density"]))
ref = smf.glm("ClaimNb ~ AreaN + C(Power) + C(VehAgeB, Treatment('1-10')) + C(DrivB, Treatment('41-50')) + BM + logD"
              " + C(VehBrand, Treatment('B1')) + C(VehGas, Treatment('Regular')) + C(Region, Treatment('R24'))",
              data=f, family=sm.families.Poisson(), offset=np.log(f["Exposure"])).fit()

assert abs(ours.deviance - ref.deviance) < 1e-6 * ref.deviance, (ours.deviance, ref.deviance)
assert len(ours.params) == len(ref.params) == len(glm.COLUMNS)
gap = np.max(np.abs(ours.fittedvalues.to_numpy() / ref.fittedvalues.to_numpy() - 1))
assert gap < 1e-8, gap
print(f"GLM test passed: deviance {ours.deviance:.4f} vs {ref.deviance:.4f}, largest prediction gap {gap:.1e}")

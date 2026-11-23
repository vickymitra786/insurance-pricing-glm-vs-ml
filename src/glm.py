"""The actuarial baseline: Poisson GLM for claim frequency, log link, log(exposure) offset.

Banding follows Noll, Salzmann and Wuthrich (2018), the standard treatment of this data set.
site/js/pricing.js rebuilds the same design matrix, column by column, from the exported coefficients.
Author: Vivek Mitra
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

AREAS = list("ABCDEF")                                     # used as 1..6, one slope
POWERS = [5, 6, 7, 8, 9]                                   # power capped at 9; 4 is the base level
VEHAGES = ["0", "11+"]                                     # base band 1-10 years
DRIV_EDGES = [18, 21, 26, 31, 41, 51, 71]
DRIV_LABELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]   # base band 41-50
BRANDS = ["B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]    # base brand B1
REGIONS = ["R11", "R21", "R22", "R23", "R25", "R26", "R31", "R41", "R42", "R43", "R52", "R53",
           "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"]     # base region R24 (largest)

COLUMNS = (["Intercept", "Area"] + [f"VehPower={p}" for p in POWERS] + [f"VehAge={b}" for b in VEHAGES]
           + [f"DrivAge={b}" for b in DRIV_LABELS if b != "41-50"] + ["BonusMalus", "logDensity"]
           + [f"VehBrand={b}" for b in BRANDS] + ["Diesel"] + [f"Region={r}" for r in REGIONS])


def driv_band(age):
    return np.array(DRIV_LABELS)[np.searchsorted(DRIV_EDGES, age, side="right") - 1]


def vehage_band(age):
    return np.where(age < 1, "0", np.where(age <= 10, "1-10", "11+"))


def design(df: pd.DataFrame) -> np.ndarray:
    """One row per policy, one column per entry in COLUMNS."""
    power = np.minimum(df["VehPower"].to_numpy(), 9)
    vehage = vehage_band(df["VehAge"].to_numpy())
    driv = driv_band(df["DrivAge"].to_numpy())
    cols = [np.ones(len(df)), df["Area"].map({a: i + 1 for i, a in enumerate(AREAS)}).to_numpy()]
    cols += [power == p for p in POWERS]
    cols += [vehage == b for b in VEHAGES]
    cols += [driv == b for b in DRIV_LABELS if b != "41-50"]
    cols += [np.minimum(df["BonusMalus"].to_numpy(), 150), np.log(df["Density"].to_numpy())]
    cols += [df["VehBrand"].to_numpy() == b for b in BRANDS]
    cols += [df["VehGas"].to_numpy() == "Diesel"]
    cols += [df["Region"].to_numpy() == r for r in REGIONS]
    return np.column_stack(cols).astype(float)


def fit_frequency(df: pd.DataFrame):
    return sm.GLM(df["ClaimNb"], design(df), family=sm.families.Poisson(),
                  offset=np.log(df["Exposure"])).fit()


def fit_severity(df: pd.DataFrame):
    """Gamma GLM on average capped claim size per claiming policy, weighted by its number of claims."""
    c = df[df["NAmt"] > 0]
    return sm.GLM(c["LossCap"] / c["NAmt"], design(c), family=sm.families.Gamma(sm.families.links.Log()),
                  var_weights=c["NAmt"]).fit(scale="X2")

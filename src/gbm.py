"""The challenger: gradient-boosted trees (LightGBM) with a Poisson objective and log(exposure) offset.

Trees see the nine raw rating factors; no banding. They are exported to JSON so the web page can
evaluate them with a few lines of JavaScript.
Author: Vivek Mitra
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from data import FACTORS
from glm import AREAS

BRANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B10", "B11", "B12", "B13", "B14"]
REGIONS = ["R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31", "R41", "R42", "R43",
           "R52", "R53", "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"]
PARAMS = dict(objective="poisson", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              deterministic=True, num_threads=8, seed=2026, verbose=-1)


def matrix(df: pd.DataFrame) -> np.ndarray:
    """Factors as numbers: Area 1-6, brand and region as category codes, diesel 0/1, the rest as is."""
    X = df[FACTORS].copy()
    X["Area"] = X["Area"].map({a: i + 1 for i, a in enumerate(AREAS)})
    X["VehBrand"] = X["VehBrand"].map({b: i for i, b in enumerate(BRANDS)})
    X["Region"] = X["Region"].map({r: i for i, r in enumerate(REGIONS)})
    X["VehGas"] = (X["VehGas"] == "Diesel").astype(int)
    return X.to_numpy(float)


def fit(train: pd.DataFrame, valid: pd.DataFrame) -> lgb.Booster:
    """Add trees until the Poisson loss on the validation slice stops improving for 100 rounds."""
    dtrain = lgb.Dataset(matrix(train), train["ClaimNb"], init_score=np.log(train["Exposure"]),
                         feature_name=FACTORS, categorical_feature=["VehBrand", "Region"])
    dvalid = lgb.Dataset(matrix(valid), valid["ClaimNb"], init_score=np.log(valid["Exposure"]), reference=dtrain)
    return lgb.train(PARAMS, dtrain, num_boost_round=5000, valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def rate(model: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    """Predicted claims per policy-year (the offset is not part of the trees)."""
    return np.exp(model.predict(matrix(df), raw_score=True, num_iteration=model.best_iteration))


def export(model: lgb.Booster) -> list:
    """Each tree as arrays over its split nodes: feature, threshold, left child, right child; plus leaf values.

    A child index >= 0 is another split node; a negative index -1-k points at leaf k.
    Numeric split: go left when x <= threshold (all inputs are whole numbers, so floor(threshold) works).
    Category split: the threshold is the list of codes that go left.
    """
    trees = []
    for info in model.dump_model(num_iteration=model.best_iteration)["tree_info"]:
        t = {"f": [], "t": [], "l": [], "r": [], "v": []}

        def walk(node):
            if "leaf_value" in node:
                t["v"].append(round(node["leaf_value"], 10))
                return -len(t["v"])
            i = len(t["f"])
            t["f"].append(node["split_feature"])
            if node["decision_type"] == "==":
                t["t"].append([int(c) for c in str(node["threshold"]).split("||")])
            else:
                t["t"].append(int(np.floor(node["threshold"])))
            t["l"].append(None)
            t["r"].append(None)
            t["l"][i] = walk(node["left_child"])
            t["r"][i] = walk(node["right_child"])
            return i

        walk(info["tree_structure"])
        trees.append(t)
    return trees

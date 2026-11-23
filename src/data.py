"""Download, clean and split the French motor third-party liability data (freMTPL2freq, freMTPL2sev).

Source: CASdatasets (Dutang and Charpentier), mirrored on OpenML as datasets 41214 and 41215.
Author: Vivek Mitra
"""
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URLS = {41214: "https://data.openml.org/datasets/0004/41214/dataset_41214.pq",   # freMTPL2freq, 678,013 policies
        41215: "https://data.openml.org/datasets/0004/41215/dataset_41215.pq"}   # freMTPL2sev, 26,639 claims
ROWS = {41214: 678013, 41215: 26639}
CLAIM_CAP = 100_000.0       # single claims above this are capped; the excess becomes a flat large-loss load
FACTORS = ["DrivAge", "BonusMalus", "VehAge", "VehPower", "Density", "Area", "VehBrand", "VehGas", "Region"]


def download(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for oid, url in URLS.items():
        path = folder / f"dataset_{oid}.pq"
        if not path.exists():
            print(f"  downloading OpenML {oid}")
            path.write_bytes(requests.get(url, timeout=600).content)
        if len(pd.read_parquet(path)) != ROWS[oid]:
            raise RuntimeError(f"{path} does not have {ROWS[oid]} rows; delete it and run again")


def load(folder: Path) -> pd.DataFrame:
    """One row per policy: claim count, exposure (years), the nine rating factors and claim cost."""
    df = pd.read_parquet(folder / "dataset_41214.pq")
    sev = pd.read_parquet(folder / "dataset_41215.pq")
    sev["Capped"] = sev["ClaimAmount"].clip(upper=CLAIM_CAP)
    cost = sev.groupby("IDpol").agg(NAmt=("ClaimAmount", "size"), Loss=("ClaimAmount", "sum"), LossCap=("Capped", "sum"))
    df = df.merge(cost, left_on="IDpol", right_index=True, how="left").fillna({"NAmt": 0, "Loss": 0.0, "LossCap": 0.0})
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4).astype(float)      # 9 policies report 5-16 claims; standard cap
    df["Exposure"] = df["Exposure"].clip(upper=1.0)                # a few records exceed one year
    for c in ["DrivAge", "BonusMalus", "VehAge", "VehPower", "Density"]:
        df[c] = df[c].astype(int)
    for c in ["Area", "VehBrand", "VehGas", "Region"]:
        df[c] = df[c].astype(str)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int, frac: float = 0.2):
    """Seeded split that keeps every policy with the same nine rating factors on the same side.

    The data hold many rows with identical factors (often the same car split across periods); a plain
    random split would let a flexible model see near-copies of its test rows during training.
    """
    group = df.groupby(FACTORS, sort=False).ngroup().to_numpy()
    rng = np.random.default_rng(seed)
    held = rng.random(group.max() + 1) < frac
    test = held[group]
    return np.flatnonzero(~test), np.flatnonzero(test)

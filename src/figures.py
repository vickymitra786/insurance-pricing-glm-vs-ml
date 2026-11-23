"""Paper figures, drawn only from results/pricing.json.
Author: Vivek Mitra
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

GLM, GBM, INK, GREY = "#1f4e8c", "#d97a1e", "#18202b", "#8a93a0"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 200})


def make(results: Path, folder: Path) -> None:
    R = json.loads(results.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    tenth = np.arange(1, 11)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    x = np.linspace(0, 1, 101)
    for ax, k, title, g in ((axes[0], "freq", "Claim counts", ""), (axes[1], "pp", "Claim costs (capped)", "pp_")):
        ax.plot(x, x, color=GREY, lw=0.8, ls="--", label="No rating")
        ax.plot(x, R["lorenz"][k + "_glm"], color=GLM, label=f"GLM, Gini {R['point'][f'gini_{g}glm']:.3f}")
        ax.plot(x, R["lorenz"][k + "_gbm"], color=GBM, label=f"Boosting, Gini {R['point'][f'gini_{g}gbm']:.3f}")
        ax.set(title=title, xlabel="Share of policy-years, lowest predicted risk first")
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("Share of losses")
    fig.tight_layout(); fig.savefig(folder / "lorenz.png"); plt.close(fig)

    L = R["double_lift"]
    a = np.array(L["freq_actual"]) * 100
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.errorbar(tenth, a, yerr=[a - np.array(L["freq_lo"]) * 100, np.array(L["freq_hi"]) * 100 - a],
                fmt="o", color=INK, ms=4, capsize=2, lw=0.8, label="Actual (95% Poisson interval)")
    ax.plot(tenth, np.array(L["freq_glm"]) * 100, color=GLM, marker="s", ms=3, label="GLM")
    ax.plot(tenth, np.array(L["freq_gbm"]) * 100, color=GBM, marker="^", ms=3, label="Boosting")
    ax.set_xticks(tenth, [f"{v:.2f}" for v in L["ratio"]], fontsize=7.5)
    ax.set(xlabel="Tenth of exposure by boosting ÷ GLM predicted frequency (mean ratio shown)", ylabel="Claims per 100 policy-years")
    ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(folder / "double_lift.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for ax, k, title, col in ((axes[0], "glm", "GLM", GLM), (axes[1], "gbm", "Boosting", GBM)):
        c = R["calibration"][k]
        ax.plot(tenth, np.array(c["pred"]) * 100, color=col, marker="s", ms=3, label="Predicted")
        ax.plot(tenth, np.array(c["actual"]) * 100, "o", color=INK, ms=4, label="Actual")
        ax.set(title=title, xlabel="Tenth of exposure by predicted frequency", xticks=tenth)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("Claims per 100 policy-years")
    fig.tight_layout(); fig.savefig(folder / "calibration.png"); plt.close(fig)

    names = {"DrivAge": "Driver age", "BonusMalus": "Bonus-malus level", "VehAge": "Vehicle age (years)",
             "VehPower": "Vehicle power class", "Density": "Density (people per km², log axis)", "Area": "Area (A=1 … F=6)"}
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.4))
    for ax, (f, title) in zip(axes.flat, names.items()):
        e = R["effects"][f]
        for key, col, lab in (("glm", GLM, "GLM"), ("gbm", GBM, "Boosting")):
            y = np.exp(np.array([np.nan if v is None else v for v in e[key]], float))
            ax.plot(e["x"], y, color=col, lw=1.4, label=lab)
        ax.axhline(1, color=GREY, lw=0.6, ls=":")
        if f == "Density":
            ax.set_xscale("log")
        ax.set_title(title, fontsize=8.5)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Frequency multiplier")
    fig.tight_layout(); fig.savefig(folder / "effects.png"); plt.close(fig)
    print(f"  figures in {folder}")

"""Draw the README and paper figures from data/figures_data/*.csv. Writes PNG, PDF and SVG to figures/.
python scripts/make_figures.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data" / "figures_data", ROOT / "figures"
OKABE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
NAMES = {"Qwen2.5-0.5B-Instruct": "Qwen 0.5B", "Qwen2.5-1.5B-Instruct": "Qwen 1.5B", "SmolLM2-1.7B-Instruct": "SmolLM2",
         "OLMo-2-0425-1B-Instruct": "OLMo 1B", "Llama-3.2-3B": "Llama 3B"}
HALO = [patheffects.withStroke(linewidth=3.0, foreground="white", alpha=0.8)]
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "no-latex"])
except ImportError:
    pass
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "savefig.dpi": 300, "figure.dpi": 150, "savefig.facecolor": "white"})


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", metadata={"Creator": None, "Software": None} if ext == "png" else None)
    plt.close(fig)


def rival_dose():
    d = pd.read_csv(DATA / "dose_response.csv")
    d = d[(d.L == 2048)].copy()
    d["x"] = d.n_conf.replace("6+", "6").astype(int)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for (m, name), c in zip(NAMES.items(), OKABE):
        g = d[d.model == m].sort_values("x")
        ax.fill_between(g.x, g.ci_lo, g.ci_hi, color=c, alpha=0.2, lw=0)
        ax.plot(g.x, g.mean_loss, color=c, lw=1.4, marker="o", ms=2.5, label=name, path_effects=HALO)
    ax.set_xticks(range(7), [0, 1, 2, 3, 4, 5, "6+"])
    ax.set_xlabel("Same-key rival statements in context")
    ax.set_ylabel("Loss (nats)")
    ax.set_title("One rival answer is enough (2048 tokens)")
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "rival_dose_response")


def attention():
    d = pd.read_csv(DATA / "threshold.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for (m, name), c in zip(NAMES.items(), OKABE):
        g = d[d.model == m].sort_values("median_astar")
        ax.fill_between(g.median_astar, g.ci_lo, g.ci_hi, color=c, alpha=0.2, lw=0)
        ax.plot(g.median_astar, g.mean_loss, color=c, lw=1.4, label=name, path_effects=HALO)
    ax.set_xlabel("Attention mass on target, a* (bin median)")
    ax.set_ylabel("Loss (nats)")
    ax.set_title("Loss rises as attention leaves the target")
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "attention_vs_loss")


def r2():
    d = pd.read_csv(DATA / "r2_bootstrap.csv").set_index("model").loc[list(NAMES)]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.barh(y + 0.19, d.r2_neglog_astar, 0.36, color=OKABE[0], xerr=[d.r2_neglog_astar - d.a_lo, d.a_hi - d.r2_neglog_astar],
            error_kw={"lw": 0.8}, label="log(1/a*)")
    ax.barh(y - 0.19, d.r2_logL, 0.36, color=OKABE[1], xerr=[d.r2_logL - d.l_lo, d.l_hi - d.r2_logL], error_kw={"lw": 0.8}, label="log length")
    ax.set_yticks(y, [NAMES[m] for m in d.index])
    ax.invert_yaxis()
    ax.set_xlabel("Variance of loss explained, R²")
    ax.set_title("Attention beats length as a predictor")
    ax.legend(frameon=False, fontsize=6.5)
    save(fig, "r2_attention_vs_length")


def masking():
    d = pd.read_csv(DATA / "appendix_neutral.csv")
    d = d[d.m == "Qwen2.5-1.5B-Instruct"].groupby("L")[["visible_acc", "masked_acc"]].mean() * 100
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(d.index, d.visible_acc, color=OKABE[0], lw=1.4, marker="o", ms=2.5, label="Neutral filler, visible", path_effects=HALO)
    ax.plot(d.index, d.masked_acc, color=OKABE[1], lw=1.4, marker="o", ms=2.5, label="Same filler, masked", path_effects=HALO)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Filler length (tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Masks are not a clean length baseline (Qwen 1.5B)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    save(fig, "mask_vs_visible")


if __name__ == "__main__":
    rival_dose(); attention(); r2(); masking()

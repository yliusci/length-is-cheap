"""C1 map (v3): mean +/- standard error of d_dist, d_sim, d_gap, d_tot per model x filler x L, all instances
(Mac runs/ctxrot_v3_*, laptop runs_laptop/**/ctxrot_v3_*). Only explore/replicate/control measurements (not exploit,
which oversamples the veins). Output: findings/c1_table.csv + findings/c1_map.{png,pdf}.
uv run python harness/analyze_c1.py"""
import glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rows = []
for f in glob.glob(str(ROOT / "runs/ctxrot_v3_*/results.jsonl")) + glob.glob(str(ROOT / "runs_laptop/**/ctxrot_v3_*/results.jsonl"), recursive=True):
    src = "laptop" if "runs_laptop" in f else "mac"
    for line in open(f):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error") or r["kind"] == "exploit":
            continue
        m, p = r["metrics"], r["params"]
        filler = "repeat_sentence" if r["kind"] == "control" else p["filler"]
        rows.append({"src": src, "model": p["model"].split("/")[1], "filler": filler, "L": p["L"],
                     "L_eff": m.get("L_eff", p["L"]), "depth": p["depth"], "nd": p["n_distractors"], "seed": r["seed"],
                     **{k: m[k] for k in ("d_dist", "d_sim", "d_gap", "d_tot")}})
df = pd.DataFrame(rows)
out = ROOT / "findings"; out.mkdir(exist_ok=True)
df.to_csv(out / "c1_raw.csv", index=False)
g = df.groupby(["model", "filler", "L"])[["d_dist", "d_sim", "d_gap", "d_tot"]]
tab = g.mean().round(3).join(g.sem().round(3), rsuffix="_se").join(g.size().rename("n"))
tab.to_csv(out / "c1_table.csv")
pd.set_option("display.width", 200, "display.max_rows", 400)
print(f"{len(df)} measurements, {df.model.nunique()} models\n")
print(tab[["n", "d_dist", "d_dist_se", "d_sim", "d_sim_se", "d_gap", "d_gap_se"]].to_string())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fills = ["near_distractors", "random_words", "repeat_sentence", "numbers"]
models = sorted(df.model.unique())
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharex=True)
for ax, comp in zip(axes, ("d_dist", "d_sim", "d_gap")):
    for mdl in models:
        s = df[(df.model == mdl) & (df.filler == "near_distractors")].groupby("L")[comp]
        mu, se = s.mean(), s.sem().fillna(0)
        if len(mu):
            ax.errorbar(mu.index, mu.values, yerr=se.values, marker="o", ms=3, capsize=2, label=mdl)
    ax.set_xscale("log", base=2); ax.axhline(0, color="0.6", lw=0.8)
    ax.set_title({"d_dist": "distance  (L_short - L_neutral)", "d_sim": "similarity  (L_neutral - L_long)",
                  "d_gap": "mask gap  (L_neutral - L_mask)"}[comp], fontsize=10)
    ax.set_xlabel("filler tokens N")
axes[0].set_ylabel("log-prob loss (nats), near-miss filler")
axes[2].legend(fontsize=7, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(out / f"c1_map.{ext}", dpi=200)
print("\nfigure:", out / "c1_map.png")

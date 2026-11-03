"""Fit C2: is the loss d_sim predicted by the needle attention mass a* and by log N?
Compares mean field (loss proportional to log N) vs extreme values, and tests d_sim ~ -log(a*) (removal of the needle from the softmax).
uv run python harness/fit_c2.py"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
rows = []
for f in glob.glob(str(ROOT / "runs/c2/*.jsonl")) + glob.glob(str(ROOT / "runs_laptop/**/c2/*.jsonl"), recursive=True):
    for l in open(f):
        r = json.loads(l)
        if "error" not in r:
            rows.append(r)
df = pd.DataFrame(rows)
if df.empty:
    print("no C2 data yet"); raise SystemExit
df["logN"] = np.log(df["n_fill"].clip(lower=1))
df["neg_log_astar"] = -np.log(df["a_star_mean"].clip(lower=1e-6))
if "a_star_short" in df:
    d2 = df.dropna(subset=["a_star_short"])
    d2 = d2[d2.a_star_short > 1e-6]
    df["dneglog"] = np.nan
    df.loc[d2.index, "dneglog"] = np.log(d2["a_star_short"].clip(lower=1e-6)) - np.log(d2["a_star_mean"].clip(lower=1e-6))
print(f"{len(df)} C2 measurements, models {sorted(df.model.unique())}\n")
for mdl, g in df.groupby("model"):
    if len(g) < 20:
        print(mdl, f"({len(g)} measurements, too few)"); continue
    def r2(x):
        c = np.corrcoef(x, g["d_sim"])[0, 1]; return c * c
    print(f"{mdl.split('/')[-1]:28s} n={len(g):4d}  R²(logN)={r2(g['logN']):.2f}  "
          f"R²(-log a*)={r2(g['neg_log_astar']):.2f}  R²(a*)={r2(g['a_star_mean']):.2f}  "
          f"corr(d_sim,a*)={np.corrcoef(g['a_star_mean'], g['d_sim'])[0,1]:+.2f}", end="")
    gd = g.dropna(subset=["dneglog"]) if "dneglog" in g.columns else g.iloc[:0]
    if len(gd) > 20:
        print(f"  R²(Δ-log a*)={np.corrcoef(gd['dneglog'], gd['d_sim'])[0,1]**2:.2f} (n={len(gd)})")
    else:
        print()

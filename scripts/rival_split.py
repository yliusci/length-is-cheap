"""Rebuild each trial's filler from its stored seed and count same-key rival statements (no model is loaded).

The filler generator used for the v3 and C2 data did not exclude the needle key, so a filler sentence restates the queried
key with probability 1/51. This script recovers that count per trial and writes the aggregated dose_response.csv.
python scripts/rival_split.py <path to runs/c2>  (needs the raw data from the Hugging Face dataset)"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from ctxrot_v3 import WORDS  # noqa: E402

N_PER_CELL = 1500
_tok = {}


def rivals(model, seed, L):
    tok = _tok.setdefault(model, AutoTokenizer.from_pretrained(model))
    rng = np.random.default_rng(seed)
    key = str(rng.choice(WORDS))
    f = np.random.default_rng([seed, 1])
    text = " ".join(f"The secret code of the {f.choice(WORDS)} is {f.integers(1000, 9999)}." for _ in range(L // 12 + 1))
    kept = tok.decode(tok(text, add_special_tokens=False)["input_ids"][:L])
    return sum(k == key for k in re.findall(r"secret code of the (\w+) is \d+", kept))


def ci(x, rng, B=1000):
    x = np.asarray(x)
    if len(x) < 2:
        return np.nan, np.nan
    b = x[rng.integers(len(x), size=(B, len(x)))].mean(1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


if __name__ == "__main__":
    rows = [r for f in glob.glob(os.path.join(sys.argv[1], "*.jsonl")) for r in map(json.loads, open(f)) if "error" not in r]
    df = pd.DataFrame(rows)
    s = pd.concat(g.sample(min(len(g), N_PER_CELL), random_state=0) for _, g in df.groupby(["model", "L"]))
    s["n_rival"] = [rivals(m, sd, L) for m, sd, L in zip(s.model, s.seed, s.L)]
    s["bin"] = s.n_rival.clip(upper=6)
    rng = np.random.default_rng(0)
    out = []
    for (m, L, c), g in s.groupby(["model", "L", "bin"]):
        lo, hi = ci(g.d_sim.values, rng)
        out.append({"model": m.split("/")[1], "L": L, "n_conf": "6+" if c == 6 else str(c), "n": len(g),
                    "mean_loss": round(g.d_sim.mean(), 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    pd.DataFrame(out).to_csv("dose_response.csv", index=False)

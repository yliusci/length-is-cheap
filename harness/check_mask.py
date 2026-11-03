"""Test: with the filler masked, the log-prob must not depend on the CONTENT of the filler (identical positions).
uv run python harness/check_mask.py [model]"""
import sys
sys.path.insert(0, "experiments")
import ctxrot_v3 as E

m = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
worst = 0.0
for seed in range(5):
    for L in (512, 2048):
        p = {"model": m, "L": L, "depth": 0.4, "filler": "near_distractors", "n_distractors": 0}
        vals = {f: E._measure(p, seed, True, False, filler=f)[0] for f in ("near_distractors", "random_words", "numbers")}
        spread = max(vals.values()) - min(vals.values())
        worst = max(worst, spread)
        print(seed, L, {k: round(v, 4) for k, v in vals.items()}, "spread", round(spread, 5))
print("WORST DIFF", worst, "-> OK" if worst < 0.05 else "-> FAIL: the mask lets the filler content leak")

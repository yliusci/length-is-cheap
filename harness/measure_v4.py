"""Filler v4: near_distractors WITHOUT any fake fact with the same key as the needle (fixes the v1 to v3 filler).

One group = 3 seeds. Seed 1: near, conflict1, neutral arms (paired). Seeds 2 and 3: near only. That is 300/100/100 for 100 groups.
  near     : near_distractors filler, needle key excluded (n_samekey = 0, checked)
  conflict1: near + exactly 1 sentence with the same key and another value, random position
  neutral  : random_words filler (length reference)
a* probe identical to measure_c2.py (heads chosen in short context, last token of the question).
Usage: CTXROT_ONLY=<model> CTXROT_DTYPE=float16 uv run python harness/measure_v4.py <L> <n_groups> <max_seconds>
Output: runs/c2_v4/<model>_L<L>.jsonl. Exit code: 0 finished, 3 memory < 15 %, 4 repeated errors.
"""
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.dirname(__file__))
import ctxrot_v3 as V  # noqa: E402
from attn_probe import probe  # noqa: E402

MODEL = os.environ["CTXROT_ONLY"]
TOPK = 8
MEM_ABORT = 15
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "c2_v4")
LAB_FILL, LAB_NEEDLE, LAB_DIST, LAB_CONF = 0, 1, 2, 3


def mem_pct():
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, timeout=5).stdout)
    except Exception:
        return 100


def _enc(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def build(p, seed, arm):
    """arm: short, near, conflict1 or neutral. Same seed = same key, same value, same distractors, same near filler."""
    rng = np.random.default_rng(seed)
    tok, model = V._load(p["model"])
    key = str(rng.choice(V.WORDS)); val = str(rng.integers(1000, 9999))
    needle = _enc(tok, f" The secret code of the {key} is {val}.")
    distract = []
    for _ in range(p["n_distractors"]):
        k2 = str(rng.choice([w for w in V.WORDS if w != key]))
        distract.append(_enc(tok, f" The secret code of the {k2} is {rng.integers(1000, 9999)}."))
    question = _enc(tok, f"\nQuestion: what is the secret code of the {key}? Answer: The secret code of the {key} is")
    answer = _enc(tok, f" {val}")
    prefix = ([tok.bos_token_id] if tok.bos_token_id is not None and tok("a")["input_ids"][0] == tok.bos_token_id else []) \
        + _enc(tok, V.PREFIX)
    rconf = np.random.default_rng([seed, 3])
    other = str(rconf.choice([v for v in range(1000, 9999) if str(v) != val]))
    conf = _enc(tok, f" The secret code of the {key} is {other}.")
    fixed = len(prefix) + len(needle) + sum(map(len, distract)) + len(question) + len(answer) + len(conf)
    L_eff = max(0, min(p["L"], getattr(model.config, "max_position_embeddings", 10**9) - fixed))
    if arm == "short":
        fill = []
    elif arm == "neutral":
        fill = V._filler("random_words", L_eff, tok, np.random.default_rng([seed, 1]))
    else:
        fill = V._filler("near_distractors", L_eff, tok, np.random.default_rng([seed, 1]), exclude=key)
    n_samekey = len(re.findall(rf"code of the {key} is", tok.decode(fill))) if arm in ("near", "conflict1") else 0
    assert n_samekey == 0, (seed, key, n_samekey)
    body = [(t, LAB_FILL) for t in fill]
    for d in distract:
        pos = int(rng.integers(0, len(body) + 1)); body[pos:pos] = [(t, LAB_DIST) for t in d]
    if arm == "conflict1":
        pos = int(rconf.integers(0, len(body) + 1))
        while 0 < pos < len(body) and body[pos][1] == LAB_DIST:
            pos += 1
        body[pos:pos] = [(t, LAB_CONF) for t in conf]
        n_samekey = 1
    cut = int(round(p["depth"] * len(body)))
    while 0 < cut < len(body) and body[cut][1] in (LAB_DIST, LAB_CONF):
        cut += 1
    body[cut:cut] = [(t, LAB_NEEDLE) for t in needle]
    lab = [l for _, l in body]
    conf_after = None
    if arm == "conflict1":
        conf_after = bool(lab.index(LAB_CONF) > lab.index(LAB_NEEDLE))
    P = len(prefix)
    ids = prefix + [t for t, _ in body] + question + answer
    n_ctx = P + len(body) + len(question)
    labels = np.full(len(ids), -1, int); labels[P:P + len(body)] = lab
    return ids, n_ctx, answer, labels, n_samekey, conf_after


def _forward(ids, n_ctx, ans):
    import torch
    dev = V._dev()
    x = torch.tensor([ids], device=dev)
    _, model = V._load(MODEL)
    out, a = probe(model, x, torch.ones_like(x), torch.arange(len(ids), device=dev)[None], [n_ctx - 1], logits_to_keep=len(ans) + 1)
    lp = torch.log_softmax(out.logits[0, :len(ans)].float(), -1)
    return float(sum(lp[i, t] for i, t in enumerate(ans))), bool(all(int(lp[i].argmax()) == t for i, t in enumerate(ans))), a[:, :, 0, :].float()


def trial(p, seed, arms):
    import torch
    ids, nc, ans, lab, _, _ = build(p, seed, "short")
    lp_s, acc_s, A = _forward(ids, nc, ans)
    nee = torch.as_tensor(np.where((lab == 1) & (np.arange(len(lab)) < nc))[0], device=A.device)
    mass = A[..., nee].sum(-1)
    H = mass.shape[1]
    topk = torch.topk(mass.flatten(), TOPK).indices.cpu().numpy()
    a_short = float(mass.flatten()[topk].mean())
    del A, mass
    rows = []
    for arm in arms:
        t0 = time.time()
        ids, nc, ans, lab, n_samekey, conf_after = build(p, seed, arm)
        lp_l, acc_l, A = _forward(ids, nc, ans)
        M = A[[int(i) // H for i in topk], [int(i) % H for i in topk]].cpu().numpy()  # [TOPK, n_ctx]
        del A
        ctx = np.arange(len(lab)) < nc
        part = lambda v: float(M[:, np.where((lab == v) & ctx)[0]].sum(1).mean())
        rows.append({"version": "filler_v4", "arm": arm, "model": p["model"], "L": p["L"], "L_eff": len(ids), "depth": p["depth"],
                     "nd": p["n_distractors"], "seed": seed, "lp_short": lp_s, "lp_long": lp_l, "loss": lp_s - lp_l,
                     "acc_short": acc_s, "acc_long": acc_l, "a_star_short": a_short, "a_star_mean": part(1), "a_dist_mean": part(2),
                     "a_fill_mean": part(0), "a_conf_mean": part(3), "a_sink_mean": float(M[:, 0].mean()),
                     "n_fill": int(((lab == 0) & ctx).sum()), "n_dist": int(((lab == 2) & ctx).sum()), "n_samekey": n_samekey,
                     "conf_after_needle": conf_after, "dt": round(time.time() - t0, 2)})
    return rows


if __name__ == "__main__":
    L, n_groups, max_s = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{MODEL.split('/')[-1]}_L{L}.jsonl")
    done = 0
    if os.path.exists(path):
        done = sum(1 for l in open(path) if '"conflict1"' in l)
    rng = np.random.default_rng(int(time.time()))
    t0 = time.time()
    errs = 0
    for g in range(done, n_groups):
        for i in range(3):
            if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "STOP")) or time.time() - t0 > max_s:
                print("stop (time limit or STOP file)", g, flush=True); sys.exit(0)
            m = mem_pct()
            if m < MEM_ABORT:
                print(f"[STOP memory] {m}%", flush=True); sys.exit(3)
            p = {"model": MODEL, "L": L, "depth": float(rng.uniform(0.1, 0.9)), "n_distractors": int(rng.integers(1, 5))}
            try:
                rows = trial(p, int(rng.integers(2**31)), ("near", "conflict1", "neutral") if i == 0 else ("near",))
                errs = 0
            except AssertionError:
                raise
            except Exception as e:
                rows = [{"error": repr(e)[:200], "L": L, "arm": "error"}]
                errs += 1
            with open(path, "a") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)
            if errs >= 5:
                print("5 errors in a row", rows[0], flush=True); sys.exit(4)
        if g % 5 == 0:
            print(g, round(time.time() - t0), "s", {r["arm"]: (round(r["loss"], 2), round(r["a_star_mean"], 3)) for r in rows if "arm" in r and "loss" in r}, flush=True)
    print("DONE", MODEL, L, flush=True)

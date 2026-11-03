"""C2: relates the log-prob loss to INTERNAL (pre-softmax) attention scores, per retrieval head.

For each config (model, L, seed) with near_distractors filler:
  - short condition: finds the "retrieval heads" = heads whose attention mass (last question token
    -> needle tokens) is the highest;
  - long condition: on these heads, extracts the pre-softmax scores s* (needle), s_j (distractors), s_f (filler), s0 (sink),
    and the attention mass a* on the needle;
  - measures the loss lp_short - lp_long.
Features aggregated over the top-K heads: a*_mean, logsumexp of the filler+distractor scores, s*_mean, s0_mean, N_eff.
Writes one JSON line per config in runs/c2/<tag>.jsonl. The fit (mean field log N vs extreme values) is in fit_c2.py.

Eager attention is mandatory (scores must be accessible): memory-expensive -> L <= 2048, one model at a time, MPS cap.
Usage: CTXROT_ONLY=<model> CTXROT_MAXL=2048 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 uv run python harness/measure_c2.py <tag> <n_configs>
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
import ctxrot_v3 as V  # noqa: E402
from attn_probe import probe  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "test"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
MODEL = os.environ.get("CTXROT_ONLY", "Qwen/Qwen2.5-1.5B-Instruct")
MAXL = int(os.environ.get("CTXROT_MAXL", 2048))
TOPK = int(os.environ.get("C2_TOPK", 8))
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "c2")
os.makedirs(OUT, exist_ok=True)


def build(p, seed, short):
    """Rebuilds a near_distractors context and returns ids, position_ids, and the needle/distract/sink/filler indices."""
    import torch
    rng = np.random.default_rng(seed)
    tok, model = V._load(p["model"])
    WORDS = V.WORDS
    key = str(rng.choice(WORDS)); val = str(rng.integers(1000, 9999))
    needle = tok(f" The secret code of the {key} is {val}.", add_special_tokens=False)["input_ids"]
    distract = []
    for _ in range(p["n_distractors"]):
        k2 = str(rng.choice([w for w in WORDS if w != key]))
        distract.append(tok(f" The secret code of the {k2} is {rng.integers(1000, 9999)}.", add_special_tokens=False)["input_ids"])
    question = tok(f"\nQuestion: what is the secret code of the {key}? Answer: The secret code of the {key} is",
                   add_special_tokens=False)["input_ids"]
    answer = tok(f" {val}", add_special_tokens=False)["input_ids"]
    prefix = ([tok.bos_token_id] if tok.bos_token_id is not None and tok("a")["input_ids"][0] == tok.bos_token_id else []) \
        + tok(V.PREFIX, add_special_tokens=False)["input_ids"]
    fixed = len(prefix) + len(needle) + sum(map(len, distract)) + len(question) + len(answer)
    max_pos = getattr(model.config, "max_position_embeddings", 10**9)
    L_eff = 0 if short else max(0, min(p["L"], max_pos - fixed))
    fill = [] if short else V._filler("near_distractors", L_eff, tok, np.random.default_rng([seed, 1]))
    body = [(t, 0) for t in fill]
    for d in distract:
        pos = int(rng.integers(0, len(body) + 1)); body[pos:pos] = [(t, 2) for t in d]
    cut = int(round(p["depth"] * len(body)))
    while 0 < cut < len(body) and body[cut][1] == 2:
        cut += 1
    body[cut:cut] = [(t, 1) for t in needle]
    labels = np.array([l for _, l in body]); toks = [t for t, _ in body]
    P = len(prefix)
    ids = prefix + toks + question + answer
    n_ctx = P + len(toks) + len(question)
    lab = np.full(len(ids), -1, int); lab[P:P + len(toks)] = labels  # -1 outside body (prefix/question/answer)
    return tok, model, ids, n_ctx, answer, lab, P


def measure(p, seed):
    import torch
    dev = V._dev()
    # short: find the retrieval heads (needle attention mass, last position of the question)
    tok, model, ids_s, nc_s, ans, lab_s, _ = build(p, seed, True)
    x = torch.tensor([ids_s], device=dev); am = torch.ones_like(x); pos = torch.arange(len(ids_s), device=dev)[None]
    out, a_s = probe(model, x, am, pos, [nc_s - 1], logits_to_keep=len(ans) + 1)  # a_s [layers,heads,1,nc_s]
    lp_s = _logp(out.logits, len(ids_s), ans)
    nee_s = np.where((lab_s == 1) & (np.arange(len(lab_s)) < nc_s))[0]
    head_mass = a_s[:, :, 0, :][..., nee_s].sum(-1)  # [layers, heads]
    H = head_mass.shape[1]
    flat = head_mass.flatten(); topk = torch.topk(flat, min(TOPK, flat.numel())).indices.cpu().numpy()
    a_star_short = float(flat[topk].mean())  # needle mass of these heads in short context (reference)
    del out, a_s
    # long: extract a* and competitors on these heads
    tok, model, ids_l, nc_l, ans, lab_l, _ = build(p, seed, False)
    x = torch.tensor([ids_l], device=dev); am = torch.ones_like(x); pos = torch.arange(len(ids_l), device=dev)[None]
    out, a_l = probe(model, x, am, pos, [nc_l - 1], logits_to_keep=len(ans) + 1)  # [layers,heads,1,nc_l]
    lp_l = _logp(out.logits, len(ids_l), ans)
    ctx = np.arange(len(lab_l)) < nc_l
    nee = np.where((lab_l == 1) & ctx)[0]; dis = np.where((lab_l == 2) & ctx)[0]; fil = np.where((lab_l == 0) & ctx)[0]
    A = a_l[:, :, 0, :]  # [layers, heads, nc_l]
    feats = []
    for idx in topk:
        layer, head = int(idx) // H, int(idx) % H
        a = A[layer, head].float().cpu().numpy()
        feats.append((float(a[nee].sum()), float(a[dis].sum()) if len(dis) else 0.0,
                      float(a[fil].sum()) if len(fil) else 0.0, float(a[0])))
    del out, a_l, A
    feats = np.array(feats)  # [K, 4] : a_star, a_dist, a_fill, a_sink
    return {"model": p["model"], "L": p["L"], "L_eff": len(ids_l), "depth": p["depth"], "nd": p["n_distractors"],
            "seed": seed, "lp_short": lp_s, "lp_long": lp_l, "d_sim": lp_s - lp_l,
            "a_star_short": a_star_short,
            "a_star_mean": float(feats[:, 0].mean()), "a_star_min": float(feats[:, 0].min()),
            "a_dist_mean": float(feats[:, 1].mean()), "a_fill_mean": float(feats[:, 2].mean()),
            "a_sink_mean": float(feats[:, 3].mean()),
            "n_fill": int(len(fil)), "n_dist": int(len(dis)), "topk": TOPK}


def _logp(logits, n_ctx, ans):
    # logits truncated by logits_to_keep=len(ans)+1: row 0 predicts ans[0], etc.
    import torch
    lp = torch.log_softmax(logits[0, :len(ans)].float(), -1)
    return float(sum(lp[i, a] for i, a in enumerate(ans)))


if __name__ == "__main__":
    path = os.path.join(OUT, f"{TAG}.jsonl")
    rng = np.random.default_rng(int(time.time()))
    done = sum(1 for _ in open(path)) if os.path.exists(path) else 0
    import subprocess
    def mem_pct():
        try: return int(subprocess.run(["sysctl","-n","kern.memorystatus_level"],capture_output=True,text=True,timeout=5).stdout)
        except Exception: return 100
    for i in range(N):
        if os.path.exists(os.path.join(os.path.dirname(__file__), "..", "STOP")):
            break
        m = mem_pct()
        if m < 12:
            print(f"[STOP memory] {m}%", flush=True); break
        while m < 20:
            time.sleep(30); m = mem_pct()
        L = int(rng.choice([x for x in (512, 1024, 2048) if x <= MAXL]))
        p = {"model": MODEL, "L": L, "depth": float(rng.uniform(0.1, 0.9)), "n_distractors": int(rng.integers(1, 5))}
        try:
            r = measure(p, int(rng.integers(2**31)))
        except Exception as e:
            r = {"error": repr(e)[:200], "L": L}
        with open(path, "a") as fh:
            fh.write(json.dumps(r) + "\n")
        if i < 3 or i % 25 == 0:
            print(done + i + 1, {k: (round(v, 3) if isinstance(v, float) else v)
                                 for k, v in r.items() if k in ("L_eff", "d_sim", "a_star_mean", "error")}, flush=True)

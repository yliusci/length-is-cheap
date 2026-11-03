"""Protocol of Du et al. (2510.05381), "[Evidence][Masks][Question]", replayed in our pipeline on MMLU.

For one MMLU item (question + 4 choices = the "evidence"), we insert N filler tokens between the item and the answer instruction:
  Lshort : no filler
  Lmask  : filler present but MASKED in attention, positions kept (= their protocol)
  Lneu   : same filler, NOT masked
  Lws    : same with a "whitespace" filler, not masked (another condition of Du et al.)
Measure: log-prob of the correct letter (and top-1 among A/B/C/D).
score   = Lshort - Lmask  (their effect "length alone degrades")
control = Lshort - Lneu   (same length, filler visible): if control is about 0 while score > 0, the effect of masking
          comes from the position gap it creates, not from length.
"""
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__))
import ctxrot_v3 as V  # noqa: E402  model loading, device, PREFIX, random_words filler

CONFIG = {"threshold": 0.5, "replicates": 5, "exploit_frac": 0.3, "max_filons": 16}
_ITEMS = None
_MAXL = int(os.environ.get("CTXROT_MAXL", 4096))
_ONLY = os.environ.get("CTXROT_ONLY", "Qwen/Qwen2.5-1.5B-Instruct")


def _items():
    global _ITEMS
    if _ITEMS is None:
        path = os.path.join(os.path.dirname(__file__), "..", "context", "mmlu_test.jsonl")
        _ITEMS = [json.loads(l) for l in open(path)]
    return _ITEMS


def sample(rng):
    return {"model": _ONLY, "L": int(rng.choice([x for x in (512, 1024, 2048, 4096, 8192) if x <= _MAXL])),
            "item": int(rng.integers(len(_items())))}


def perturb(p, rng, scale):
    q = dict(p)
    q["item"] = int(rng.integers(len(_items())))  # one vein = one regime (model, L), not one item
    return q


def same_basin(p, q):
    return p["L"] == q["L"]


def _forward(p, seed, kind, mask):
    import torch
    tok, model = V._load(p["model"])
    it = _items()[p["item"]]
    letters = "ABCD"
    evidence = "Question: " + it["question"].strip() + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, it["choices"])) + "\n"
    prefix = ([tok.bos_token_id] if tok.bos_token_id is not None and tok("a")["input_ids"][0] == tok.bos_token_id else []) \
        + tok(V.PREFIX, add_special_tokens=False)["input_ids"]
    ev = tok(evidence, add_special_tokens=False)["input_ids"]
    q = tok("\nAnswer with the letter of the correct option.\nAnswer:", add_special_tokens=False)["input_ids"]
    cand = [tok(f" {l}", add_special_tokens=False)["input_ids"][-1] for l in letters]
    fixed = len(prefix) + len(ev) + len(q) + 1
    L = 0 if kind == "short" else max(0, min(p["L"], getattr(model.config, "max_position_embeddings", 10**9) - fixed))
    rng = np.random.default_rng([seed, 1])
    if kind == "whitespace":
        fill = tok(" " * L, add_special_tokens=False)["input_ids"][:L]
        fill += tok(" ", add_special_tokens=False)["input_ids"] * (L - len(fill))
    elif L:
        fill = V._filler("random_words", L, tok, rng)
    else:
        fill = []
    ids = prefix + ev + fill + q
    attn = torch.ones(1, len(ids), dtype=torch.long, device=V._dev())
    if mask and fill:
        s = len(prefix) + len(ev)
        attn[0, s:s + len(fill)] = 0
    x = torch.tensor([ids], device=V._dev())
    pos = torch.arange(len(ids), device=V._dev())[None]
    with torch.no_grad():
        out = model(input_ids=x, attention_mask=attn, position_ids=pos, logits_to_keep=1)
    logp = torch.log_softmax(out.logits[0, -1].float(), -1)
    gold = cand[it["answer"]]
    lp = float(logp[gold])
    acc = bool(int(torch.stack([logp[c] for c in cand]).argmax()) == it["answer"])
    return lp, acc, len(fill)


def run(p, seed):
    ls, a_s, _ = _forward(p, seed, "short", False)
    lm, a_m, n = _forward(p, seed, "neutral", True)
    ln, a_n, _ = _forward(p, seed, "neutral", False)
    lw, a_w, _ = _forward(p, seed, "whitespace", False)
    return {"score": ls - lm, "d_mask": ls - lm, "d_neu": ls - ln, "d_ws": ls - lw,
            "lp_short": ls, "lp_mask": lm, "lp_neu": ln, "lp_ws": lw,
            "acc_short": a_s, "acc_mask": a_m, "acc_neu": a_n, "acc_ws": a_w, "n_fill": n}


def control(p, seed):
    r = run(p, seed)
    return {**r, "score": r["d_neu"]}

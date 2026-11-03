"""Replication of the "masking" arm of Du et al. (2510.05381, section 4.2, table 9): [Evidence][Masks][Question].

Prompts = figures 8 and 10 of their appendix. Accepted differences: local models (Llama-3.2-3B, Qwen2.5-7B-Instruct),
no HumanEval or GSM8K. Unpublished format, so chosen here: MMLU options numbered 1-4, VarSum "xNN = v".

A masked token is never an attention key: it changes no other token. So we remove it from the input and
keep the jump in position_ids, which costs O(short) at 30K. `check` verifies equality with the real mask.

Usage: du_repro.py check | du_repro.py run <mmlu|varsum> <n_items>
Output: runs/du_repro/<model>_<task>.jsonl (automatic resume). Stop: STOP file or memory < 12 %.
"""
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL = os.environ.get("CTXROT_ONLY", "meta-llama/Llama-3.2-3B")
LENGTHS = (0, 3750, 7500, 11250, 15000, 18750, 22500, 26250, 30000)
MAX_NEW = {"mmlu": 8, "varsum": 160}
SEED = 20251006
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()
    return tok, model


def enc(tok, s):
    return tok(s, add_special_tokens=False)["input_ids"]


def bos(tok):
    return [tok.bos_token_id] if tok.bos_token_id is not None and tok("a")["input_ids"][0] == tok.bos_token_id else []


def items(task, n):
    rng = np.random.default_rng(SEED)
    if task == "mmlu":
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "context", "mmlu_test.jsonl"))]
        idx = rng.permutation(len(rows))[:n]
        return [{"id": int(i), "ev": rows[i]["question"].strip(),
                 "q": "Choose the option that best satisfies the problem description.\n"
                      + "\n".join(f"{k + 1}. {c}" for k, c in enumerate(rows[i]["choices"]))
                      + "\nGive only the number for the correct option.",
                 "tail": "The correct option is", "gold": int(rows[i]["answer"]) + 1} for i in idx]
    out = []
    for i in range(n):
        names = [f"x{k}" for k in rng.permutation(100)[:50]]
        vals = [int(v) for v in rng.integers(1, 100, 50)]
        pick = rng.choice(50, 3, replace=False)
        out.append({"id": i, "ev": "\n".join(f"{a} = {v}" for a, v in zip(names, vals)),
                    "q": "What is the sum of " + ", ".join(names[j] for j in pick) + "?",
                    "tail": "Let's think step by step.", "gold": int(sum(vals[j] for j in pick))})
    return out


def parts(tok, it):
    head = bos(tok) + enc(tok, f"# Problem Description\n{it['ev']}\n# Others\n")
    tail = enc(tok, f"\n# Question\n{it['q']}\n# Answer\n{it['tail']}")
    return head, tail


@torch.no_grad()
def generate(model, tok, head, tail, gap, max_new):
    """Greedy, manual cache: model.generate would recompute position_ids and lose the jump."""
    ids = head + tail
    pos = list(range(len(head))) + [len(head) + gap + i for i in range(len(tail))]
    x = torch.tensor([ids], device=DEV)
    p = torch.tensor([pos], device=DEV)
    cache, out, stop = None, [], {tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>"), tok.convert_tokens_to_ids("<|im_end|>")}
    for _ in range(max_new):
        r = model(input_ids=x, position_ids=p, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = r.past_key_values
        t = int(r.logits[0, -1].argmax())
        if t in stop:
            break
        out.append(t)
        x = torch.tensor([[t]], device=DEV)
        p = p[:, -1:] + 1
    return tok.decode(out, skip_special_tokens=True)


def score(task, text, gold):
    if task == "mmlu":
        m = re.search(r"[1-4]", text)
        return bool(m) and int(m.group()) == gold
    m = re.findall(r"-?\d+", text.replace(",", ""))
    return bool(m) and int(m[-1]) == gold


@torch.no_grad()
def check():
    """Shortcut logits vs real attention_mask, on the first step, with real filler."""
    tok, model = load()
    it = items("mmlu", 3)
    for gap in (256, 1024):
        for e in it:
            head, tail = parts(tok, e)
            fill = (enc(tok, "the quick brown fox jumps over the lazy dog " * gap))[:gap]
            ids = head + fill + tail
            am = torch.ones(1, len(ids), dtype=torch.long, device=DEV)
            am[0, len(head):len(head) + gap] = 0
            full = model(input_ids=torch.tensor([ids], device=DEV), attention_mask=am,
                         position_ids=torch.arange(len(ids), device=DEV)[None], logits_to_keep=1).logits[0, -1].float()
            pos = list(range(len(head))) + [len(head) + gap + i for i in range(len(tail))]
            cut = model(input_ids=torch.tensor([head + tail], device=DEV),
                        position_ids=torch.tensor([pos], device=DEV), logits_to_keep=1).logits[0, -1].float()
            none = model(input_ids=torch.tensor([head + tail], device=DEV), logits_to_keep=1).logits[0, -1].float()
            print(gap, "max|mask-shortcut|", float((full - cut).abs().max()),
                  "max|shortcut-no_gap|", float((cut - none).abs().max()), "top", int(full.argmax()) == int(cut.argmax()), flush=True)


def mem_pct():
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, timeout=5).stdout)
    except Exception:
        return 100


def run(task, n):
    tok, model = load()
    os.makedirs(os.path.join(ROOT, "runs", "du_repro"), exist_ok=True)
    path = os.path.join(ROOT, "runs", "du_repro", f"{MODEL.split('/')[-1]}_{task}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            d = json.loads(l)
            done.add((d["id"], d["L"]))
    t0 = time.time()
    data = items(task, n)
    for k, it in enumerate(data):
        head, tail = parts(tok, it)
        for L in LENGTHS:
            if (it["id"], L) in done:
                continue
            if os.path.exists(os.path.join(ROOT, "STOP")):
                return
            m = mem_pct()
            if m < 12:
                print(f"[STOP memory] {m}%", flush=True)
                return
            while m < 20:
                time.sleep(30)
                m = mem_pct()
            t = time.time()
            try:
                txt = generate(model, tok, head, tail, L, MAX_NEW[task])
                r = {"id": it["id"], "L": L, "ok": score(task, txt, it["gold"]), "gold": it["gold"], "out": txt, "dt": round(time.time() - t, 3)}
            except Exception as e:
                r = {"id": it["id"], "L": L, "error": repr(e)[:200]}
            with open(path, "a") as fh:
                fh.write(json.dumps(r) + "\n")
        if k % 25 == 0:
            print(k, round(time.time() - t0), "s", {key: r.get(key) for key in ("ok", "dt", "error")}, flush=True)
    print("DONE", task, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check()
    else:
        run(sys.argv[2], int(sys.argv[3]))

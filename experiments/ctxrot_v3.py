"""Context rot v3: C1 decomposition with a NEUTRAL reference.

Observation (Qwen 1.5B at 30K, laptop run): with a non-similar filler, Llong is about Lshort; only the masked condition drops.
Masking with kept positions creates an empty "gap" of positions, which is out of distribution: Lmask is not a
pure-distance reference. v3 takes as reference an unmasked NEUTRAL filler (random_words), at the same positions.
Arms measured (same seeds, same structure; the filler content has its own random generator):
  Lshort  : needle right before the question
  Lneu    : needle at depth d inside N tokens of neutral filler (random_words), not masked
  Llong   : same with the studied filler p["filler"]
  Lmask   : Llong with the filler masked (positions kept), kept to quantify the gap artifact
  d_dist = Lshort - Lneu   (distance + generic filler)      d_sim = Lneu - Llong (competition due to similarity)
  d_gap  = Lneu - Lmask    (cost of the position gap)       d_tot = Lshort - Llong = d_dist + d_sim
The total sequence is truncated to stay <= config.max_position_embeddings (field L_eff).
score = d_sim ; control = same measure with ANOTHER neutral filler (repeat_sentence) as the studied filler: d_sim is expected to be about 0.
"""
import os

import numpy as np

CONFIG = {"threshold": 1.0, "replicates": 5, "exploit_frac": 0.5, "max_filons": 16}
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
          "allenai/OLMo-2-0425-1B-Instruct", "meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-7B-Instruct"]
MAX_L = {"HuggingFaceTB/SmolLM2-1.7B-Instruct": 4096, "allenai/OLMo-2-0425-1B-Instruct": 4096}
PREFIX = "You are a careful assistant. Read the document and answer the question.\n"
FILLERS = ["random_words", "repeat_sentence", "near_distractors", "numbers"]
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "..", "models", "hf"))

_cache = {}


def _dev():
    import torch
    if torch.cuda.is_available():
        frac = os.environ.get("CTXROT_VRAM_FRAC")
        if frac and not getattr(_dev, "_capped", False):
            torch.cuda.set_per_process_memory_fraction(float(frac)); _dev._capped = True
        return "cuda"
    return "mps"
WORDS = ("apple river stone cloud paper window garden silver motor yellow forest candle ladder bottle "
         "mirror pencil castle rocket marble tunnel violin anchor basket cactus dragon falcon harbor "
         "island jacket kettle lemon magnet needle orange pillow quartz saddle tiger umbrella velvet "
         "walnut zipper bridge camera desert engine feather glacier hammer insect jungle").split()


def sample(rng):
    m = str(rng.choice(MODELS, p=[.3, .25, .15, .15, .1, .05]))
    Ls = [x for x in (512, 1024, 2048, 4096, 8192, 16384) if x <= MAX_L.get(m, 16384)]
    return {"model": m, "L": int(rng.choice(Ls)),
            "depth": float(rng.uniform(0, 1)), "filler": str(rng.choice(FILLERS)),
            "n_distractors": int(rng.integers(0, 6))}


def perturb(p, rng, scale):
    q = dict(p)
    q["depth"] = float(np.clip(q["depth"] + rng.normal(0, 0.2 * scale), 0, 1))
    if rng.random() < 0.3:
        q["L"] = int(np.clip(q["L"] * 2.0 ** int(rng.choice([-1, 1])), 128, 32768))  # the wrappers re-cap per model (_M, CTXROT_MAXL)
    if rng.random() < 0.2:
        q["n_distractors"] = int(np.clip(q["n_distractors"] + rng.integers(-1, 2), 0, 10))
    return q


def same_basin(p, q):
    return (p["model"], p["filler"], p["L"]) == (q["model"], q["filler"], q["L"]) and abs(p["depth"] - q["depth"]) < 0.15


def _load(name):
    if name not in _cache:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, os.environ.get("CTXROT_DTYPE", "float32")), attn_implementation="sdpa")
        if len(_cache) >= int(os.environ.get("CTXROT_MAX_CACHED", 3)): _cache.pop(next(iter(_cache)))
        _cache[name] = (tok, model.to(_dev()).eval())
    return _cache[name]


def _filler(kind, n_tokens, tok, rng, exclude=None):
    if kind == "random_words":
        text = " ".join(rng.choice(WORDS, size=n_tokens))
    elif kind == "repeat_sentence":
        text = " The weather report for the coastal region was unremarkable." * (n_tokens // 10 + 1)
    elif kind == "numbers":
        text = " ".join(str(x) for x in rng.integers(0, 10000, size=n_tokens))
    else:  # near_distractors: fake facts with the same form as the needle
        pool = [w for w in WORDS if w != exclude]  # exclude = needle key: no fake fact with the same key
        text = " ".join(f"The secret code of the {rng.choice(pool)} is {rng.integers(1000, 9999)}."
                        for _ in range(n_tokens // 12 + 1))
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
    return ids


def _measure(p, seed, mask_filler: bool, short: bool, filler=None):
    import torch
    rng = np.random.default_rng(seed)
    tok, model = _load(p["model"])
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
        + tok(PREFIX, add_special_tokens=False)["input_ids"]
    fixed = len(prefix) + len(needle) + sum(map(len, distract)) + len(question) + len(answer)
    max_pos = getattr(model.config, "max_position_embeddings", 10**9)
    L_eff = max(0, min(p["L"], max_pos - fixed))
    fill = [] if short else _filler(filler or p["filler"], L_eff, tok, np.random.default_rng([seed, 1]),
                                    exclude=key if os.environ.get("CTXROT_EXCLUDE_KEY") == "1" else None)
    # labels: 0 = filler (maskable), 1 = needle, 2 = distractor; units are never cut
    body = [(t, 0) for t in fill]
    for d in distract:
        pos = int(rng.integers(0, len(body) + 1)); body[pos:pos] = [(t, 2) for t in d]
    cut = int(round(p["depth"] * len(body)))
    while 0 < cut < len(body) and body[cut][1] == 2:
        cut += 1
    body[cut:cut] = [(t, 1) for t in needle]
    labels = np.array([l for _, l in body])
    body = [t for t, _ in body]
    P = len(prefix)
    ids = prefix + body + question + answer
    n_ctx = P + len(body) + len(question)
    is_filler = np.zeros(len(ids), bool)
    is_filler[P:P + len(body)] = labels == 0
    x = torch.tensor([ids], device=_dev())
    attn = torch.ones_like(x)
    if mask_filler:
        attn[0, torch.tensor(is_filler, device=_dev())] = 0
    pos = torch.arange(len(ids), device=_dev())[None]
    with torch.no_grad():
        if os.environ.get("CTXROT_KEEP_LOGITS") == "1":
            # compute logits only for the positions that are read (saves ~vocab x N x 2 bytes)
            out = model(input_ids=x, attention_mask=attn, position_ids=pos, logits_to_keep=len(answer) + 1)
            logp = torch.log_softmax(out.logits[0, :len(answer)].float(), -1)
        else:
            out = model(input_ids=x, attention_mask=attn, position_ids=pos)
            logp = torch.log_softmax(out.logits[0, n_ctx - 1:n_ctx - 1 + len(answer)].float(), -1)
    lp = float(sum(logp[i, a] for i, a in enumerate(answer)))
    top1 = bool(all(int(logp[i].argmax()) == a for i, a in enumerate(answer)))
    return lp, top1, len(ids), L_eff


def run(p, seed, filler=None):
    f = filler or p["filler"]
    ls, a_s, *_ = _measure(p, seed, False, short=True, filler=f)
    ln, a_n, *_ = _measure(p, seed, False, short=False, filler="random_words")
    ll, a_l, n, L_eff = _measure(p, seed, False, short=False, filler=f)
    lm, a_m, *_ = _measure(p, seed, True, short=False, filler=f)
    return {"score": ln - ll, "d_sim": ln - ll, "d_dist": ls - ln, "d_gap": ln - lm, "d_tot": ls - ll,
            "lp_short": ls, "lp_neu": ln, "lp_long": ll, "lp_mask": lm,
            "acc_short": a_s, "acc_neu": a_n, "acc_long": a_l, "acc_mask": a_m, "n_tokens": n, "L_eff": L_eff}


def control(p, seed):
    return run(p, seed, filler="repeat_sentence")

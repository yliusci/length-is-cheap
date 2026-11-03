"""Lightweight attention probe for C2: SDPA forward (no L x L matrix), and attention of ONLY the positions that are read.

A pre-hook on each `self_attn` gets hidden_states and (cos, sin), then recomputes:
  q = q_proj(h[sel]) (optional q_norm), k = k_proj(h) (optional k_norm), RoPE, GQA,
  p = softmax(q . k^T * scaling + causal mask + padding mask)  -> [layers, heads, n_sel, L]
Cost O(L) per layer. Covers Qwen2/2.5, Llama/SmolLM2 and OLMo-2 (q_norm/k_norm over the full dimension);
per-head q_norm (Qwen3) too. No sliding window (none of the project's models has one).

    from attn_probe import probe
    out, attn = probe(model, input_ids, attention_mask, position_ids, sel, logits_to_keep=k)
    # attn: float32 [n_layers, n_heads, len(sel), L], rows sum to 1

Validation: uv run python harness/attn_probe.py [model]  (compares to attn_implementation="eager", short L)
"""
import sys

import torch


def _norm(mod, x, n_heads, head_dim):
    if mod is None:
        return x
    if mod.weight.shape[-1] == head_dim:  # per-head norm (Qwen3)
        return mod(x.view(*x.shape[:-1], n_heads, head_dim)).view(x.shape)
    return mod(x)  # norm over the full dimension (OLMo-2)


def probe(model, input_ids, attention_mask, position_ids, sel, **fwd_kwargs):
    """sel: positions of the queries that are read (list of ints). Batch size 1."""
    sel_t = torch.as_tensor(sel, device=input_ids.device)
    keep = attention_mask[0].bool()  # visible keys (the masked filler is excluded)
    L = input_ids.shape[1]
    causal = torch.arange(L, device=input_ids.device)[None, :] <= sel_t[:, None]  # [n_sel, L]
    allowed = causal & keep[None, :]
    rows = []
    hooks = []
    for layer in model.model.layers:
        attn = layer.self_attn
        rope = sys.modules[type(attn).__module__].apply_rotary_pos_emb

        def pre(mod, args, kwargs, rope=rope):
            h = kwargs.get("hidden_states", args[0] if args else None)
            cos, sin = kwargs["position_embeddings"]
            hd = mod.head_dim
            nq = mod.q_proj.out_features // hd
            nk = mod.k_proj.out_features // hd
            q = _norm(getattr(mod, "q_norm", None), mod.q_proj(h[:, sel_t]), nq, hd)
            k = _norm(getattr(mod, "k_norm", None), mod.k_proj(h), nk, hd)
            q = q.view(1, len(sel), nq, hd).transpose(1, 2)
            k = k.view(1, L, nk, hd).transpose(1, 2)
            q, _ = rope(q, q, cos[:, sel_t], sin[:, sel_t])
            _, k = rope(k, k, cos, sin)
            k = k.repeat_interleave(nq // nk, dim=1)
            s = (q.float() @ k.float().transpose(-1, -2)) * mod.scaling  # [1, nq, n_sel, L]
            s = s.masked_fill(~allowed[None, None], float("-inf"))
            rows.append(torch.softmax(s, -1)[0])

        hooks.append(attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **fwd_kwargs)
    finally:
        for hk in hooks:
            hk.remove()
    return out, torch.stack(rows)


if __name__ == "__main__":
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(__file__), "..", "models", "hf"))
    name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    dev = "cuda" if torch.cuda.is_available() else "mps"
    tok = AutoTokenizer.from_pretrained(name)
    g = torch.Generator().manual_seed(0)
    worst = 0.0
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
    if True:
        for L, masked in ((300, False), (300, True), (500, True)):
            ids = torch.randint(100, 20000, (1, L), generator=g).to(dev)
            am = torch.ones_like(ids)
            if masked:
                am[0, 20:L // 2] = 0
            pos = torch.arange(L, device=dev)[None]
            sel = [L - 3, L - 2, L - 1]
            m.set_attn_implementation("sdpa")
            _, a = probe(m, ids, am, pos, sel, logits_to_keep=1)
            m.set_attn_implementation("eager")
            with torch.no_grad():
                e = m(input_ids=ids, attention_mask=am, position_ids=pos, output_attentions=True, logits_to_keep=1).attentions
            e = torch.stack([x[0, :, sel] for x in e]).float()
            d = (a - e).abs().max().item()
            worst = max(worst, d)
            print(name, "L", L, "mask", masked, "max diff", f"{d:.2e}")
    print("WORST DIFF", f"{worst:.2e}", "-> OK" if worst < 1e-4 else "-> FAIL")

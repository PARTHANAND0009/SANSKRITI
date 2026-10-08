"""Stage 3b: activation patching (denoising) as a causal check on depth.

Clean prompt: the question in a chosen option order (default: the original order).
Corrupted prompt: the same prompt with the entity span in the stem (entities.parquet
span_start/span_end) replaced by a same-attribute entity from a different state (swap
pool, seeded per question). Options are unchanged.

For each readout layer k in 0..L, the corrupted run is repeated with the final-token
residual at layer k replaced by the clean run's value (k = 0: embedding output, the
input of block 1; k >= 1: output of block k). All L+1 patched runs go in one batch.

Metric: logit difference LD = logit(gold letter) - max logit(other letters), on the
model's output restricted to A-D. Recovery_k = (LD_patched_k - LD_corrupt) /
(LD_clean - LD_corrupt). Patching depth at threshold t = first k with recovery >= t,
for t in 0.3, 0.5, 0.7 (layers; /L in the *_frac columns). Rows where the corruption
moves LD by less than --min-effect logits are kept but flagged valid = False.

Stratified sample (--build-sample): 3 attributes x 12 states x 2 frequency tiers x 20
questions, from the analysis set (no ambiguous_gold, no leaks_answer), questions with a
stem span and a frequency tier, attributes whose questions all have >= 3 swap
candidates. Attributes: the 3 with most eligible questions; states: the 12 that fill
the most cells (sum over cells of min(count, 20)). Written to
data/processed/patch_sample.csv and patch_sample_cells.csv (actual counts).

  python -m run.patch --build-sample
  python -m run.patch --model llama31_8b --device cuda
  python -m run.patch --model qwen25_05b_proxy --device cpu --limit 20 --allow-proxy
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crystal.io import (load_analysis_set, load_entities, load_freq, load_run_config, model_config, resolve,
                        set_seed, variant_columns)
from crystal.lens import decoder_layers
from data.prep import build_prompt, option_order

THRESHOLDS = (0.3, 0.5, 0.7)
MIN_SWAP_CANDIDATES = 3


# ------------------------------------------------------------------ sample

def eligible_questions() -> pd.DataFrame:
    an = load_analysis_set()                         # excludes ambiguous_gold and leaks_answer
    ent = load_entities()
    fr = load_freq()[["qid", "log_freq", "tier"]]
    pool = pd.read_parquet(resolve("data/processed/swap_pool.parquet"))
    pool = pool[pool.swap_ok]
    df = an.merge(ent, on="qid").merge(fr, on="qid")
    df = df[df.span_start.notna() & df.tier.notna()].copy()
    per_attr = pool.groupby("attribute").size()
    per_attr_state = pool.groupby(["attribute", "state"]).size()
    df["n_swap_candidates"] = [int(per_attr.get(a, 0) - per_attr_state.get((a, s), 0))
                               for a, s in zip(df.attribute, df.state)]
    attr_min = df.groupby("attribute").n_swap_candidates.min()
    keep_attr = attr_min[attr_min >= MIN_SWAP_CANDIDATES].index
    return df[df.attribute.isin(keep_attr)]


def build_sample(n_attr=3, n_states=12, per_cell=20, seed=1234, attributes=None, states=None):
    df = eligible_questions()
    attrs = list(attributes) if attributes else df.attribute.value_counts().index[:n_attr].tolist()
    d = df[df.attribute.isin(attrs)]
    cells = d.groupby(["state", "attribute", "tier"]).size()
    if states:
        sts = list(states)
    else:
        fill = {}
        for s in d.state.unique():
            fill[s] = sum(min(cells.get((s, a, t), 0), per_cell) for a in attrs for t in ("low", "high"))
        sts = sorted(fill, key=lambda s: (-fill[s], s))[:n_states]
    d = d[d.state.isin(sts)]
    parts = []
    for (s, a, t), g in d.groupby(["state", "attribute", "tier"]):
        parts.append(g.sample(min(per_cell, len(g)), random_state=seed))
    sample = pd.concat(parts).sort_values("qid")
    counts = (sample.groupby(["attribute", "state", "tier"]).size()
              .reindex(pd.MultiIndex.from_product([attrs, sorted(sts), ["low", "high"]],
                                                  names=["attribute", "state", "tier"]), fill_value=0)
              .rename("n").reset_index())
    cols = ["qid", "state", "attribute", "question_type", "tier", "log_freq", "entity", "span_start", "span_end",
            "n_swap_candidates"]
    return sample[cols], counts


# ------------------------------------------------------------------ prompts

def pick_swap(qid: str, attribute: str, state: str, entity: str, pool: pd.DataFrame, seed: int):
    cand = pool[(pool.attribute == attribute) & (pool.state != state) & pool.swap_ok
                & (pool.entity.str.lower() != str(entity).lower())].sort_values(["state", "entity"])
    if cand.empty:
        return None, None
    h = int(hashlib.sha256(f"{seed}:swap:{qid}".encode()).hexdigest()[:16], 16)
    r = cand.iloc[random.Random(h).randrange(len(cand))]
    return r.entity, r.state


def clean_and_corrupt(row, swap_entity: str, variant: str, base_seed: int):
    order = option_order(variant, int(row.gold_idx), row.qid, base_seed)
    opts = [row.options[i] for i in order]
    s, e = int(row.span_start), int(row.span_end)
    corrupt_stem = row.stem[:s] + swap_entity + row.stem[e:]
    return build_prompt(row.stem, opts), build_prompt(corrupt_stem, opts), order.index(int(row.gold_idx))


# ------------------------------------------------------------------ patching

def _encode(tok, prompt, device):
    enc = tok([prompt], return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def final_token_resid(model, ids, mask):
    """[L+1, d] final-token residual (0 = embedding output) and the option-agnostic logits."""
    layers = decoder_layers(model)
    store = {}

    def pre(mod, args, kwargs):
        store[0] = (args[0] if args else kwargs["hidden_states"])[:, -1].detach().clone()

    def post(i):
        def f(mod, inp, out):
            store[i] = (out[0] if isinstance(out, tuple) else out)[:, -1].detach().clone()
        return f

    hs = [layers[0].register_forward_pre_hook(pre, with_kwargs=True)]
    hs += [l.register_forward_hook(post(i + 1)) for i, l in enumerate(layers)]
    try:
        out = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
    return torch.cat([store[i] for i in range(len(layers) + 1)]), out.logits[:, -1]


@torch.no_grad()
def patched_logits(model, ids, mask, clean_resid: torch.Tensor):
    """Run L+1 copies of the corrupted prompt; copy k gets clean_resid[k] at the final
    position of readout layer k. Returns final logits [L+1, V]."""
    layers = decoder_layers(model)
    n = len(layers) + 1
    ids_b, mask_b = ids.expand(n, -1).contiguous(), mask.expand(n, -1).contiguous()

    def pre(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        h = h.clone()
        h[0, -1] = clean_resid[0].to(h.dtype)
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = h
        return args, kwargs

    def post(k):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[k, -1] = clean_resid[k].to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return f

    hs = [layers[0].register_forward_pre_hook(pre, with_kwargs=True)]
    hs += [l.register_forward_hook(post(i + 1)) for i, l in enumerate(layers)]
    try:
        out = model(input_ids=ids_b, attention_mask=mask_b, use_cache=False, logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
    return out.logits[:, -1]


def logit_diff(opt_logits: torch.Tensor, gold: int) -> torch.Tensor:
    """[..., 4] -> gold minus best other option."""
    other = opt_logits.clone()
    other[..., gold] = -float("inf")
    return opt_logits[..., gold] - other.max(-1).values


def first_at_least(rec: np.ndarray, t: float):
    hit = np.nonzero(rec >= t)[0]
    return int(hit[0]) if len(hit) else None


def patch_question(model, tok, clean_prompt, corrupt_prompt, gold, option_ids, device):
    idx = torch.as_tensor(option_ids, device=device)
    c_ids, c_mask = _encode(tok, clean_prompt, device)
    x_ids, x_mask = _encode(tok, corrupt_prompt, device)
    clean_resid, clean_logits = final_token_resid(model, c_ids, c_mask)
    _, corrupt_logits = final_token_resid(model, x_ids, x_mask)
    pl = patched_logits(model, x_ids, x_mask, clean_resid)
    ld_clean = logit_diff(clean_logits[0, idx].float(), gold).item()
    ld_corr = logit_diff(corrupt_logits[0, idx].float(), gold).item()
    ld_patch = logit_diff(pl[:, idx].float(), gold).cpu().numpy()
    denom = ld_clean - ld_corr
    rec = (ld_patch - ld_corr) / denom if denom != 0 else np.full_like(ld_patch, np.nan)
    return ld_clean, ld_corr, ld_patch, rec


# ------------------------------------------------------------------ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="stage 3b: activation patching")
    ap.add_argument("--build-sample", action="store_true")
    ap.add_argument("--attributes", nargs="*", default=None)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variant", default="prompt", help="option order of clean and corrupted prompts")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-effect", type=float, default=1.0, help="min |LD_clean - LD_corrupt| for valid rows")
    ap.add_argument("--shard-size", type=int, default=50)
    ap.add_argument("--cpu-layers", type=int, default=None)
    ap.add_argument("--allow-proxy", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    cfg = load_run_config()
    set_seed(cfg["seed"])
    proc = resolve(cfg["paths"]["processed"])

    if args.build_sample:
        sample, counts = build_sample(per_cell=args.per_cell, seed=cfg["seed"], attributes=args.attributes,
                                      states=args.states)
        sample.to_csv(proc / "patch_sample.csv", index=False)
        counts.to_csv(proc / "patch_sample_cells.csv", index=False)
        pv = counts.pivot_table(index="state", columns=["attribute", "tier"], values="n")
        print(f"patch sample: {len(sample)} questions (target {len(counts)} cells x {args.per_cell} = "
              f"{len(counts) * args.per_cell}); cells below target: {(counts.n < args.per_cell).sum()}")
        print(pv.to_string())
        if not args.model:
            return

    from run.forward import load_model_and_tokenizer, resolve_option_ids

    mcfg = model_config(args.model)
    if mcfg.get("proxy") and not args.allow_proxy:
        raise SystemExit(f"{args.model} is a proxy model; use --allow-proxy for tests")
    sample = pd.read_csv(proc / "patch_sample.csv")
    if args.limit:
        sample = sample.head(args.limit)
    from crystal.io import load_prompts
    prompts = load_prompts().set_index("qid")
    pool = pd.read_parquet(proc / "swap_pool.parquet")
    model, tok = load_model_and_tokenizer(mcfg, args.device, cpu_layers=args.cpu_layers)
    option_ids = resolve_option_ids(mcfg, tok)
    L = len(decoder_layers(model))
    out_dir = resolve(args.out) / f"patch_{args.model}_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({
        "model_id": mcfg["id"], "proxy": bool(mcfg.get("proxy")), "variant": args.variant, "L": L,
        "min_effect": args.min_effect, "thresholds": THRESHOLDS, "n_sample": len(sample),
        "option_ids": option_ids, "device": args.device}, indent=2))
    for si in range(0, len(sample), args.shard_size):
        path = out_dir / f"shard_{si // args.shard_size:05d}.parquet"
        part = sample.iloc[si:si + args.shard_size]
        if path.exists() and pd.read_parquet(path).qid.tolist() == part.qid.tolist():
            print(f"{path.name}: exists, skipped")
            continue
        rows, t0 = [], time.time()
        for r in part.itertuples():
            row = prompts.loc[r.qid]
            row = row.copy()
            row["qid"] = r.qid
            row["span_start"], row["span_end"] = r.span_start, r.span_end
            swap, swap_state = pick_swap(r.qid, r.attribute, r.state, r.entity, pool, cfg["seed"])
            base = {"qid": r.qid, "state": r.state, "attribute": r.attribute, "tier": r.tier,
                    "entity": r.entity, "swap_entity": swap, "swap_state": swap_state, "L": L}
            if swap is None:
                rows.append({**base, "valid": False, "note": "no swap candidate"})
                continue
            clean_p, corr_p, gold = clean_and_corrupt(row, swap, args.variant, cfg["prep"]["permute_seed"])
            ldc, ldx, ldp, rec = patch_question(model, tok, clean_p, corr_p, gold, option_ids, args.device)
            depths = {f"patch_l{int(t * 100)}": first_at_least(rec, t) for t in THRESHOLDS}
            rows.append({**base, "ld_clean": ldc, "ld_corrupt": ldx, "effect": ldc - ldx,
                         "valid": bool(ldc > 0 and (ldc - ldx) >= args.min_effect),
                         "recovery": rec.astype(np.float32), "ld_patched": ldp.astype(np.float32), **depths,
                         **{f"{k}_frac": (None if v is None else v / L) for k, v in depths.items()},
                         "clean_prompt": clean_p, "corrupt_prompt": corr_p, "note": ""})
        df = pd.DataFrame(rows)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        print(f"{path.name}: {len(df)} questions, {int(df.valid.sum())} valid, {time.time() - t0:.0f}s")
    res = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("shard_*.parquet"))])
    v = res[res.valid]
    print(f"{args.model}: {len(res)} questions, {len(v)} valid (clean correct, effect >= {args.min_effect} logits)")
    if len(v):
        for t in THRESHOLDS:
            c = f"patch_l{int(t * 100)}"
            print(f"  recovery >= {t}: median layer {v[c].median()} of {L} (defined for {v[c].notna().sum()})")


if __name__ == "__main__":
    main()

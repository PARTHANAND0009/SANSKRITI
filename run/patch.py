"""Stage 3b: activation patching (denoising) as a causal check on lens depth.

The corrupted prompt replaces the entity span in the stem with an entity of the same
attribute from a different state (swap pool, seeded per question); the options stay. For
each readout layer k the corrupted run is repeated with the final-token residual at layer k
copied from the clean run; all L+1 patched runs go in one batch.

LD = logit(gold letter) - max logit(other letters), restricted to A-D.
recovery_k = (LD_patched_k - LD_corrupt) / (LD_clean - LD_corrupt); patching depth at
threshold t is the first k with recovery_k >= t. Rows where the corruption moves LD by less
than --min-effect logits, or the clean answer is wrong, are kept with valid = False.

Eligible questions: analysis set, with a stem span, a frequency tier and enough swap
candidates. Country Prediction (the answer is always India) and stems that name a state
(the state cue survives the swap) are excluded because a swap cannot change the answer.

--build-sample: strata are attribute x frequency tier over all eligible states. Each stratum
gets an equal quota, with leftovers redistributed, and questions are taken round-robin
across states; the anchor states get a minimum. --grid keeps the earlier 3 attributes x 12
states x 2 tiers builder.

  python -m run.patch --build-sample
  python -m run.patch --model llama31_8b --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch

from crystal.hooks import record_residuals, stacked
from crystal.io import (
    load_analysis_set,
    load_entities,
    load_freq,
    load_prompts,
    load_run_config,
    model_config,
    resolve,
    set_seed,
    stable_seed,
)
from crystal.lens import decoder_layers
from crystal.log import setup_logging
from crystal.models import load_model_and_tokenizer, resolve_option_ids
from data.entities import state_name_regex
from data.prep import build_prompt, option_order

log = logging.getLogger(__name__)

PATCH_CFG = load_run_config()["patching"]
THRESHOLDS = tuple(PATCH_CFG["thresholds"])
MIN_SWAP_CANDIDATES = PATCH_CFG["min_swap_candidates"]
ANCHOR_STATES = tuple(PATCH_CFG["anchor_states"])
EXCLUDED_QUESTION_TYPES = ("Country Prediction",)
SAMPLE_COLS = [
    "qid",
    "state",
    "attribute",
    "question_type",
    "tier",
    "log_freq",
    "entity",
    "span_start",
    "span_end",
    "n_swap_candidates",
]


def eligible_questions() -> pd.DataFrame:
    an = load_analysis_set()
    ent = load_entities()
    fr = load_freq()[["qid", "log_freq", "tier"]]
    pool = pd.read_parquet(resolve("data/processed/swap_pool.parquet"))
    pool = pool[pool.swap_ok]
    df = an.merge(ent, on="qid").merge(fr, on="qid")
    df = df[df.span_start.notna() & df.tier.notna()].copy()
    df = df[~df.question_type.isin(EXCLUDED_QUESTION_TYPES)]
    rx = state_name_regex(an.state.unique())
    df = df[~df.stem.map(lambda s: bool(rx.search(s)))]
    per_attr = pool.groupby("attribute").size()
    per_attr_state = pool.groupby(["attribute", "state"]).size()
    df["n_swap_candidates"] = [
        int(per_attr.get(a, 0) - per_attr_state.get((a, s), 0)) for a, s in zip(df.attribute, df.state)
    ]
    attr_min = df.groupby("attribute").n_swap_candidates.min()
    keep_attr = attr_min[attr_min >= MIN_SWAP_CANDIDATES].index
    return df[df.attribute.isin(keep_attr)]


def _state_round_robin(g: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Order a stratum so consecutive rows cycle through states; its first n rows spread over states."""
    g = g.sample(frac=1, random_state=seed)
    g = g.assign(
        _k=g.groupby("state").cumcount(), _s=g.state.map({s: i for i, s in enumerate(sorted(g.state.unique()))})
    )
    return g.sort_values(["_k", "_s"]).drop(columns=["_k", "_s"])


def allocate(sizes: dict, total: int) -> dict:
    """Equal quota per stratum, capped at its size, with the remainder redistributed (water-filling)."""
    alloc, left, open_ = {k: 0 for k in sizes}, total, set(sizes)
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for k in sorted(open_):
            take = min(share, sizes[k] - alloc[k], left)
            alloc[k] += take
            left -= take
        open_ = {k for k in open_ if alloc[k] < sizes[k]}
    return alloc


def build_sample_strata(target, seed, anchors=ANCHOR_STATES, anchor_min=PATCH_CFG["anchor_min"]):
    """Attribute x tier strata over all eligible states; anchor states get at least `anchor_min`.

    Returns:
        (sample, per-stratum counts with the eligible size and quota)
    """
    df = eligible_questions()
    sizes = df.groupby(["attribute", "tier"]).size().to_dict()
    alloc = allocate(sizes, target)
    chosen = []
    for a in anchors:  # anchors first, round-robin over their strata, counted in the quotas
        g = df[df.state == a]
        if g.empty:
            continue
        g = g.sample(frac=1, random_state=seed)
        g = g.assign(_k=g.groupby(["attribute", "tier"]).cumcount()).sort_values(["_k", "attribute", "tier"])
        chosen.append(g.drop(columns="_k").head(anchor_min))
    pre = pd.concat(chosen) if chosen else df.iloc[:0]
    parts = [pre]
    for (a, t), n in alloc.items():
        g = df[(df.attribute == a) & (df.tier == t) & ~df.qid.isin(pre.qid)]
        already = int(((pre.attribute == a) & (pre.tier == t)).sum())
        parts.append(_state_round_robin(g, seed).head(max(0, n - already)))
    sample = pd.concat(parts).drop_duplicates("qid").sort_values("qid")
    cells = (
        sample.groupby(["attribute", "tier"])
        .size()
        .rename("n")
        .reset_index()
        .merge(
            pd.Series(sizes, name="eligible").rename_axis(["attribute", "tier"]).reset_index(), on=["attribute", "tier"]
        )
    )
    cells = cells.merge(
        pd.Series(alloc, name="quota").rename_axis(["attribute", "tier"]).reset_index(), on=["attribute", "tier"]
    )
    return sample[SAMPLE_COLS], cells


def build_sample_grid(seed, n_attr=3, n_states=12, per_cell=20, attributes=None, states=None):
    """Earlier builder: the n_attr largest attributes x the n_states states filling most cells x 2 tiers."""
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
    parts = [g.sample(min(per_cell, len(g)), random_state=seed) for _, g in d.groupby(["state", "attribute", "tier"])]
    sample = pd.concat(parts).sort_values("qid")
    counts = (
        sample.groupby(["attribute", "state", "tier"])
        .size()
        .reindex(
            pd.MultiIndex.from_product([attrs, sorted(sts), ["low", "high"]], names=["attribute", "state", "tier"]),
            fill_value=0,
        )
        .rename("n")
        .reset_index()
    )
    return sample[SAMPLE_COLS], counts


def pick_swap(qid: str, attribute: str, state: str, entity: str, pool: pd.DataFrame, seed: int):
    cand = pool[
        (pool.attribute == attribute)
        & (pool.state != state)
        & pool.swap_ok
        & (pool.entity.str.lower() != str(entity).lower())
    ].sort_values(["state", "entity"])
    if cand.empty:
        return None, None
    r = cand.iloc[random.Random(stable_seed(f"swap:{qid}", seed)).randrange(len(cand))]
    return r.entity, r.state


def clean_and_corrupt(row, swap_entity: str, variant: str, base_seed: int):
    order = option_order(variant, int(row.gold_idx), row.qid, base_seed)
    opts = [row.options[i] for i in order]
    s, e = int(row.span_start), int(row.span_end)
    corrupt_stem = row.stem[:s] + swap_entity + row.stem[e:]
    return build_prompt(row.stem, opts), build_prompt(corrupt_stem, opts), order.index(int(row.gold_idx))


def _encode(tok, prompt, device):
    enc = tok([prompt], return_tensors="pt")
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def final_token_resid(model, ids, mask):
    """Final-token residual [L+1, d] of a single prompt, and its final logits [1, V]."""
    with record_residuals(model, lambda h: h[:, -1].detach().clone()) as store:
        out = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    return torch.cat(stacked(store, len(decoder_layers(model)))), out.logits[:, -1]


@torch.no_grad()
def patched_logits(model, ids, mask, clean_resid: torch.Tensor):
    """Final logits [L+1, V] of L+1 corrupted runs; run k gets clean_resid[k] at layer k's final position."""
    layers = decoder_layers(model)
    n = len(layers) + 1
    ids_b, mask_b = ids.expand(n, -1).contiguous(), mask.expand(n, -1).contiguous()

    def pre(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        h = h.clone()
        h[0, -1] = clean_resid[0].to(h.dtype)
        if args:
            return (h, *args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = h
        return args, kwargs

    def post(k):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h.clone()
            h[k, -1] = clean_resid[k].to(h.dtype)
            return (h, *out[1:]) if isinstance(out, tuple) else h

        return f

    handles = [layers[0].register_forward_pre_hook(pre, with_kwargs=True)]
    handles += [layer.register_forward_hook(post(i + 1)) for i, layer in enumerate(layers)]
    try:
        out = model(input_ids=ids_b, attention_mask=mask_b, use_cache=False, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    return out.logits[:, -1]


def logit_diff(opt_logits: torch.Tensor, gold: int) -> torch.Tensor:
    """Gold minus the best other option: [..., 4] -> [...]."""
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


def main(argv=None):
    setup_logging()
    ap = argparse.ArgumentParser(description="stage 3b: activation patching")
    ap.add_argument("--build-sample", action="store_true")
    ap.add_argument(
        "--grid", action="store_true", help="earlier builder: 3 attributes x 12 states x 2 tiers x --per-cell"
    )
    ap.add_argument("--target", type=int, default=PATCH_CFG["target"], help="strata builder: total questions")
    ap.add_argument("--anchor-min", type=int, default=PATCH_CFG["anchor_min"])
    ap.add_argument("--attributes", nargs="*", default=None)
    ap.add_argument("--states", nargs="*", default=None)
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--model")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variant", default="prompt", help="option order of clean and corrupted prompts")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--min-effect", type=float, default=PATCH_CFG["min_effect"], help="min LD_clean - LD_corrupt for valid rows"
    )
    ap.add_argument("--shard-size", type=int, default=PATCH_CFG["shard_size"])
    ap.add_argument("--cpu-layers", type=int, default=None)
    ap.add_argument("--allow-proxy", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    cfg = load_run_config()
    set_seed(cfg["seed"])
    proc = resolve(cfg["paths"]["processed"])

    if args.build_sample and args.grid:
        sample, counts = build_sample_grid(
            cfg["seed"], per_cell=args.per_cell, attributes=args.attributes, states=args.states
        )
        sample.to_csv(proc / "patch_sample.csv", index=False)
        counts.to_csv(proc / "patch_sample_cells.csv", index=False)
        pv = counts.pivot_table(index="state", columns=["attribute", "tier"], values="n")
        log.info(
            f"patch sample (grid): {len(sample)} questions (target {len(counts)} cells x {args.per_cell} = "
            f"{len(counts) * args.per_cell}); cells below target: {(counts.n < args.per_cell).sum()}"
        )
        log.info(pv.to_string())
        if not args.model:
            return
    elif args.build_sample:
        sample, cells = build_sample_strata(args.target, cfg["seed"], anchor_min=args.anchor_min)
        sample.to_csv(proc / "patch_sample.csv", index=False)
        cells.to_csv(proc / "patch_sample_cells.csv", index=False)
        by_state = sample.groupby(["state", "tier"]).size().unstack(fill_value=0)
        by_state.to_csv(proc / "patch_sample_states.csv")
        log.info(
            f"patch sample (strata: attribute x tier, all eligible states): {len(sample)} questions "
            f"(target {args.target}), {sample.state.nunique()} states, {sample.attribute.nunique()} attributes"
        )
        log.info(cells.to_string(index=False))
        log.info("\nper state (tier):\n" + by_state.assign(total=by_state.sum(axis=1)).sort_values("total").to_string())
        if not args.model:
            return

    mcfg = model_config(args.model)
    if mcfg.get("proxy") and not args.allow_proxy:
        raise SystemExit(f"{args.model} is a proxy model; use --allow-proxy for tests")
    sample = pd.read_csv(proc / "patch_sample.csv")
    if args.limit:
        sample = sample.head(args.limit)
    prompts = load_prompts().set_index("qid")
    pool = pd.read_parquet(proc / "swap_pool.parquet")
    model, tok = load_model_and_tokenizer(mcfg, args.device, cpu_layers=args.cpu_layers)
    option_ids = resolve_option_ids(mcfg, tok)
    L = len(decoder_layers(model))
    out_dir = resolve(args.out) / f"patch_{args.model}_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "model_id": mcfg["id"],
                "proxy": bool(mcfg.get("proxy")),
                "variant": args.variant,
                "L": L,
                "min_effect": args.min_effect,
                "thresholds": THRESHOLDS,
                "n_sample": len(sample),
                "option_ids": option_ids,
                "device": args.device,
            },
            indent=2,
        )
    )
    for si in range(0, len(sample), args.shard_size):
        path = out_dir / f"shard_{si // args.shard_size:05d}.parquet"
        part = sample.iloc[si : si + args.shard_size]
        if path.exists() and pd.read_parquet(path).qid.tolist() == part.qid.tolist():
            log.info(f"{path.name}: exists, skipped")
            continue
        rows, t0 = [], time.time()
        for r in part.itertuples():
            row = prompts.loc[r.qid].copy()
            row["qid"] = r.qid
            row["span_start"], row["span_end"] = r.span_start, r.span_end
            swap, swap_state = pick_swap(r.qid, r.attribute, r.state, r.entity, pool, cfg["seed"])
            base = {
                "qid": r.qid,
                "state": r.state,
                "attribute": r.attribute,
                "tier": r.tier,
                "entity": r.entity,
                "swap_entity": swap,
                "swap_state": swap_state,
                "L": L,
            }
            if swap is None:
                rows.append({**base, "valid": False, "note": "no swap candidate"})
                continue
            clean_p, corr_p, gold = clean_and_corrupt(row, swap, args.variant, cfg["prep"]["permute_seed"])
            ldc, ldx, ldp, rec = patch_question(model, tok, clean_p, corr_p, gold, option_ids, args.device)
            depths = {f"patch_l{int(t * 100)}": first_at_least(rec, t) for t in THRESHOLDS}
            rows.append(
                {
                    **base,
                    "ld_clean": ldc,
                    "ld_corrupt": ldx,
                    "effect": ldc - ldx,
                    "valid": bool(ldc > 0 and (ldc - ldx) >= args.min_effect),
                    "recovery": rec.astype(np.float32),
                    "ld_patched": ldp.astype(np.float32),
                    **depths,
                    **{f"{k}_frac": (None if v is None else v / L) for k, v in depths.items()},
                    "clean_prompt": clean_p,
                    "corrupt_prompt": corr_p,
                    "note": "",
                }
            )
        df = pd.DataFrame(rows)
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        log.info(f"{path.name}: {len(df)} questions, {int(df.valid.sum())} valid, {time.time() - t0:.0f}s")
    res = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("shard_*.parquet"))])
    v = res[res.valid]
    log.info(f"{args.model}: {len(res)} questions, {len(v)} valid (clean correct, effect >= {args.min_effect} logits)")
    if len(v):
        for t in THRESHOLDS:
            c = f"patch_l{int(t * 100)}"
            log.info(f"  recovery >= {t}: median layer {v[c].median()} of {L} (defined for {v[c].notna().sum()})")


if __name__ == "__main__":
    main()

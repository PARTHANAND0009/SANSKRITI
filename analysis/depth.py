"""Stage 4a: crystallization depth per (model, question, variant), and across variants.

Readout layers are 0..L (0 = embeddings, k = output of block k); a depth in layers
is divided by L to give d. Readouts: logit lens, or tuned lens (--readout tuned,
translators from run/tune_lens.py).

Per-variant metrics (one prompt order; `table` subcommand writes them):
  l_star      original definition: first layer where the gold letter is top-1 under
              the A-D restricted softmax and stays top-1 through layer L.
  l_star_cal  same after removing the layer's letter prior: per layer, subtract each
              letter's mean log-prob. The prior must come from rows whose gold letters are
              balanced (pooled over the 4 rotations; in a single cyclic variant every gold
              is the same letter and its own prior would remove the answer signal), so a
              lone unbalanced table leaves these columns null.
  d_soft_cal  expected layer under the normalised positive increments of the gold
              probability across layers, using the calibrated probabilities
              (softmax of the calibrated log-probs). Continuous; robust to one-layer flips.
  (On a single order the calibrated gold-minus-best-distractor margin is > 0 exactly
  when gold is calibrated top-1, so the single-order margin depth equals l_star_cal.)

Across variants (`aggregate` subcommand; normally the 4 cyclic rotations), with
option probabilities aligned by answer content, not letter:
  l_star_mean, l_star_cal_mean   mean of the per-variant values (null unless all defined)
  l_star_cyc  first layer where the gold content is top-1, and stays, under the
              content probabilities averaged across variants (letter bias cancels when
              every content appears at every letter once, i.e. the 4 rotations)
  d_margin    first layer where the across-variant mean of the calibrated gold-minus-
              best-distractor log-prob margin is > 0 and stays > 0
  d_soft      expected layer (as above) on the across-variant mean gold content prob
Columns named l_* / d_* hold layers; the d_frac_* columns divide by L.

Caveat (pilot finding): uncalibrated early layers rank one default letter first for
every question, so single-order l_star is early for questions whose gold is that
letter. Use the 4-rotation aggregate.

  python -m analysis.depth table --model llama31_8b --variant cyc0 [--cpu-layers 0]
  python -m analysis.depth aggregate --model llama31_8b --variants cyc0 cyc1 cyc2 cyc3
  python -m analysis.depth reliability --model llama31_8b --variants prompt prompt_permuted
  python -m analysis.depth split-half --model llama31_8b      # cyc0+cyc2 vs cyc1+cyc3
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from crystal.io import (load_acts, load_prompts, load_run_config, model_config, resolve, set_seed,
                        variant_columns)
from crystal.lens import crystallization_layer, gold_rank, lens_option_logits


# ------------------------------------------------------------------ primitives

MAX_GOLD_LETTER_SHARE = 0.4   # a table's own letter prior is used only if gold letters are this balanced


def letter_prior(*prob_arrays: np.ndarray) -> np.ndarray:
    """Per-layer mean log-prob of each letter, pooled over all rows given: [L+1, 4].
    Must be estimated on rows whose gold letters are balanced (e.g. the 4 cyclic
    rotations pooled); otherwise it absorbs the answer signal itself (in cyc1 every
    gold is B, so B's mean includes the gold boost)."""
    lp = np.concatenate([np.log(p + 1e-12) for p in prob_arrays])
    return lp.mean(0)


def gold_letters_balanced(gold: np.ndarray) -> bool:
    return np.bincount(gold.astype(np.int64), minlength=4).max() / len(gold) <= MAX_GOLD_LETTER_SHARE


def calibrate(probs: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
    """probs [n, L+1, 4] -> calibrated log-probs (log-prob minus the letter prior,
    default: the prior of these rows)."""
    lp = np.log(probs + 1e-12)
    return lp - (lp.mean(0) if prior is None else prior)[None]


def letter_prior_calibrated(probs: np.ndarray, gold: np.ndarray, prior=None):
    """(l_star_cal list, calibrated top1_is_gold [n, L+1])."""
    top = calibrate(probs, prior).argmax(-1) == gold[:, None]
    return [crystallization_layer(r) for r in top], top


def first_stays(flags_2d: np.ndarray) -> list:
    return [crystallization_layer(r) for r in flags_2d]


def soft_layer(p_gold: np.ndarray) -> np.ndarray:
    """p_gold [n, L+1] -> expected layer under the normalised positive increments
    p[l] - p[l-1] (l = 1..L); NaN when the gold probability never increases."""
    inc = np.clip(np.diff(p_gold, axis=1), 0, None)
    tot = inc.sum(1)
    layers = np.arange(1, p_gold.shape[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        e = (inc * layers).sum(1) / tot
    return np.where(tot > 0, e, np.nan)


def softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def to_int_array(xs) -> pd.arrays.IntegerArray:
    return pd.array(xs, dtype="Int64")


# ------------------------------------------------------------------ per variant

@torch.no_grad()
def option_probs(model, acts: np.ndarray, option_ids, translator=None, batch: int = 256) -> np.ndarray:
    """acts [n, L+1, d] -> restricted A-D probabilities [n, L+1, 4]."""
    n, n_read, _ = acts.shape
    probs = np.empty((n, n_read, 4), dtype=np.float32)
    for s in range(0, n, batch):
        a = torch.from_numpy(acts[s:s + batch])
        for l in range(n_read):
            probs[s:s + batch, l] = torch.softmax(
                lens_option_logits(model, a[:, l], l, option_ids, translator=translator).float(), -1).cpu().numpy()
    return probs


def table_from_probs(qids, probs: np.ndarray, gold: np.ndarray, prior: np.ndarray | None = None) -> pd.DataFrame:
    """prior: pooled letter prior [L+1, 4]. If None, the table's own prior is used when
    its gold letters are balanced; otherwise (e.g. a single cyclic variant, where every
    gold is the same letter) the calibrated columns are left null."""
    n, n_read, _ = probs.shape
    L = n_read - 1
    gold = gold.astype(np.int64)
    p = torch.from_numpy(probs)
    g = torch.from_numpy(gold)
    ranks = gold_rank(p, g[:, None].expand(-1, n_read)).numpy()
    gold_p = np.take_along_axis(probs, gold[:, None, None], -1)[..., 0]
    lstar = first_stays(ranks == 0)
    if prior is None and not gold_letters_balanced(gold):
        lstar_cal, top_cal_final, soft = [None] * n, np.full(n, np.nan), np.full(n, np.nan)
    else:
        cal = calibrate(probs, prior)
        top_cal = cal.argmax(-1) == gold[:, None]
        lstar_cal, top_cal_final = first_stays(top_cal), top_cal[:, -1]
        p_cal_gold = np.take_along_axis(softmax_last(cal), gold[:, None, None], -1)[..., 0]
        soft = soft_layer(p_cal_gold)
    return pd.DataFrame({
        "qid": list(qids),
        "L": L,
        "l_star": to_int_array(lstar),
        "d": [None if x is None else x / L for x in lstar],
        "correct_final": ranks[:, -1] == 0,
        "l_star_cal": to_int_array(lstar_cal),
        "d_cal": [None if x is None else x / L for x in lstar_cal],
        "correct_final_cal": top_cal_final,
        "d_soft_cal": soft,
        "gold_prob_final": gold_p[:, -1],
        "gold_prob_by_layer": list(gold_p.astype(np.float32)),
        "gold_rank_by_layer": list(ranks.astype(np.int8)),
        # restricted option probabilities per layer, flattened [(L+1)*4] (A..D per layer)
        "option_probs_by_layer": list(probs.reshape(n, -1)),
    })


def depth_table(model, qids, acts: np.ndarray, gold: np.ndarray, option_ids, translator=None, prior=None) -> pd.DataFrame:
    return table_from_probs(qids, option_probs(model, acts, option_ids, translator), gold, prior)


def probs_of(tab: pd.DataFrame) -> np.ndarray:
    P = np.stack(tab.option_probs_by_layer.to_numpy())
    return P.reshape(len(tab), -1, 4)


def recompute_from_probs(tab: pd.DataFrame, gold: np.ndarray, prior=None) -> pd.DataFrame:
    """Rebuild the per-variant metrics of a stored table (e.g. one written before a
    metric was added) from its option_probs_by_layer, without the model."""
    keep = [c for c in ("state", "attribute", "question_type") if c in tab.columns]
    new = table_from_probs(tab.qid.tolist(), probs_of(tab), gold, prior)
    for i, c in enumerate(keep):
        new.insert(1 + i, c, tab[c].to_numpy())
    return new


# ------------------------------------------------------------------ across variants

def content_aligned(tab: pd.DataFrame, variant: str, prompts: pd.DataFrame, base_seed: int) -> np.ndarray:
    """[n, L+1, 4] with the last axis indexed by original option index (content)."""
    from data.prep import option_order

    P = probs_of(tab)
    out = np.empty_like(P)
    pr = prompts.loc[tab.qid]
    for i, (qid, g) in enumerate(zip(tab.qid, pr.gold_idx)):
        order = option_order(variant, int(g), qid, base_seed)   # letter j shows content order[j]
        out[i][:, order] = P[i]
    return out


def aggregate(tables: dict, prompts: pd.DataFrame, base_seed: int) -> pd.DataFrame:
    """tables: variant -> per-variant table (same model). Inner join on qid."""
    variants = list(tables)
    qids = sorted(set.intersection(*[set(t.qid) for t in tables.values()]))
    T = {v: t.set_index("qid").loc[qids].reset_index() for v, t in tables.items()}
    L = int(T[variants[0]].L.iloc[0]) if "L" in T[variants[0]] else probs_of(T[variants[0]]).shape[1] - 1
    gold = prompts.loc[qids, "gold_idx"].to_numpy().astype(np.int64)
    prior = letter_prior(*[probs_of(T[v]) for v in variants])                              # pooled, balanced
    from data.prep import option_order
    for v in variants:                                                                     # per-variant cal metrics, pooled prior
        g_v = np.array([option_order(v, int(g), q, base_seed).index(int(g)) for q, g in zip(qids, gold)])
        T[v] = recompute_from_probs(T[v], g_v, prior)
    Pc = np.stack([content_aligned(T[v], v, prompts, base_seed) for v in variants])        # [V, n, L+1, 4]
    cal = np.stack([calibrate(probs_of(T[v]), prior) for v in variants])                   # letter space
    calc = np.empty_like(cal)
    for k, v in enumerate(variants):                                                       # -> content space
        for i, (qid, g) in enumerate(zip(qids, gold)):
            calc[k, i][:, option_order(v, int(g), qid, base_seed)] = cal[k, i]
    n = len(qids)
    mean_p = Pc.mean(0)
    top_cyc = mean_p.argmax(-1) == gold[:, None]
    gold_c = np.take_along_axis(calc, gold[None, :, None, None].repeat(len(variants), 0), -1)[..., 0]
    other = calc.copy()
    np.put_along_axis(other, gold[None, :, None, None].repeat(len(variants), 0), -np.inf, -1)
    margin = (gold_c - other.max(-1)).mean(0)                                              # [n, L+1]
    p_gold_mean = np.take_along_axis(mean_p, gold[:, None, None], -1)[..., 0]

    def mean_or_null(col):
        vals = np.stack([T[v][col].astype("Float64").to_numpy(dtype=float, na_value=np.nan) for v in variants])
        return np.where(np.isnan(vals).any(0), np.nan, vals.mean(0))

    lcyc = first_stays(top_cyc)
    lmar = first_stays(margin > 0)
    soft = soft_layer(p_gold_mean)
    out = pd.DataFrame({
        "qid": qids, "L": L, "n_variants": len(variants), "variants": ",".join(variants),
        "acc": np.mean([T[v].correct_final.to_numpy() for v in variants], 0),
        "correct_all": np.all([T[v].correct_final.to_numpy() for v in variants], 0),
        "correct_cyc_final": top_cyc[:, -1],
        "l_star_mean": mean_or_null("l_star"),
        "l_star_cal_mean": mean_or_null("l_star_cal"),
        "l_star_cyc": to_int_array(lcyc),
        "d_margin": to_int_array(lmar),
        "d_soft": soft,
        "gold_prob_mean_final": p_gold_mean[:, -1],
    })
    for c in ("l_star_mean", "l_star_cal_mean", "l_star_cyc", "d_margin", "d_soft"):
        out[f"d_frac_{c}"] = out[c].astype("Float64") / L
    return out


# ------------------------------------------------------------------ reliability

SINGLE_ORDER_METRICS = ("l_star", "l_star_cal", "d_soft_cal")


def reliability(a: pd.DataFrame, b: pd.DataFrame, metrics=SINGLE_ORDER_METRICS, subset: str = "correct_both") -> pd.DataFrame:
    """Test-retest between two orders of the same questions. subset: 'correct_both'
    (questions correct at layer L in both orders; same rows for every metric) or 'defined'."""
    j = a.merge(b, on="qid", suffixes=("_a", "_b"))
    if subset == "correct_both":
        j = j[j.correct_final_a & j.correct_final_b]
    rows = []
    for m in metrics:
        x = j[f"{m}_a"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        y = j[f"{m}_b"].astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        ok = ~(np.isnan(x) | np.isnan(y))
        x, y = x[ok], y[ok]
        rho = spearmanr(x, y)[0] if len(x) > 2 else np.nan
        rows.append({"metric": m, "subset": subset, "n": int(ok.sum()), "spearman": rho,
                     "within_1_layer": float(np.mean(np.abs(x - y) <= 1)) if len(x) else np.nan,
                     "identical": float(np.mean(x == y)) if len(x) else np.nan,
                     "mean_abs_diff_layers": float(np.mean(np.abs(x - y))) if len(x) else np.nan,
                     "sd_layers": float(np.std(np.r_[x, y])) if len(x) else np.nan})
    return pd.DataFrame(rows)


AGG_METRICS = ("l_star_mean", "l_star_cal_mean", "l_star_cyc", "d_margin", "d_soft")


def split_half(tables: dict, prompts: pd.DataFrame, base_seed: int, halves=(("cyc0", "cyc2"), ("cyc1", "cyc3"))) -> pd.DataFrame:
    """Reliability of the across-variant metrics: aggregate each half of the rotations
    separately, correlate per question (correct in both halves), and project to the
    full set of rotations with Spearman-Brown (k = 2). Each half holds 2 rotations, so
    the letter prior cancels only partly within a half: the estimate is conservative
    for metrics on raw probabilities (l_star_cyc, d_soft)."""
    a = aggregate({v: tables[v] for v in halves[0]}, prompts, base_seed)
    b = aggregate({v: tables[v] for v in halves[1]}, prompts, base_seed)
    a = a.assign(correct_final=a.correct_cyc_final)
    b = b.assign(correct_final=b.correct_cyc_final)
    r = reliability(a, b, metrics=AGG_METRICS)
    r["halves"] = f"{'+'.join(halves[0])} vs {'+'.join(halves[1])}"
    r["spearman_brown_full"] = [spearman_brown(x, 2) for x in r.spearman]
    return r


def spearman_brown(r: float, k: int) -> float:
    """Projected reliability of the mean of k parallel orders from single-order r."""
    return k * r / (1 + (k - 1) * r)


# ------------------------------------------------------------------ CLI

def _load_table(out, model, variant, readout):
    return pd.read_parquet(out / f"depth_{model}_{variant}_{readout}.parquet")


def cmd_table(args, cfg):
    from run.forward import load_model_and_tokenizer

    mcfg = model_config(args.model)
    if mcfg.get("proxy") and not args.allow_proxy:
        raise SystemExit(f"{args.model} is a proxy model; refusing (use --allow-proxy for tests)")
    acts_dir = resolve(cfg["paths"]["acts"]) / args.model / args.variant
    meta = json.loads((acts_dir / "meta.json").read_text())
    qids, acts, _ = load_acts(args.model, args.variant)
    prompts = load_prompts().set_index("qid").loc[list(qids)]
    gold_col = variant_columns(args.variant)[1]
    model, _ = load_model_and_tokenizer(mcfg, args.device, cpu_layers=args.cpu_layers)
    translator = None
    if args.readout == "tuned":
        from run.tune_lens import load_tuned_lens
        translator = load_tuned_lens(args.lens or resolve("lenses") / f"{args.model}.pt", device=args.device)
    tab = depth_table(model, qids, acts, prompts[gold_col].to_numpy(), meta["option_ids"], translator)
    for i, c in enumerate(("state", "attribute", "question_type")):
        tab.insert(1 + i, c, prompts[c].to_numpy())
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"depth_{args.model}_{args.variant}_{args.readout}"
    tab.to_parquet(out / f"{stem}.parquet", index=False)
    (out / f"{stem}.meta.json").write_text(json.dumps({**meta, "readout": args.readout}, indent=2))
    L = int(tab.L.iloc[0])
    c = tab[tab.correct_final]
    print(f"{args.model} [{args.variant}, {args.readout}] n={len(tab)} L={L} accuracy {tab.correct_final.mean():.3f}; "
          f"correct: l_star median {c.l_star.median()}, l_star_cal median {c.l_star_cal.median()}, "
          f"d_soft_cal median {np.nanmedian(c.d_soft_cal):.2f}")
    print(f"wrote {out / stem}.parquet")


def cmd_aggregate(args, cfg):
    out = resolve(args.out)
    prompts = load_prompts().set_index("qid")
    tables = {v: _load_table(out, args.model, v, args.readout) for v in args.variants}
    agg = aggregate(tables, prompts, cfg["prep"]["permute_seed"])
    meta = prompts.loc[agg.qid, ["state", "attribute", "question_type"]].reset_index(drop=True)
    agg = pd.concat([agg.iloc[:, :1], meta, agg.iloc[:, 1:]], axis=1)
    name = args.name or ("cyc" if set(args.variants) == {"cyc0", "cyc1", "cyc2", "cyc3"} else "+".join(args.variants))
    path = out / f"depth_{args.model}_{name}_{args.readout}_agg.parquet"
    agg.to_parquet(path, index=False)
    c = agg[agg.correct_cyc_final]
    print(f"{args.model} aggregate over {args.variants} ({args.readout}): n={len(agg)}, content-averaged "
          f"accuracy {agg.correct_cyc_final.mean():.3f}")
    for m in ("l_star_mean", "l_star_cal_mean", "l_star_cyc", "d_margin", "d_soft"):
        v = c[m].astype("Float64").dropna().astype(float)
        print(f"  {m:<16} n={len(v):5d} median {v.median():.2f}  mean {v.mean():.2f}  (layers, of L={int(agg.L.iloc[0])})")
    print(f"wrote {path}")


def cmd_reliability(args, cfg):
    out = resolve(args.out)
    a, b = (_load_table(out, args.model, v, args.readout) for v in args.variants)
    prompts = load_prompts().set_index("qid")
    # rebuild per-variant metrics from stored probabilities so older tables get every metric
    prior = letter_prior(probs_of(a), probs_of(b))   # pooled over both orders
    a = recompute_from_probs(a, prompts.loc[a.qid, variant_columns(args.variants[0])[1]].to_numpy(), prior)
    b = recompute_from_probs(b, prompts.loc[b.qid, variant_columns(args.variants[1])[1]].to_numpy(), prior)
    res = pd.concat([reliability(a, b, subset=s) for s in ("correct_both", "defined")])
    res.insert(0, "model", args.model)
    res.insert(1, "orders", " vs ".join(args.variants))
    res["spearman_brown_4_orders"] = [spearman_brown(r, 4) for r in res.spearman]
    path = out / f"reliability_{args.model}_{'_vs_'.join(args.variants)}_{args.readout}.csv"
    res.to_csv(path, index=False)
    print(res.round(3).to_string(index=False))
    print(f"wrote {path}")


def cmd_split_half(args, cfg):
    out = resolve(args.out)
    prompts = load_prompts().set_index("qid")
    tables = {v: _load_table(out, args.model, v, args.readout) for v in ("cyc0", "cyc1", "cyc2", "cyc3")}
    res = split_half(tables, prompts, cfg["prep"]["permute_seed"])
    res.insert(0, "model", args.model)
    path = out / f"reliability_{args.model}_splithalf_{args.readout}.csv"
    res.to_csv(path, index=False)
    print(res.round(3).to_string(index=False))
    print(f"wrote {path}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["table"] + argv            # backward compatible: no subcommand = table
    ap = argparse.ArgumentParser(description="stage 4a: crystallization depth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--model", required=True)
    t.add_argument("--variant", default="cyc0")
    t.add_argument("--device", default="cpu")
    t.add_argument("--cpu-layers", type=int, default=None)
    t.add_argument("--readout", choices=["logit", "tuned"], default="logit")
    t.add_argument("--lens", default=None, help="tuned lens file (default lenses/{model}.pt)")
    t.add_argument("--allow-proxy", action="store_true", help="testing only; proxy results are never analysis")
    a = sub.add_parser("aggregate")
    a.add_argument("--model", required=True)
    a.add_argument("--variants", nargs="+", default=["cyc0", "cyc1", "cyc2", "cyc3"])
    a.add_argument("--name", default=None)
    r = sub.add_parser("reliability")
    r.add_argument("--model", required=True)
    r.add_argument("--variants", nargs=2, default=["prompt", "prompt_permuted"])
    h = sub.add_parser("split-half", help="reliability of the 4-rotation metrics (needs cyc0..cyc3 tables)")
    h.add_argument("--model", required=True)
    for s in (a, r, h):
        s.add_argument("--readout", choices=["logit", "tuned"], default="logit")
    for s in (t, a, r, h):
        s.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    cfg = load_run_config()
    set_seed(cfg["seed"])
    {"table": cmd_table, "aggregate": cmd_aggregate, "reliability": cmd_reliability,
     "split-half": cmd_split_half}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()

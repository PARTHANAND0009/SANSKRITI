"""Stage 4a: crystallization depth per (model, question, readout).

Implemented: logit lens. Tuned lens: pending (needs stage 3a translators).

For each question in the stage 2 shards and each readout layer 0..L
(0 = embeddings, k = output of block k):
  restricted softmax over the model's A-D option tokens of lens_logits(...)
  -> gold rank. l* = crystallization_layer(rank == 0) (crystal.lens), d = l*/L.
Questions the model gets wrong at layer L have l* = None (kept, not dropped).

Output: results/depth_{model}_{variant}_logit.parquet with
  qid, l_star, d, correct_final, l_star_cal, d_cal, correct_final_cal (letter-prior
  calibrated secondary readout, see letter_prior_calibrated), gold_prob_final, gold_prob_by_layer (list),
  gold_rank_by_layer (list), option_probs_by_layer (flattened [(L+1)*4]), and the
  stage 2 run metadata in results/*.meta.json.

Caveat (pilot finding): in early layers the logit lens ranks one default letter
first for every question (B for Llama-3.1-8B and Qwen2.5-7B at layer 3), so
questions whose gold is that letter get a spuriously early l*. Compare against
the option-permuted variant or correct for the per-layer letter prior before
interpreting per-question l*.

  python -m analysis.depth --model llama31_8b [--variant prompt] [--cpu-layers 0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crystal.io import load_acts, load_prompts, load_run_config, model_config, resolve, set_seed
from crystal.lens import crystallization_layer, gold_rank, lens_option_logits


def letter_prior_calibrated(probs: np.ndarray, gold: np.ndarray):
    """Secondary readout. probs [n, L+1, 4]. Per layer, subtract each letter's mean
    log-probability over the n questions (the layer's letter prior), then take the
    top option. Returns (l_star_cal list, top1_is_gold [n, L+1]). The primary l*
    is unchanged; this column exists because uncalibrated early layers rank one
    default letter first for every question (pilot finding)."""
    lp = np.log(probs + 1e-12)
    cal = lp - lp.mean(0, keepdims=True)
    top = cal.argmax(-1) == gold[:, None]
    return [crystallization_layer(r) for r in top], top


@torch.no_grad()
def depth_table(model, qids, acts: np.ndarray, gold: np.ndarray, option_ids, batch: int = 256) -> pd.DataFrame:
    """acts [n, L+1, d] -> one row per question."""
    n, n_read, _ = acts.shape
    L = n_read - 1
    probs = np.empty((n, n_read, 4), dtype=np.float32)
    for s in range(0, n, batch):
        a = torch.from_numpy(acts[s:s + batch])
        for l in range(n_read):
            probs[s:s + batch, l] = torch.softmax(
                lens_option_logits(model, a[:, l], l, option_ids).float(), -1).cpu().numpy()
    p = torch.from_numpy(probs)
    g = torch.from_numpy(gold.astype(np.int64))
    ranks = gold_rank(p, g[:, None].expand(-1, n_read)).numpy()
    gold_p = p.gather(-1, g[:, None, None].expand(-1, n_read, 1))[..., 0].numpy()
    lstar = [crystallization_layer(r == 0) for r in ranks]
    lstar_cal, top_cal = letter_prior_calibrated(probs, gold.astype(np.int64))
    return pd.DataFrame({
        "qid": list(qids),
        "l_star": pd.array(lstar, dtype="Int64"),
        "d": [None if x is None else x / L for x in lstar],
        "correct_final": ranks[:, -1] == 0,
        "l_star_cal": pd.array(lstar_cal, dtype="Int64"),
        "d_cal": [None if x is None else x / L for x in lstar_cal],
        "correct_final_cal": top_cal[:, -1],
        "gold_prob_final": gold_p[:, -1],
        "gold_prob_by_layer": list(gold_p.astype(np.float32)),
        "gold_rank_by_layer": list(ranks.astype(np.int8)),
        # all four restricted option probabilities per layer, flattened [(L+1)*4], A..D
        # per layer; kept so letter-prior corrections can be computed without the model
        "option_probs_by_layer": list(probs.reshape(n, -1)),
    })


def main(argv=None):
    ap = argparse.ArgumentParser(description="stage 4a: logit-lens crystallization depth")
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", choices=["prompt", "prompt_permuted"], default="prompt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cpu-layers", type=int, default=None,
                    help="CPU: load with disk offload (0 = only embeddings/norm/head in RAM)")
    ap.add_argument("--allow-proxy", action="store_true", help="testing only; proxy results are never analysis")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    from run.forward import load_model_and_tokenizer

    cfg = load_run_config()
    set_seed(cfg["seed"])
    mcfg = model_config(args.model)
    if mcfg.get("proxy") and not args.allow_proxy:
        raise SystemExit(f"{args.model} is a proxy model; refusing (use --allow-proxy for tests)")
    acts_dir = resolve(cfg["paths"]["acts"]) / args.model / args.variant
    meta = json.loads((acts_dir / "meta.json").read_text())
    qids, acts, _ = load_acts(args.model, args.variant)
    prompts = load_prompts().set_index("qid").loc[list(qids)]
    gold_col = "gold_idx" if args.variant == "prompt" else "gold_idx_permuted"
    model, _ = load_model_and_tokenizer(mcfg, args.device, cpu_layers=args.cpu_layers)
    tab = depth_table(model, qids, acts, prompts[gold_col].to_numpy(), meta["option_ids"])
    tab.insert(1, "state", prompts["state"].to_numpy())
    tab.insert(2, "attribute", prompts["attribute"].to_numpy())
    tab.insert(3, "question_type", prompts["question_type"].to_numpy())

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"depth_{args.model}_{args.variant}_logit"
    tab.to_parquet(out / f"{stem}.parquet", index=False)
    (out / f"{stem}.meta.json").write_text(json.dumps({**meta, "readout": "logit_lens"}, indent=2))

    L = acts.shape[1] - 1
    c = tab[tab.correct_final]
    print(f"{args.model} [{args.variant}] n={len(tab)}  L={L}  final accuracy {tab.correct_final.mean():.3f}  "
          f"mean gold prob {tab.gold_prob_final.mean():.3f}")
    if len(c):
        print(f"correct at layer L: {len(c)}; l* median {int(c.l_star.median())}, "
              f"d mean {c.d.astype(float).mean():.3f}, median {c.d.astype(float).median():.3f}")
        print("d quartiles:", np.round(np.quantile(c.d.astype(float), [0.25, 0.5, 0.75]), 3).tolist())
        cc = tab[tab.correct_final_cal]
        print(f"letter-prior calibrated: final 'correct' {tab.correct_final_cal.mean():.3f}; l*_cal median "
              f"{int(cc.l_star_cal.median())}, d_cal mean {cc.d_cal.astype(float).mean():.3f}, quartiles "
              f"{np.round(np.quantile(cc.d_cal.astype(float), [0.25, 0.5, 0.75]), 3).tolist()}")
        print("\nby question_type (correct only):")
        print(c.groupby("question_type").agg(n=("qid", "size"), d_mean=("d", lambda s: s.astype(float).mean()))
              .round(3).to_string())
    print(f"\nwrote {out / stem}.parquet")


if __name__ == "__main__":
    main()

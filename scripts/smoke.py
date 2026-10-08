"""Smoke checks on stage 2 output, run after run/forward.py on the same rows.

Logs final-layer accuracy (model output and logit lens on the stored fp16 activations), the
mean gold probability, and per-layer gold ranks for a few questions, then asserts that the
lens at the last layer reproduces the model's output logits on live prompts.

  python scripts/smoke.py --model qwen25_05b_proxy --device cpu --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal.io import (
    VARIANTS,
    load_run_config,
    model_config,
    read_shards,
    resolve,
    select_rows,
    set_seed,
    variant_columns,
)
from crystal.lens import crystallization_layer, decoder_layers, gold_rank, lens_logits, lens_option_logits
from crystal.log import setup_logging
from crystal.models import load_model_and_tokenizer
from run.forward import left_pad_batch, position_ids_from_mask

log = logging.getLogger(__name__)


@torch.no_grad()
def layer_option_logits(model, acts: np.ndarray, option_ids) -> torch.Tensor:
    """Logit-lens option logits [n, L+1, 4] from stored activations."""
    a = torch.from_numpy(acts)
    per_layer = [lens_option_logits(model, a[:, layer], layer, option_ids).float().cpu() for layer in range(a.shape[1])]
    return torch.stack(per_layer, dim=1)


@torch.no_grad()
def live_lens_check(model, tok, prompts):
    """Assert that the lens at layer L reproduces the model's full-vocabulary output logits.

    The lens gets exactly the tensor the model's head sees (the last block's output over the
    whole sequence), so default assert_close tolerances apply. On the single sliced position
    that stage 2 stores, the RMSNorm reduction order differs; that gap (about 1e-5 in fp32
    on the proxy) is logged, not asserted.
    """
    ids, mask = left_pad_batch(tok, prompts)
    dev = next(model.parameters()).device
    store = {}
    h = decoder_layers(model)[-1].register_forward_hook(
        lambda m, i, o: store.__setitem__("r", o[0] if isinstance(o, tuple) else o)
    )
    try:
        out = model(
            input_ids=ids.to(dev),
            attention_mask=mask.to(dev),
            position_ids=position_ids_from_mask(mask).to(dev),
            use_cache=False,
            logits_to_keep=0,
        )
    finally:
        h.remove()
    L = len(decoder_layers(model))
    ref = out.logits[:, -1]
    lens = lens_logits(model, store["r"], L)[:, -1]
    sliced = lens_logits(model, store["r"][:, -1], L)
    log.info(
        f"live check: max |lens - output| over full vocab = {(lens.float() - ref.float()).abs().max().item():.3g} "
        f"(same input); {(sliced.float() - ref.float()).abs().max().item():.3g} on the sliced final position, "
        f"argmax agreement {(sliced.argmax(-1) == ref.argmax(-1)).float().mean().item():.3f}"
    )
    torch.testing.assert_close(lens, ref)


def smoke(
    model, tok, df, qids, acts, out_opt, option_ids, n_examples=5, gold_col="gold_idx", prompt_col="prompt"
) -> dict:
    df = df.set_index("qid").loc[list(qids)].reset_index()
    gold = torch.tensor(df[gold_col].to_numpy(dtype=np.int64))
    out_p = torch.softmax(torch.from_numpy(out_opt), -1)
    acc_out = (out_p.argmax(-1) == gold).float().mean().item()
    mean_gold_p = out_p.gather(-1, gold[:, None]).mean().item()

    lens = layer_option_logits(model, acts, option_ids)
    lens_p = torch.softmax(lens, -1)
    L = lens.shape[1] - 1
    ranks = gold_rank(lens_p, gold[:, None].expand(-1, L + 1))  # [n, L+1]
    acc_lens_last = (lens_p[:, -1].argmax(-1) == gold).float().mean().item()
    agree = (lens_p[:, -1].argmax(-1) == out_p.argmax(-1)).float().mean().item()
    fp16_dev = (lens[:, -1] - torch.from_numpy(out_opt)).abs().max().item()

    log.info(f"n = {len(df)}   chance = 0.25")
    log.info(f"final-layer accuracy (model output, restricted softmax): {acc_out:.3f}")
    log.info(f"final-layer accuracy (logit lens on stored fp16 acts):   {acc_lens_last:.3f}")
    log.info(
        f"argmax agreement lens(fp16 acts) vs model output: {agree:.3f}; "
        f"max |option logit diff| = {fp16_dev:.3g} (fp16 storage)"
    )
    log.info("readout layers 0..L: 0 = embeddings, k = output of block k")
    log.info(f"mean gold probability (model output): {mean_gold_p:.3f}")
    log.info(f"mean gold rank per layer: {' '.join(f'{x:.2f}' for x in ranks.float().mean(0).tolist())}")
    log.info(f"\ngold rank per layer (0 = top-1), first {n_examples} questions:")
    for i in range(min(n_examples, len(df))):
        lstar = crystallization_layer((ranks[i] == 0).tolist())
        d = "-" if lstar is None else f"{lstar / L:.2f}"
        ranks_str = "".join(map(str, ranks[i].tolist()))
        log.info(f"  {df.qid[i]} gold={'ABCD'[int(gold[i])]} l*={lstar} d={d}  ranks: {ranks_str}")
    live_lens_check(model, tok, df[prompt_col].head(n_examples).tolist())
    log.info("assert lens(last layer) == model output: OK")
    return {
        "acc_out": acc_out,
        "acc_lens_last": acc_lens_last,
        "mean_gold_p": mean_gold_p,
        "argmax_agreement": agree,
        "ranks": ranks,
    }


def main(argv=None):
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--split", choices=["analysis", "all"], default="analysis")
    ap.add_argument("--include-leaks", action="store_true", help="keep leaks_answer rows in the analysis split")
    ap.add_argument("--variant", choices=list(VARIANTS), default="cyc0")
    ap.add_argument("--n-examples", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--cpu-layers", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = load_run_config()
    set_seed(cfg["seed"])
    mcfg = model_config(args.model)
    out_dir = resolve(cfg["paths"]["acts"]) / args.model / args.variant
    meta = json.loads((out_dir / "meta.json").read_text())
    qids, acts, out_opt = read_shards(out_dir)
    df = select_rows(args.split, args.limit, include_leaks=args.include_leaks, sample=args.sample, seed=cfg["seed"])
    if qids.tolist()[: len(df)] != df.qid.tolist():
        raise SystemExit("stored shards do not hold the same rows as --split/--limit")
    n = len(df)
    model, tok = load_model_and_tokenizer(mcfg, args.device, cpu_layers=args.cpu_layers)
    prompt_col, gold_col = variant_columns(args.variant)
    smoke(
        model,
        tok,
        df,
        qids[:n],
        acts[:n],
        out_opt[:n],
        meta["option_ids"],
        n_examples=args.n_examples or cfg["sample_sizes"]["smoke_rank_examples"],
        gold_col=gold_col,
        prompt_col=prompt_col,
    )
    if mcfg.get("proxy"):
        log.info("\n(proxy model: smoke test only, never used in analysis)")


if __name__ == "__main__":
    main()

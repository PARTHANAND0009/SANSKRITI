"""Smoke checks on stage 2 output (run after run/forward.py on the same rows).

Reports, for the first --limit analysis rows:
  - final-layer restricted-softmax accuracy (model output, and logit lens on the
    stored fp16 last-layer activations)
  - mean gold probability under the restricted softmax (model output)
  - gold rank per layer (logit lens) for --n-examples questions, and l*
And asserts that lens_logits at the last layer reproduces the model's output
logits, recomputed live in the model dtype on --n-examples prompts.

  python scripts/smoke.py --model qwen25_05b_proxy --device cpu --limit 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crystal.io import load_run_config, model_config, resolve, select_rows, set_seed  # noqa: E402
from crystal.lens import crystallization_layer, decoder_layers, gold_rank, lens_logits, lens_option_logits  # noqa: E402


def load_shards(out_dir: Path):
    shards = sorted(out_dir.glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {out_dir}; run run/forward.py first")
    zs = [np.load(s, allow_pickle=False) for s in shards]
    return (np.concatenate([z["qids"] for z in zs]), np.concatenate([z["acts"] for z in zs]),
            np.concatenate([z["out_opt_logits"] for z in zs]))


@torch.no_grad()
def layer_option_logits(model, acts: np.ndarray, option_ids) -> torch.Tensor:
    """[n, L, 4] logit-lens option logits from stored activations."""
    a = torch.from_numpy(acts)
    return torch.stack([lens_option_logits(model, a[:, l], l, option_ids).float().cpu()
                        for l in range(a.shape[1])], dim=1)


@torch.no_grad()
def live_lens_check(model, tok, prompts, report=print):
    """Lens at the last layer vs the model's own full-vocabulary logits, live.

    The assertion feeds the lens the same tensor the model's head sees (the last
    block's output over the whole sequence, with the model also computing logits
    for every position, so both sides run identical shapes), so it tests
    that lens_logits is the model's readout function; same dtype on both sides,
    default assert_close tolerances. Applying the lens to the single sliced
    position instead (what stage 2 stores) changes the RMSNorm reduction order;
    that difference is reported, not asserted (about 1e-5 in fp32 on the proxy).
    """
    from run.forward import left_pad_batch, position_ids_from_mask

    ids, mask = left_pad_batch(tok, prompts)
    dev = next(model.parameters()).device
    store = {}
    h = decoder_layers(model)[-1].register_forward_hook(
        lambda m, i, o: store.__setitem__("r", o[0] if isinstance(o, tuple) else o))
    try:
        out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev),
                    position_ids=position_ids_from_mask(mask).to(dev), use_cache=False, logits_to_keep=0)
    finally:
        h.remove()
    L = len(decoder_layers(model))
    ref = out.logits[:, -1]
    lens = lens_logits(model, store["r"], L)[:, -1]
    sliced = lens_logits(model, store["r"][:, -1], L)
    report(f"live check: max |lens - output| over full vocab = {(lens.float() - ref.float()).abs().max().item():.3g} "
           f"(same input); {(sliced.float() - ref.float()).abs().max().item():.3g} on the sliced final position, "
           f"argmax agreement {(sliced.argmax(-1) == ref.argmax(-1)).float().mean().item():.3f}")
    torch.testing.assert_close(lens, ref)


def smoke(model, tok, df, qids, acts, out_opt, option_ids, n_examples=5, gold_col="gold_idx",
          prompt_col="prompt", report=print) -> dict:
    df = df.set_index("qid").loc[list(qids)].reset_index()
    gold = torch.tensor(df[gold_col].to_numpy(dtype=np.int64))
    out_p = torch.softmax(torch.from_numpy(out_opt), -1)
    acc_out = (out_p.argmax(-1) == gold).float().mean().item()
    mean_gold_p = out_p.gather(-1, gold[:, None]).mean().item()

    lens = layer_option_logits(model, acts, option_ids)            # [n, L+1, 4]
    lens_p = torch.softmax(lens, -1)
    L = lens.shape[1] - 1                                         # blocks; readout layers are 0..L
    ranks = gold_rank(lens_p, gold[:, None].expand(-1, L + 1))     # [n, L+1]
    acc_lens_last = (lens_p[:, -1].argmax(-1) == gold).float().mean().item()
    agree = (lens_p[:, -1].argmax(-1) == out_p.argmax(-1)).float().mean().item()
    fp16_dev = (lens[:, -1] - torch.from_numpy(out_opt)).abs().max().item()

    report(f"n = {len(df)}   chance = 0.25")
    report(f"final-layer accuracy (model output, restricted softmax): {acc_out:.3f}")
    report(f"final-layer accuracy (logit lens on stored fp16 acts):   {acc_lens_last:.3f}")
    report(f"argmax agreement lens(fp16 acts) vs model output: {agree:.3f}; "
           f"max |option logit diff| = {fp16_dev:.3g} (fp16 storage)")
    report("readout layers 0..L: 0 = embeddings, k = output of block k")
    report(f"mean gold probability (model output): {mean_gold_p:.3f}")
    report(f"mean gold rank per layer: {' '.join(f'{x:.2f}' for x in ranks.float().mean(0).tolist())}")
    report(f"\ngold rank per layer (0 = top-1), first {n_examples} questions:")
    for i in range(min(n_examples, len(df))):
        lstar = crystallization_layer((ranks[i] == 0).tolist())
        report(f"  {df.qid[i]} gold={'ABCD'[int(gold[i])]} l*={lstar} "
               f"d={'-' if lstar is None else f'{lstar / L:.2f}'}  ranks: {''.join(map(str, ranks[i].tolist()))}")
    live_lens_check(model, tok, df[prompt_col].head(n_examples).tolist(), report=report)
    report("assert lens(last layer) == model output: OK")
    return {"acc_out": acc_out, "acc_lens_last": acc_lens_last, "mean_gold_p": mean_gold_p,
            "argmax_agreement": agree, "ranks": ranks}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--split", choices=["analysis", "all"], default="analysis")
    ap.add_argument("--include-leaks", action="store_true",
                    help="keep leaks_answer rows in the analysis split (excluded by default)")
    ap.add_argument("--variant", choices=["prompt", "prompt_permuted"], default="prompt")
    ap.add_argument("--n-examples", type=int, default=None)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--cpu-layers", type=int, default=None)
    args = ap.parse_args(argv)

    from run.forward import load_model_and_tokenizer

    cfg = load_run_config()
    set_seed(cfg["seed"])
    mcfg = model_config(args.model)
    out_dir = resolve(cfg["paths"]["acts"]) / args.model / args.variant
    meta = json.loads((out_dir / "meta.json").read_text())
    qids, acts, out_opt = load_shards(out_dir)
    df = select_rows(args.split, args.limit, include_leaks=args.include_leaks, sample=args.sample,
                     seed=cfg["seed"])
    if qids.tolist()[:len(df)] != df.qid.tolist():
        raise SystemExit("stored shards do not hold the same rows as --split/--limit")
    n = len(df)
    model, tok = load_model_and_tokenizer(mcfg, args.device, cpu_layers=args.cpu_layers)
    gold_col = "gold_idx" if args.variant == "prompt" else "gold_idx_permuted"
    smoke(model, tok, df, qids[:n], acts[:n], out_opt[:n], meta["option_ids"],
          n_examples=args.n_examples or cfg["sample_sizes"]["smoke_rank_examples"],
          gold_col=gold_col, prompt_col=args.variant)
    if mcfg.get("proxy"):
        print("\n(proxy model: smoke test only, never used in analysis)")


if __name__ == "__main__":
    main()

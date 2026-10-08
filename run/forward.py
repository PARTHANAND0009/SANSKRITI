"""Stage 2: residual stream at the final prompt token, for the embeddings and every block.

Batches are left-padded with explicit position_ids (cumsum of the mask), so position -1 is
the final prompt token in every row and padded rows see the same positions as unpadded ones.

Output: acts/{model}/{variant}/shard_XXXXX.npz with
  qids            [n]           str
  acts            [n, L+1, d]   float16   layer 0 = embeddings, layer k = block k output
  out_opt_logits  [n, 4]        float32   the model's own output logits at A..D
and meta.json. Shards that already hold the expected qids are skipped, so a rerun resumes.

  python -m run.forward --model llama31_8b --device cuda --variant cyc0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from crystal.hooks import record_residuals, stacked
from crystal.io import VARIANTS, load_run_config, model_config, resolve, select_rows, set_seed, variant_columns
from crystal.lens import decoder_layers
from crystal.log import setup_logging
from crystal.models import load_model_and_tokenizer, resolve_option_ids

log = logging.getLogger(__name__)


def left_pad_batch(tok, prompts: list[str]):
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left")
    return enc["input_ids"], enc["attention_mask"]


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    pos = mask.long().cumsum(-1) - 1
    return pos.clamp(min=0)


@torch.no_grad()
def collect_resid(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, option_ids):
    """Final-token residuals and option logits for a left-padded batch.

    Returns:
        acts [B, L+1, d] float32 (layer 0 = embeddings), out_opt_logits [B, 4] float32, on CPU.
    """
    if not bool(attention_mask[:, -1].all()):
        raise ValueError("final position is padding for some row: batch must be left-padded")
    dev = next(model.parameters()).device
    with record_residuals(model, lambda h: h[:, -1, :].detach().float().cpu()) as store:
        out = model(
            input_ids=input_ids.to(dev),
            attention_mask=attention_mask.to(dev),
            position_ids=position_ids_from_mask(attention_mask).to(dev),
            use_cache=False,
            logits_to_keep=1,
        )
    acts = torch.stack(stacked(store, len(decoder_layers(model))), dim=1)
    idx = torch.as_tensor(option_ids, device=out.logits.device)
    opt = out.logits[:, -1, :].float()[:, idx].cpu()
    return acts, opt


def run_shard(model, tok, prompts, option_ids, batch_size: int):
    """Activations for one shard in input order, batched by length to limit padding."""
    lengths = [len(tok(p)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    acts = [None] * len(prompts)
    opts = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        ids, mask = left_pad_batch(tok, [prompts[i] for i in idx])
        a, o = collect_resid(model, ids, mask, option_ids)
        for j, i in enumerate(idx):
            acts[i], opts[i] = a[j], o[j]
    return torch.stack(acts), torch.stack(opts)


def shard_path(out_dir: Path, i: int) -> Path:
    return out_dir / f"shard_{i:05d}.npz"


def shard_done(path: Path, qids) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            saved = z["qids"].tolist()
            complete = {"acts", "out_opt_logits"} <= set(z.files)
    except Exception:  # noqa: BLE001  (a truncated shard from a killed run is recomputed)
        return False
    if saved != list(qids):
        raise RuntimeError(
            f"{path} holds different qids than this run would write; "
            "the row selection changed. Remove the directory or use another --out."
        )
    return complete


def save_shard(path: Path, qids, acts: torch.Tensor, opts: torch.Tensor) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp,
        qids=np.asarray(qids, dtype=str),
        acts=acts.numpy().astype(np.float16),
        out_opt_logits=opts.numpy().astype(np.float32),
    )
    os.replace(tmp, path)


def run_forward(
    model, tok, df, out_dir: Path, option_ids, prompt_field="prompt", shard_size=256, batch_size=8, meta=None
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if meta is not None:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    paths = []
    n_shards = (len(df) + shard_size - 1) // shard_size
    for si in range(n_shards):
        part = df.iloc[si * shard_size : (si + 1) * shard_size]
        path = shard_path(out_dir, si)
        paths.append(path)
        qids = part["qid"].tolist()
        if shard_done(path, qids):
            log.info(f"shard {si + 1}/{n_shards}: exists, skipped")
            continue
        t0 = time.time()
        acts, opts = run_shard(model, tok, part[prompt_field].tolist(), option_ids, batch_size)
        save_shard(path, qids, acts, opts)
        log.info(f"shard {si + 1}/{n_shards}: {len(qids)} rows in {time.time() - t0:.1f}s -> {path}")
    return paths


def main(argv=None):
    setup_logging()
    cfg = load_run_config()
    ap = argparse.ArgumentParser(description="stage 2: residual stream at the final prompt token")
    ap.add_argument("--model", required=True, help="key in config/models.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=cfg["forward"]["shard_size"])
    ap.add_argument("--batch-size", type=int, default=None, help="default: models.yaml")
    ap.add_argument("--split", choices=["analysis", "all"], default="analysis")
    ap.add_argument("--include-leaks", action="store_true", help="keep leaks_answer rows in the analysis split")
    ap.add_argument(
        "--variant", choices=list(VARIANTS), default="cyc0", help="cyc0..cyc3: gold at A..D; perm: seeded shuffle"
    )
    ap.add_argument("--out", default=None, help="default: {acts}/{model}/{variant}")
    ap.add_argument("--sample", type=int, default=None, help="seeded random sample of N rows from the split")
    ap.add_argument(
        "--cpu-layers", type=int, default=None, help="CPU only: blocks kept in RAM, the rest streamed from disk"
    )
    args = ap.parse_args(argv)

    set_seed(cfg["seed"])
    mcfg = model_config(args.model)
    df = select_rows(args.split, args.limit, include_leaks=args.include_leaks, sample=args.sample, seed=cfg["seed"])
    model, tok = load_model_and_tokenizer(mcfg, args.device, cpu_layers=args.cpu_layers)
    opt_ids = resolve_option_ids(mcfg, tok)
    out_dir = Path(args.out) if args.out else resolve(cfg["paths"]["acts"]) / args.model / args.variant
    import transformers

    meta = {
        "model_key": args.model,
        "model_id": mcfg["id"],
        "proxy": bool(mcfg.get("proxy")),
        "n_layers": mcfg["n_layers"],
        "n_readout_layers": mcfg["n_layers"] + 1,
        "layer_convention": "0 = embeddings, k = output of block k",
        "d_model": model.config.hidden_size,
        "dtype": mcfg["dtype"],
        "split": args.split,
        "include_leaks": args.include_leaks,
        "limit": args.limit,
        "sample": args.sample,
        "cpu_layers": args.cpu_layers,
        "variant": args.variant,
        "n_rows": len(df),
        "shard_size": args.shard_size,
        "option_token_variant": mcfg["option_token_variant"],
        "option_ids": opt_ids,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "device": args.device,
    }
    run_forward(
        model,
        tok,
        df,
        out_dir,
        opt_ids,
        prompt_field=variant_columns(args.variant)[0],
        shard_size=args.shard_size,
        batch_size=args.batch_size or mcfg["batch_size"],
        meta=meta,
    )


if __name__ == "__main__":
    main()

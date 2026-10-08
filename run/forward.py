"""Stage 2: residual stream at the final prompt token, embeddings + every decoder layer.

For each question, a forward pre-hook on the first decoder block keeps its
input (the embedding output, layer 0) and a forward hook on every block keeps
its output (resid_post, layers 1..L), at the last prompt position. Batches are left-padded, with
attention_mask and explicit position_ids (cumsum of the mask), so position -1
is the final prompt token for every row and padded rows see the same positions
as unpadded ones.

Output: acts/{model}/{variant}/shard_XXXXX.npz with
  qids            [n]           str
  acts            [n, L+1, d]   float16   layer 0 = embeddings, layer k = block k output
  out_opt_logits  [n, 4]        float32   the model's own output logits at A..D
plus meta.json in the same directory. Existing shards whose qids match are
skipped, so an interrupted run resumes where it stopped.

  python -m run.forward --model llama31_8b --device cuda --split analysis
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from crystal.io import VARIANTS, load_run_config, model_config, resolve, select_rows, set_seed, variant_columns
from crystal.lens import decoder_layers
from crystal.tokens import option_ids as tokenizer_option_ids

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def left_pad_batch(tok, prompts: list[str]):
    enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left")
    return enc["input_ids"], enc["attention_mask"]


def position_ids_from_mask(mask: torch.Tensor) -> torch.Tensor:
    pos = mask.long().cumsum(-1) - 1
    return pos.clamp(min=0)


@torch.no_grad()
def collect_resid(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, option_ids):
    """Returns (acts [B, L+1, d] float32 on CPU, out_opt_logits [B, 4] float32 on CPU).
    acts[:, 0] is the embedding output, acts[:, k] the output of block k.

    Requires left padding: the last column must be a real token for every row.
    """
    if not bool(attention_mask[:, -1].all()):
        raise ValueError("final position is padding for some row: batch must be left-padded")
    layers = decoder_layers(model)
    store: dict[int, torch.Tensor] = {}

    def hook(i):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            store[i] = h[:, -1, :].detach().float().cpu()
        return f

    def pre_hook(mod, args, kwargs):
        h = args[0] if args else kwargs["hidden_states"]
        store[0] = h[:, -1, :].detach().float().cpu()

    handles = [layers[0].register_forward_pre_hook(pre_hook, with_kwargs=True)]
    handles += [l.register_forward_hook(hook(i + 1)) for i, l in enumerate(layers)]
    try:
        dev = next(model.parameters()).device
        out = model(
            input_ids=input_ids.to(dev),
            attention_mask=attention_mask.to(dev),
            position_ids=position_ids_from_mask(attention_mask).to(dev),
            use_cache=False,
            logits_to_keep=1,
        )
    finally:
        for h in handles:
            h.remove()
    acts = torch.stack([store[i] for i in range(len(layers) + 1)], dim=1)
    idx = torch.as_tensor(option_ids, device=out.logits.device)
    opt = out.logits[:, -1, :].float()[:, idx].cpu()
    return acts, opt


def run_shard(model, tok, prompts, option_ids, batch_size: int):
    """Activations for one shard, batched by length to limit padding, returned in input order."""
    lengths = [len(tok(p)["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    acts = [None] * len(prompts)
    opts = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
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
    except Exception:
        return False  # truncated / unreadable: recompute
    if saved != list(qids):
        raise RuntimeError(f"{path} holds different qids than this run would write; "
                           "the row selection changed. Remove the directory or use another --out.")
    return complete


def save_shard(path: Path, qids, acts: torch.Tensor, opts: torch.Tensor) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, qids=np.asarray(qids, dtype=str), acts=acts.numpy().astype(np.float16),
             out_opt_logits=opts.numpy().astype(np.float32))
    os.replace(tmp, path)


def run_forward(model, tok, df, out_dir: Path, option_ids, prompt_field="prompt",
                shard_size=256, batch_size=8, meta=None, log=print) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if meta is not None:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    paths = []
    n_shards = (len(df) + shard_size - 1) // shard_size
    for si in range(n_shards):
        part = df.iloc[si * shard_size:(si + 1) * shard_size]
        path = shard_path(out_dir, si)
        paths.append(path)
        qids = part["qid"].tolist()
        if shard_done(path, qids):
            log(f"shard {si + 1}/{n_shards}: exists, skipped")
            continue
        t0 = time.time()
        acts, opts = run_shard(model, tok, part[prompt_field].tolist(), option_ids, batch_size)
        save_shard(path, qids, acts, opts)
        log(f"shard {si + 1}/{n_shards}: {len(qids)} rows in {time.time() - t0:.1f}s -> {path}")
    return paths


def offload_device_map(model_id: str, n_layers: int, cpu_layers: int) -> dict:
    """Embeddings, final norm, rotary embedding and LM head in RAM ("cpu"); the
    first `cpu_layers` decoder blocks in RAM, the rest streamed from disk by
    accelerate (weights are read from the safetensors files per forward).
    Keeping the head and norm resident lets crystal.lens read them directly."""
    dm = {"model.embed_tokens": "cpu", "model.norm": "cpu", "model.rotary_emb": "cpu", "lm_head": "cpu"}
    for i in range(n_layers):
        dm[f"model.layers.{i}"] = "cpu" if i < cpu_layers else "disk"
    return dm


def load_model_and_tokenizer(mcfg: dict, device: str, cpu_layers: int | None = None, offload_dir=None):
    """cpu_layers=None: whole model on `device`. cpu_layers=k (CPU only): keep k
    decoder blocks in RAM and offload the rest to disk, for models larger than RAM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # local_path (models.yaml): a local copy, e.g. made by scripts/convert_bf16.py
    src = str(resolve(mcfg["local_path"])) if mcfg.get("local_path") else mcfg["id"]
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": DTYPES[mcfg["dtype"]]}
    if mcfg.get("attn_implementation"):
        kw["attn_implementation"] = mcfg["attn_implementation"]
    if cpu_layers is None:
        model = AutoModelForCausalLM.from_pretrained(src, **kw).to(device).eval()
    else:
        if device != "cpu":
            raise ValueError("disk offload (--cpu-layers) is only supported with --device cpu")
        offload_dir = Path(offload_dir or resolve("acts") / ".offload" / mcfg["key"])
        offload_dir.mkdir(parents=True, exist_ok=True)
        model = AutoModelForCausalLM.from_pretrained(
            src, device_map=offload_device_map(mcfg["id"], mcfg["n_layers"], cpu_layers),
            offload_folder=str(offload_dir), offload_state_dict=True, **kw).eval()
    n = len(decoder_layers(model))
    if n != mcfg["n_layers"]:
        raise RuntimeError(f"{mcfg['id']}: {n} decoder layers, models.yaml says {mcfg['n_layers']}")
    return model, tok


def resolve_option_ids(mcfg: dict, tok) -> list[int]:
    v = mcfg.get("option_token_variant", "pending")
    if v in ("pending", "none"):
        raise SystemExit(f"{mcfg['key']}: option_token_variant is {v!r}; run `make tokens` first")
    return tokenizer_option_ids(tok, v)


def main(argv=None):
    ap = argparse.ArgumentParser(description="stage 2: resid_post at the final prompt token")
    ap.add_argument("--model", required=True, help="key in config/models.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=None, help="default: models.yaml")
    ap.add_argument("--split", choices=["analysis", "all"], default="analysis")
    ap.add_argument("--include-leaks", action="store_true",
                    help="keep leaks_answer rows in the analysis split (excluded by default)")
    ap.add_argument("--variant", choices=list(VARIANTS), default="cyc0",
                    help="cyc0..cyc3: gold at A..D (main design); perm: seeded shuffle")
    ap.add_argument("--out", default=None, help="default: {acts}/{model}/{variant}")
    ap.add_argument("--sample", type=int, default=None,
                    help="seeded random sample of N rows from the split (seed: run.yaml)")
    ap.add_argument("--cpu-layers", type=int, default=None,
                    help="CPU only: keep this many decoder blocks in RAM, stream the rest from disk")
    args = ap.parse_args(argv)

    cfg = load_run_config()
    set_seed(cfg["seed"])
    mcfg = model_config(args.model)
    df = select_rows(args.split, args.limit, include_leaks=args.include_leaks, sample=args.sample,
                     seed=cfg["seed"])
    model, tok = load_model_and_tokenizer(mcfg, args.device, cpu_layers=args.cpu_layers)
    opt_ids = resolve_option_ids(mcfg, tok)
    out_dir = Path(args.out) if args.out else resolve(cfg["paths"]["acts"]) / args.model / args.variant
    import transformers

    meta = {
        "model_key": args.model, "model_id": mcfg["id"], "proxy": bool(mcfg.get("proxy")),
        "n_layers": mcfg["n_layers"], "n_readout_layers": mcfg["n_layers"] + 1,
        "layer_convention": "0 = embeddings, k = output of block k", "d_model": model.config.hidden_size, "dtype": mcfg["dtype"],
        "split": args.split, "include_leaks": args.include_leaks, "limit": args.limit, "sample": args.sample,
        "cpu_layers": args.cpu_layers, "variant": args.variant, "n_rows": len(df),
        "shard_size": args.shard_size, "option_token_variant": mcfg["option_token_variant"],
        "option_ids": opt_ids, "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "device": args.device,
    }
    run_forward(model, tok, df, out_dir, opt_ids, prompt_field=variant_columns(args.variant)[0],
                shard_size=args.shard_size, batch_size=args.batch_size or mcfg["batch_size"], meta=meta)


if __name__ == "__main__":
    main()

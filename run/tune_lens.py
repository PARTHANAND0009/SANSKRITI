"""Stage 3a: tuned lens (Belrose et al. 2023).

One translator per readout layer, T_l(h) = h + A_l (h / s_l) + b_l, initialised to the
identity and trained on the frozen model to minimise the summed per-layer
KL(p_final || p_lens_l) on WikiText-103 (held out from SANSKRITI).

s_l is a fixed input scale per layer (mean residual norm on a calibration batch). Residual
norms grow by orders of magnitude with depth, and without the scale a single step size is too
small early and unstable late: at lr 1 the layer-L translator diverged from round-off
gradients. SGD with Nesterov momentum follows the paper; Adam would rescale the round-off
gradients at layer L, where the identity is already exact, into full-size steps.

Outputs: lenses/{model}.pt and lenses/{model}.eval.json (held-out KL per layer, logit vs tuned).

  python -m run.tune_lens --model llama31_8b --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

from crystal.hooks import record_residuals, stacked
from crystal.io import load_run_config, model_config, resolve, set_seed
from crystal.lens import decoder_layers, lens_logits, n_readout_layers
from crystal.log import setup_logging
from crystal.models import load_model_and_tokenizer

log = logging.getLogger(__name__)

CALIBRATION_SEED_OFFSET = 7  # calibration batch drawn independently of the training batches
EVAL_SEED_OFFSET = 1


class TunedLens(torch.nn.Module):
    """Callable (h, layer) -> translated h, as crystal.lens.lens_logits expects."""

    def __init__(self, n_readout: int, d_model: int):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(n_readout, d_model, d_model))
        self.b = torch.nn.Parameter(torch.zeros(n_readout, d_model))
        self.register_buffer("scale", torch.ones(n_readout))

    def forward(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        x = h.to(self.A.dtype)
        return (x + (x / self.scale[layer]) @ self.A[layer].T + self.b[layer]).to(h.dtype)

    @torch.no_grad()
    def set_scale(self, resid) -> None:
        """resid: per-layer residuals [..., d] from a calibration batch."""
        self.scale.copy_(torch.stack([r.float().norm(dim=-1).mean() for r in resid]).to(self.scale))

    def deviation_from_identity(self) -> np.ndarray:
        """Per layer [L+1, 2]: ||A_l||_F / s_l and ||b_l|| / s_l (both 0 at the identity)."""
        s = self.scale.detach().cpu().numpy()
        return np.stack(
            [self.A.detach().flatten(1).norm(dim=1).cpu().numpy() / s, self.b.detach().norm(dim=1).cpu().numpy() / s], 1
        )


def load_tuned_lens(path, device="cpu") -> TunedLens:
    ck = torch.load(path, map_location=device, weights_only=False)
    lens = TunedLens(ck["n_readout_layers"], ck["d_model"]).to(device)
    lens.load_state_dict(ck["state_dict"])
    return lens.eval()


@torch.no_grad()
def residuals_all_positions(model, input_ids, attention_mask):
    """Per-layer residuals [B, T, d] for layers 0..L, and the final logits [B, T, V]."""
    with record_residuals(model, lambda h: h.detach()) as store:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return stacked(store, len(decoder_layers(model))), out.logits


def text_batches(tok, split: str, seq_len: int, n_tokens: int, seed: int, min_doc_chars: int = 200):
    """WikiText-103 token chunks [n_chunks, seq_len], each starting with BOS if the tokenizer has one."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split, cache_dir=str(resolve("data/raw/hf")))
    ids = []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    body = seq_len - (1 if tok.bos_token_id is not None else 0)
    for i in order:
        t = ds[int(i)]["text"].strip()
        if len(t) < min_doc_chars:
            continue
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        if len(ids) >= n_tokens:
            break
    chunks = [ids[s : s + body] for s in range(0, len(ids) - body + 1, body)]
    if tok.bos_token_id is not None:
        chunks = [[tok.bos_token_id, *c] for c in chunks]
    return torch.tensor(chunks)


def kl_final_vs_lens(model, lens, resid, final_logp, layer, positions):
    """Mean KL(p_final || p_lens) at `layer` over the selected positions."""
    h = resid[layer][positions]
    logits = lens_logits(model, h, layer, translator=lens).float()
    lp = F.log_softmax(logits, -1)
    t = final_logp[positions]
    return (t.exp() * (t - lp)).sum(-1).mean()


@torch.no_grad()
def evaluate(model, lens, chunks, batch, device, max_tokens_per_seq=None):
    """Held-out KL to the final output per layer: (logit lens, tuned lens)."""
    L1 = n_readout_layers(model)
    tot_raw, tot_tuned, n = np.zeros(L1), np.zeros(L1), 0
    for s in range(0, len(chunks), batch):
        ids = chunks[s : s + batch].to(device)
        resid, logits = residuals_all_positions(model, ids, torch.ones_like(ids))
        final_logp = F.log_softmax(logits.float(), -1)
        pos = (slice(None), slice(1, max_tokens_per_seq))  # skip the BOS position
        k = final_logp[pos].shape[0] * final_logp[pos].shape[1]
        for layer in range(L1):
            tot_raw[layer] += kl_final_vs_lens(model, None, resid, final_logp, layer, pos).item() * k
            tot_tuned[layer] += kl_final_vs_lens(model, lens, resid, final_logp, layer, pos).item() * k
        n += k
    return tot_raw / n, tot_tuned / n


def train(
    model,
    chunks,
    d_model,
    device,
    steps,
    batch,
    tokens_per_seq,
    lr,
    momentum,
    weight_decay,
    warmup,
    seed,
    grad_clip=1.0,
):
    L1 = n_readout_layers(model)
    lens = TunedLens(L1, d_model).to(device)
    g0 = torch.Generator().manual_seed(seed + CALIBRATION_SEED_OFFSET)
    calib = chunks[torch.randint(0, len(chunks), (min(batch, len(chunks)),), generator=g0)].to(device)
    lens.set_scale(residuals_all_positions(model, calib, torch.ones_like(calib))[0])
    opt = torch.optim.SGD(lens.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warmup)))
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, len(chunks), (batch,), generator=g)
        ids = chunks[idx].to(device)
        resid, logits = residuals_all_positions(model, ids, torch.ones_like(ids))
        final_logp = F.log_softmax(logits.float(), -1)
        T = ids.shape[1]
        pos_t = torch.stack([torch.randperm(T - 1, generator=g)[:tokens_per_seq] + 1 for _ in range(batch)])
        pos = (torch.arange(batch)[:, None].expand_as(pos_t), pos_t)
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for layer in range(L1):  # backward per layer keeps memory flat
            loss = kl_final_vs_lens(model, lens, resid, final_logp, layer, pos)
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(lens.parameters(), grad_clip)
        opt.step()
        sched.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log.info(f"  step {step + 1}/{steps}  mean KL over layers {total / L1:.4f}  ({time.time() - t0:.0f}s)")
    return lens


def main(argv=None):
    setup_logging()
    cfg = load_run_config()
    tl = cfg["tuned_lens"]
    ap = argparse.ArgumentParser(description="stage 3a: train a tuned lens")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=tl["steps"])
    ap.add_argument("--batch", type=int, default=tl["batch"])
    ap.add_argument("--seq-len", type=int, default=tl["seq_len"])
    ap.add_argument(
        "--tokens-per-seq", type=int, default=tl["tokens_per_seq"], help="positions per sequence in the loss"
    )
    ap.add_argument("--train-tokens", type=int, default=tl["train_tokens"])
    ap.add_argument("--eval-tokens", type=int, default=tl["eval_tokens"])
    ap.add_argument("--lr", type=float, default=tl["lr"])
    ap.add_argument("--momentum", type=float, default=tl["momentum"])
    ap.add_argument("--weight-decay", type=float, default=tl["weight_decay"])
    ap.add_argument("--warmup", type=int, default=tl["warmup"])
    ap.add_argument("--cpu-layers", type=int, default=None)
    ap.add_argument("--out", default="lenses")
    args = ap.parse_args(argv)

    set_seed(cfg["seed"])
    mcfg = model_config(args.model)
    model, tok = load_model_and_tokenizer(mcfg, args.device, cpu_layers=args.cpu_layers)
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = model.config.hidden_size
    train_chunks = text_batches(tok, "train", args.seq_len, args.train_tokens, cfg["seed"], tl["min_doc_chars"])
    eval_chunks = text_batches(
        tok, "validation", args.seq_len, args.eval_tokens, cfg["seed"] + EVAL_SEED_OFFSET, tl["min_doc_chars"]
    )
    log.info(
        f"{args.model}: {len(train_chunks)} train / {len(eval_chunks)} eval chunks of {args.seq_len} tokens "
        f"(WikiText-103), {n_readout_layers(model)} readout layers"
    )
    lens = train(
        model,
        train_chunks,
        d_model,
        args.device,
        args.steps,
        args.batch,
        args.tokens_per_seq,
        args.lr,
        args.momentum,
        args.weight_decay,
        args.warmup,
        cfg["seed"],
        grad_clip=tl["grad_clip"],
    )
    raw, tuned = evaluate(model, lens, eval_chunks, args.batch, args.device)
    dev = lens.deviation_from_identity()
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": lens.state_dict(),
            "n_readout_layers": n_readout_layers(model),
            "d_model": d_model,
            "model_id": mcfg["id"],
            "proxy": bool(mcfg.get("proxy")),
            "args": vars(args),
            "train_data": "Salesforce/wikitext wikitext-103-raw-v1 train",
        },
        out / f"{args.model}.pt",
    )
    report = {
        "model": args.model,
        "model_id": mcfg["id"],
        "proxy": bool(mcfg.get("proxy")),
        "eval_data": "wikitext-103-raw-v1 validation",
        "eval_chunks": len(eval_chunks),
        "kl_logit_lens": raw.tolist(),
        "kl_tuned_lens": tuned.tolist(),
        "translator_dev_A": dev[:, 0].tolist(),
        "translator_dev_b": dev[:, 1].tolist(),
        "input_scale": lens.scale.tolist(),
        "args": vars(args),
    }
    (out / f"{args.model}.eval.json").write_text(json.dumps(report, indent=2))
    log.info("layer  KL(logit lens)  KL(tuned lens)  |A/s|   input scale s")
    for layer in range(len(raw)):
        scale = lens.scale[layer].item()
        log.info(f"{layer:5d}  {raw[layer]:14.4f}  {tuned[layer]:14.4f}  {dev[layer, 0]:.5f}  {scale:9.2f}")
    log.info(f"wrote {out / args.model}.pt")


if __name__ == "__main__":
    main()

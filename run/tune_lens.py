"""Stage 3a: tuned lens (Belrose et al. 2023).

One affine translator per readout layer l in 0..L:  T_l(h) = h + A_l (h / s_l) + b_l, with
A_l and b_l initialised to zero (identity). s_l is a fixed per-layer input scale (mean token
norm of h_l on a calibration batch, stored with the lens): residual norms grow by orders of
magnitude with depth (hundreds at Qwen2.5-0.5B's last layers), so without it one SGD step
size is too small early and unstable late (at lr 1 the layer-L translator diverged from
identity, growing ~3x per step from round-off gradients). It is trained so that the lens distribution
lens_logits(model, T_l(h_l), l) matches the model's own final output distribution:
loss = sum over layers of mean_tokens KL(p_final || p_lens_l). The model is frozen.

Training text is generic and held out from SANSKRITI: WikiText-103 (train split for
training, validation split for evaluation). As in the paper the optimiser is SGD with
Nesterov momentum. (Adam would rescale the near-zero round-off gradients at layer L,
where the identity already reproduces the output, into full-size steps.)

Outputs: lenses/{model}.pt (translator weights + metadata) and lenses/{model}.eval.json
(per-layer held-out KL to the final output, logit lens vs tuned lens).

  python -m run.tune_lens --model qwen25_05b_proxy --device cpu --steps 150
  python -m run.tune_lens --model llama31_8b --device cuda          # defaults for GPU
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from crystal.io import load_run_config, model_config, resolve, set_seed
from crystal.lens import decoder_layers, lens_logits, n_readout_layers


class TunedLens(torch.nn.Module):
    """Callable (h, layer) -> translated h, as crystal.lens.lens_logits expects."""

    def __init__(self, n_layers_readout: int, d_model: int):
        super().__init__()
        self.A = torch.nn.Parameter(torch.zeros(n_layers_readout, d_model, d_model))
        self.b = torch.nn.Parameter(torch.zeros(n_layers_readout, d_model))
        self.register_buffer("scale", torch.ones(n_layers_readout))

    def forward(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        x = h.to(self.A.dtype)
        return (x + (x / self.scale[layer]) @ self.A[layer].T + self.b[layer]).to(h.dtype)

    @torch.no_grad()
    def set_scale(self, resid) -> None:
        """resid: [L+1] list of [..., d] residuals from a calibration batch."""
        self.scale.copy_(torch.stack([r.float().norm(dim=-1).mean() for r in resid]).to(self.scale))

    def deviation_from_identity(self) -> np.ndarray:
        """Per layer: ||A_l / s_l||_F (operator size relative to the input scale; 0 = identity)
        and ||b_l|| / s_l (bias relative to the typical residual norm)."""
        s = self.scale.detach().cpu().numpy()
        return np.stack([self.A.detach().flatten(1).norm(dim=1).cpu().numpy() / s,
                         self.b.detach().norm(dim=1).cpu().numpy() / s], 1)


def load_tuned_lens(path, device="cpu") -> TunedLens:
    ck = torch.load(path, map_location=device, weights_only=False)
    lens = TunedLens(ck["n_readout_layers"], ck["d_model"]).to(device)
    lens.load_state_dict(ck["state_dict"])
    return lens.eval()


@torch.no_grad()
def residuals_all_positions(model, input_ids, attention_mask):
    """[L+1] list of [B, T, d] (0 = embedding output, k = block k output) and the
    model's final logits [B, T, V]."""
    layers = decoder_layers(model)
    store = {}

    def pre(mod, args, kwargs):
        store[0] = (args[0] if args else kwargs["hidden_states"]).detach()

    def post(i):
        def f(mod, inp, out):
            store[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return f

    hs = [layers[0].register_forward_pre_hook(pre, with_kwargs=True)]
    hs += [l.register_forward_hook(post(i + 1)) for i, l in enumerate(layers)]
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    return [store[i] for i in range(len(layers) + 1)], out.logits


def text_batches(tok, split: str, seq_len: int, n_tokens: int, seed: int):
    """Token chunks of WikiText-103 (raw), [n_chunks, seq_len]. BOS prepended when the
    tokenizer uses one, as the model sees in use."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split,
                      cache_dir=str(resolve("data/raw/hf")))
    text, ids = [], []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ds))
    body = seq_len - (1 if tok.bos_token_id is not None else 0)
    for i in order:
        t = ds[int(i)]["text"].strip()
        if len(t) < 200:
            continue
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        if len(ids) >= n_tokens:
            break
    chunks = [ids[s:s + body] for s in range(0, len(ids) - body + 1, body)]
    if tok.bos_token_id is not None:
        chunks = [[tok.bos_token_id] + c for c in chunks]
    return torch.tensor(chunks)


def kl_final_vs_lens(model, lens, resid, final_logp, layer, positions):
    """Mean over the selected tokens of KL(p_final || p_lens) at `layer`."""
    h = resid[layer][positions]
    logits = lens_logits(model, h, layer, translator=lens).float()
    lp = F.log_softmax(logits, -1)
    t = final_logp[positions]
    return (t.exp() * (t - lp)).sum(-1).mean()


@torch.no_grad()
def evaluate(model, lens, chunks, batch, device, max_tokens_per_seq=None):
    """Per-layer held-out KL to the final output for the logit lens and the tuned lens."""
    L1 = n_readout_layers(model)
    tot_raw, tot_tuned, n = np.zeros(L1), np.zeros(L1), 0
    for s in range(0, len(chunks), batch):
        ids = chunks[s:s + batch].to(device)
        resid, logits = residuals_all_positions(model, ids, torch.ones_like(ids))
        final_logp = F.log_softmax(logits.float(), -1)
        pos = (slice(None), slice(1, max_tokens_per_seq))   # skip BOS position
        k = final_logp[pos].shape[0] * final_logp[pos].shape[1]
        for l in range(L1):
            tot_raw[l] += kl_final_vs_lens(model, None, resid, final_logp, l, pos).item() * k
            tot_tuned[l] += kl_final_vs_lens(model, lens, resid, final_logp, l, pos).item() * k
        n += k
    return tot_raw / n, tot_tuned / n


def train(model, chunks, d_model, device, steps, batch, tokens_per_seq, lr, momentum, weight_decay,
          warmup, seed, log=print):
    L1 = n_readout_layers(model)
    lens = TunedLens(L1, d_model).to(device)
    g0 = torch.Generator().manual_seed(seed + 7)
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
        for l in range(L1):                                  # per-layer backward keeps memory flat
            loss = kl_final_vs_lens(model, lens, resid, final_logp, l, pos)
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log(f"  step {step + 1}/{steps}  mean KL over layers {total / L1:.4f}  ({time.time() - t0:.0f}s)")
    return lens


def main(argv=None):
    ap = argparse.ArgumentParser(description="stage 3a: train a tuned lens")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--tokens-per-seq", type=int, default=128, help="positions per sequence used in the loss")
    ap.add_argument("--train-tokens", type=int, default=4_000_000)
    ap.add_argument("--eval-tokens", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--cpu-layers", type=int, default=None)
    ap.add_argument("--out", default="lenses")
    args = ap.parse_args(argv)

    from run.forward import load_model_and_tokenizer

    cfg = load_run_config()
    set_seed(cfg["seed"])
    mcfg = model_config(args.model)
    model, tok = load_model_and_tokenizer(mcfg, args.device, cpu_layers=args.cpu_layers)
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = model.config.hidden_size
    train_chunks = text_batches(tok, "train", args.seq_len, args.train_tokens, cfg["seed"])
    eval_chunks = text_batches(tok, "validation", args.seq_len, args.eval_tokens, cfg["seed"] + 1)
    print(f"{args.model}: {len(train_chunks)} train / {len(eval_chunks)} eval chunks of {args.seq_len} tokens "
          f"(WikiText-103), {n_readout_layers(model)} readout layers")
    lens = train(model, train_chunks, d_model, args.device, args.steps, args.batch, args.tokens_per_seq,
                 args.lr, args.momentum, args.weight_decay, args.warmup, cfg["seed"])
    raw, tuned = evaluate(model, lens, eval_chunks, args.batch, args.device)
    dev = lens.deviation_from_identity()
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": lens.state_dict(), "n_readout_layers": n_readout_layers(model), "d_model": d_model,
                "model_id": mcfg["id"], "proxy": bool(mcfg.get("proxy")), "args": vars(args),
                "train_data": "Salesforce/wikitext wikitext-103-raw-v1 train"}, out / f"{args.model}.pt")
    report = {"model": args.model, "model_id": mcfg["id"], "proxy": bool(mcfg.get("proxy")),
              "eval_data": "wikitext-103-raw-v1 validation", "eval_chunks": len(eval_chunks),
              "kl_logit_lens": raw.tolist(), "kl_tuned_lens": tuned.tolist(),
              "translator_dev_A": dev[:, 0].tolist(), "translator_dev_b": dev[:, 1].tolist(),
              "input_scale": lens.scale.tolist(), "args": vars(args)}
    (out / f"{args.model}.eval.json").write_text(json.dumps(report, indent=2))
    print("layer  KL(logit lens)  KL(tuned lens)  |A/s|   input scale s")
    for l in range(len(raw)):
        print(f"{l:5d}  {raw[l]:14.4f}  {tuned[l]:14.4f}  {dev[l, 0]:.5f}  {lens.scale[l].item():9.2f}")
    print(f"wrote {out / args.model}.pt")


if __name__ == "__main__":
    main()

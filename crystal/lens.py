"""Read a residual-stream vector at any layer out as vocabulary logits.

Logit lens: final norm, then the unembedding, then Gemma 2's final logit softcap.
Tuned lens (Belrose et al. 2023): a per-layer affine translator (run/tune_lens.py) is
applied to the residual first.

Readout layers are 0..L: layer 0 is the embedding output, i.e. the input to the first
block (including Gemma's sqrt(d_model) scaling), and layer k is the output of block k.
Layer L reproduces the model's own output logits.
"""

from __future__ import annotations

import torch


def final_norm(model) -> torch.nn.Module:
    return model.get_decoder().norm


def unembed(model) -> torch.nn.Module:
    # lm_head shares its weight with embed_tokens when embeddings are tied (Gemma 2,
    # Qwen2.5-0.5B); using the model's own output module covers tied and untied heads.
    return model.get_output_embeddings()


def decoder_layers(model):
    return model.get_decoder().layers


def n_readout_layers(model) -> int:
    return len(decoder_layers(model)) + 1


def check_layer(model, layer: int) -> None:
    n = len(decoder_layers(model))
    if not 0 <= layer <= n:
        raise ValueError(f"layer {layer} out of range 0..{n} (0 = embeddings, {n} = last block)")


def softcap(model) -> float | None:
    return getattr(model.config, "final_logit_softcapping", None)


def _normed(model, resid: torch.Tensor, layer: int, translator) -> torch.Tensor:
    check_layer(model, layer)
    p = next(model.parameters())
    h = resid.to(device=p.device, dtype=p.dtype)
    if translator is not None:
        h = translator(h, layer)
    return final_norm(model)(h)


def _softcapped(model, logits: torch.Tensor) -> torch.Tensor:
    cap = softcap(model)
    return logits if cap is None else torch.tanh(logits / cap) * cap


def lens_logits(model, resid: torch.Tensor, layer: int, translator=None) -> torch.Tensor:
    """Vocabulary logits [..., V] from resid [..., d_model] at readout layer `layer`.

    translator: a tuned lens, callable (resid, layer) -> resid; None gives the logit lens.
    """
    return _softcapped(model, unembed(model)(_normed(model, resid, layer, translator)))


def lens_option_logits(model, resid: torch.Tensor, layer: int, option_ids, translator=None) -> torch.Tensor:
    """lens_logits(...)[..., option_ids] without the full vocabulary (norm and softcap act elementwise)."""
    h = _normed(model, resid, layer, translator)
    head = unembed(model)
    idx = torch.as_tensor(option_ids, device=h.device)
    bias = None if head.bias is None else head.bias[idx]
    return _softcapped(model, torch.nn.functional.linear(h, head.weight[idx], bias))


def restricted_probs(logits: torch.Tensor, option_ids) -> torch.Tensor:
    """Softmax over the four option tokens: [..., V] -> [..., 4]."""
    idx = torch.as_tensor(option_ids, device=logits.device)
    return torch.softmax(logits.float()[..., idx], dim=-1)


def gold_rank(probs: torch.Tensor, gold_idx: torch.Tensor) -> torch.Tensor:
    """Rank of the gold option among the four (0 = top-1). probs [..., 4], gold_idx [...]."""
    g = probs.gather(-1, gold_idx.unsqueeze(-1))
    return (probs > g).sum(-1)


def crystallization_layer(top1_is_gold) -> int | None:
    """l*: first readout layer from which gold is top-1 through layer L; None if not top-1 at L."""
    flags = list(map(bool, top1_is_gold))
    if not flags or not flags[-1]:
        return None
    layer = len(flags) - 1
    while layer > 0 and flags[layer - 1]:
        layer -= 1
    return layer

"""Shared readout: map a residual-stream vector at some layer to vocabulary logits.

Logit lens: final norm -> unembedding (-> Gemma 2 final logit softcapping).
Tuned lens (Belrose et al. 2023): an affine translator per layer is applied to
the residual first, then the same head. Translators are trained in stage 3a
(run/lens.py); this module only applies them.
"""
from __future__ import annotations

import torch


def final_norm(model) -> torch.nn.Module:
    """The norm applied after the last decoder layer (model.model.norm for
    Llama / Qwen2 / Gemma2)."""
    return model.get_decoder().norm


def unembed(model) -> torch.nn.Module:
    """The output projection. get_output_embeddings() returns lm_head, which
    shares its weight tensor with embed_tokens when tie_word_embeddings is set
    (Gemma 2, Qwen2.5-0.5B); Llama-3.1-8B and Qwen2.5-7B have untied heads.
    Using the module the model itself uses keeps both cases correct."""
    return model.get_output_embeddings()


def decoder_layers(model):
    return model.get_decoder().layers


def softcap(model) -> float | None:
    return getattr(model.config, "final_logit_softcapping", None)


def lens_logits(model, resid: torch.Tensor, layer: int, translator=None) -> torch.Tensor:
    """Vocabulary logits read out from `resid`, the output of decoder layer `layer`
    (0-indexed, resid_post).

    resid: [..., d_model]. Computed in the model's parameter dtype. If
    `translator` is given (a tuned lens: callable (resid, layer) -> resid), it is
    applied before the norm. With translator=None this is the logit lens, and at
    layer = n_layers - 1 it reproduces the model's own output logits.
    """
    n = len(decoder_layers(model))
    if not 0 <= layer < n:
        raise ValueError(f"layer {layer} out of range for {n} decoder layers")
    p = next(model.parameters())
    h = resid.to(device=p.device, dtype=p.dtype)
    if translator is not None:
        h = translator(h, layer)
    logits = unembed(model)(final_norm(model)(h))
    cap = softcap(model)
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


def restricted_probs(logits: torch.Tensor, option_ids) -> torch.Tensor:
    """Softmax restricted to the four option tokens. logits [..., V] -> [..., 4]."""
    idx = torch.as_tensor(option_ids, device=logits.device)
    return torch.softmax(logits.float()[..., idx], dim=-1)


def gold_rank(probs: torch.Tensor, gold_idx: torch.Tensor) -> torch.Tensor:
    """Rank (0 = top-1) of the gold option among the 4. probs [..., 4], gold_idx [...]."""
    g = probs.gather(-1, gold_idx.unsqueeze(-1))
    return (probs > g).sum(-1)


def crystallization_layer(top1_is_gold) -> int | None:
    """l*: first layer where gold is top-1 and stays top-1 through the last layer.

    top1_is_gold: sequence of bools for decoder layers 0..L-1 (resid_post).
    Returns l* 1-indexed (l* = k means "output of the k-th decoder block"), so
    d = l*/L lies in (0, 1] and d = 1 means only the last layer gets it right.
    Returns None if gold is not top-1 at the last layer.
    """
    flags = list(map(bool, top1_is_gold))
    if not flags or not flags[-1]:
        return None
    l = len(flags) - 1
    while l > 0 and flags[l - 1]:
        l -= 1
    return l + 1

"""Forward hooks on the residual stream, in the readout-layer convention of crystal.lens."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager

import torch

from crystal.lens import decoder_layers


@contextmanager
def record_residuals(model, keep: Callable[[torch.Tensor], torch.Tensor]):
    """Yield a dict filled during the forward pass: layer -> keep(hidden state).

    Layer 0 is the input of the first block (embedding output, after Gemma's sqrt(d)
    scaling), taken with a pre-hook; layer k is the output of block k.
    """
    layers = decoder_layers(model)
    store: dict[int, torch.Tensor] = {}

    def pre(mod, args, kwargs):
        store[0] = keep(args[0] if args else kwargs["hidden_states"])

    def post(i):
        def f(mod, inp, out):
            store[i] = keep(out[0] if isinstance(out, tuple) else out)

        return f

    handles = [layers[0].register_forward_pre_hook(pre, with_kwargs=True)]
    handles += [layer.register_forward_hook(post(i + 1)) for i, layer in enumerate(layers)]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def stacked(store: dict, n_layers: int) -> list[torch.Tensor]:
    """The recorded states in layer order 0..n_layers."""
    return [store[i] for i in range(n_layers + 1)]

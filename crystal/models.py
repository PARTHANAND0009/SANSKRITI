"""Model and tokenizer loading for every stage that runs a model."""

from __future__ import annotations

from pathlib import Path

import torch

from crystal.io import resolve
from crystal.lens import decoder_layers
from crystal.tokens import option_ids

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def offload_device_map(n_layers: int, cpu_layers: int) -> dict:
    """Keep embeddings, final norm, rotary embedding, LM head and the first `cpu_layers`
    blocks in RAM; accelerate streams the remaining blocks from disk on each forward.
    The head and norm must stay resident because crystal.lens reads them directly."""
    dm = {"model.embed_tokens": "cpu", "model.norm": "cpu", "model.rotary_emb": "cpu", "lm_head": "cpu"}
    for i in range(n_layers):
        dm[f"model.layers.{i}"] = "cpu" if i < cpu_layers else "disk"
    return dm


def load_model_and_tokenizer(mcfg: dict, device: str, cpu_layers: int | None = None, offload_dir=None):
    """Load a model from config/models.yaml in eval mode.

    Args:
        cpu_layers: CPU only. Keep this many decoder blocks in RAM and offload the rest to
            disk, for models larger than RAM. None loads the whole model on `device`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
            src,
            device_map=offload_device_map(mcfg["n_layers"], cpu_layers),
            offload_folder=str(offload_dir),
            offload_state_dict=True,
            **kw,
        ).eval()
    n = len(decoder_layers(model))
    if n != mcfg["n_layers"]:
        raise RuntimeError(f"{mcfg['id']}: {n} decoder layers, models.yaml says {mcfg['n_layers']}")
    return model, tok


def resolve_option_ids(mcfg: dict, tok) -> list[int]:
    v = mcfg.get("option_token_variant", "pending")
    if v in ("pending", "none"):
        raise SystemExit(f"{mcfg['key']}: option_token_variant is {v!r}; run `make tokens` first")
    return option_ids(tok, v)

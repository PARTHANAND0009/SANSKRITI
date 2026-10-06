"""Offline fixtures: tiny random models of each family and a char-level tokenizer.

Nothing here is downloaded; nothing here is real data.
"""
from __future__ import annotations

import string

import torch
from transformers import (
    AutoModelForCausalLM,
    Gemma2Config,
    LlamaConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
)

CHARS = string.printable  # 100 chars incl. space and newline
VOCAB = 128
N_LAYERS = 4

_COMMON = dict(
    vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=N_LAYERS,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
    pad_token_id=0, bos_token_id=1, eos_token_id=2,
)

FAMILIES = {
    "llama": lambda tie: LlamaConfig(**_COMMON, tie_word_embeddings=tie),
    "qwen2": lambda tie: Qwen2Config(**_COMMON, tie_word_embeddings=tie),
    # small softcap so the cap actually bites on random logits
    "gemma2": lambda tie: Gemma2Config(**_COMMON, tie_word_embeddings=tie, final_logit_softcapping=0.5,
                                       sliding_window=8),
}


def tiny_model(family: str, tie: bool = False, seed: int = 0):
    torch.manual_seed(seed)
    cfg = FAMILIES[family](tie)
    kw = {"attn_implementation": "eager"} if family == "gemma2" else {}
    m = AutoModelForCausalLM.from_config(cfg, **kw).eval()
    # random init leaves norm weights at their identity value; perturb them so a
    # lens that skipped the final norm would be caught
    with torch.no_grad():
        m.get_decoder().norm.weight.normal_(0, 0.5)
    return m


def char_tokenizer() -> PreTrainedTokenizerFast:
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}
    for c in CHARS:
        vocab.setdefault(c, len(vocab))
    assert len(vocab) <= VOCAB
    t = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    t.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", bos_token="<s>",
                                   eos_token="</s>", unk_token="<unk>")

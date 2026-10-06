"""lens_logits at the last layer must reproduce the model's own output logits."""
import pytest
import torch

from crystal.lens import crystallization_layer, decoder_layers, gold_rank, lens_logits, restricted_probs
from tests.fixtures import N_LAYERS, VOCAB, tiny_model


def _last_resid_and_logits(model, ids, mask=None):
    store = {}
    layer = decoder_layers(model)[-1]
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("r", o[0] if isinstance(o, tuple) else o))
    try:
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
    finally:
        h.remove()
    return store["r"], out.logits


@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("family", ["llama", "qwen2", "gemma2"])
def test_lens_last_layer_equals_output(family, tie):
    model = tiny_model(family, tie=tie)
    assert (model.get_output_embeddings().weight is model.get_input_embeddings().weight) == tie
    ids = torch.randint(4, VOCAB, (3, 9), generator=torch.Generator().manual_seed(1))
    resid, logits = _last_resid_and_logits(model, ids)
    with torch.no_grad():
        lens = lens_logits(model, resid, N_LAYERS)
    torch.testing.assert_close(lens, logits)


def test_gemma2_softcap_is_applied():
    model = tiny_model("gemma2")
    cap = model.config.final_logit_softcapping
    ids = torch.randint(4, VOCAB, (2, 7), generator=torch.Generator().manual_seed(2))
    resid, logits = _last_resid_and_logits(model, ids)
    assert logits.abs().max() <= cap
    with torch.no_grad():
        raw = model.get_output_embeddings()(model.get_decoder().norm(resid))
    assert raw.abs().max() > cap  # the cap bites in this fixture, so the test is meaningful
    assert not torch.allclose(raw, logits)


def test_layer_range_checked():
    model = tiny_model("llama")
    with pytest.raises(ValueError):
        lens_logits(model, torch.zeros(1, 64), N_LAYERS + 1)


def test_restricted_helpers():
    logits = torch.tensor([[0.0, 5.0, 1.0, 2.0, 3.0]])
    p = restricted_probs(logits, [1, 2, 3, 4])
    torch.testing.assert_close(p.sum(-1), torch.ones(1))
    assert gold_rank(p, torch.tensor([0])).item() == 0
    assert gold_rank(p, torch.tensor([1])).item() == 3


def test_crystallization_layer():
    # flags over readout layers 0..L (here L = 4); index 0 = embeddings
    assert crystallization_layer([0, 0, 0, 1, 1]) == 3
    assert crystallization_layer([0, 1, 0, 1, 1]) == 3
    assert crystallization_layer([0, 1, 1, 1, 1]) == 1
    assert crystallization_layer([1, 1, 1, 1, 1]) == 0
    assert crystallization_layer([0, 1, 1, 1, 0]) is None
    assert crystallization_layer([0, 0, 0, 0, 1]) == 4

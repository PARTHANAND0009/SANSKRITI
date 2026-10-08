"""Tuned lens on a tiny random model (CPU, seconds)."""

import torch

from run.tune_lens import TunedLens, evaluate, load_tuned_lens, train
from tests.fixtures import N_LAYERS, VOCAB, tiny_model


def test_tuned_lens_identity_init_and_save_load(tmp_path):
    lens = TunedLens(N_LAYERS + 1, 64)
    h = torch.randn(3, 64)
    for l in range(N_LAYERS + 1):
        torch.testing.assert_close(lens(h, l), h)
    torch.save({"state_dict": lens.state_dict(), "n_readout_layers": N_LAYERS + 1, "d_model": 64}, tmp_path / "x.pt")
    assert isinstance(load_tuned_lens(tmp_path / "x.pt"), TunedLens)


def test_tuned_lens_beats_logit_lens_and_last_layer_stays_identity():
    model = tiny_model("llama")
    for p in model.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(0)
    train_chunks = torch.randint(4, VOCAB, (64, 24), generator=g)
    eval_chunks = torch.randint(4, VOCAB, (16, 24), generator=g)
    lens = train(
        model,
        train_chunks,
        64,
        "cpu",
        steps=60,
        batch=8,
        tokens_per_seq=16,
        lr=1.0,
        momentum=0.9,
        weight_decay=1e-3,
        warmup=5,
        seed=0,
    )
    raw, tuned = evaluate(model, lens, eval_chunks, batch=8, device="cpu")
    assert (tuned[:N_LAYERS] < raw[:N_LAYERS]).all(), (raw, tuned)
    dev = lens.deviation_from_identity()
    assert dev[N_LAYERS, 0] < 1e-4 and dev[N_LAYERS, 1] < 1e-4, dev[N_LAYERS]
    assert abs(tuned[N_LAYERS] - raw[N_LAYERS]) < 1e-4


def test_scale_keeps_identity_and_is_saved(tmp_path):
    lens = TunedLens(3, 8)
    lens.set_scale([torch.full((5, 8), float(k + 1)) for k in range(3)])
    h = torch.randn(2, 8)
    torch.testing.assert_close(lens(h, 2), h)
    torch.save({"state_dict": lens.state_dict(), "n_readout_layers": 3, "d_model": 8}, tmp_path / "s.pt")
    torch.testing.assert_close(load_tuned_lens(tmp_path / "s.pt").scale, lens.scale)

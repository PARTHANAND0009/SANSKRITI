"""Activation patching on a tiny random model."""
import numpy as np
import torch

from crystal.tokens import option_ids
from data.prep import build_prompt
from run.patch import first_at_least, logit_diff, patch_question
from tests.fixtures import N_LAYERS, char_tokenizer, tiny_model


def test_patching_endpoints():
    tok = char_tokenizer()
    ids = option_ids(tok, "bare")
    opts = ["red", "blue", "green", "black"]
    clean = build_prompt("Which colour is the Kaziranga rhino?", opts)
    corrupt = build_prompt("Which colour is the Sundarbans tiger?", opts)
    for family in ("llama", "qwen2", "gemma2"):
        model = tiny_model(family)
        ldc, ldx, ldp, rec = patch_question(model, tok, clean, corrupt, 2, ids, "cpu")
        assert len(rec) == N_LAYERS + 1
        # last layer: the final-token residual fully determines the output -> exactly clean
        np.testing.assert_allclose(ldp[-1], ldc, rtol=1e-5, atol=1e-5)
        assert abs(rec[-1] - 1) < 1e-4
        # layer 0: both prompts end in the same token, so the patch changes nothing
        np.testing.assert_allclose(ldp[0], ldx, rtol=1e-5, atol=1e-5)
        assert abs(rec[0]) < 1e-4


def test_helpers():
    assert logit_diff(torch.tensor([1.0, 3.0, 2.0, 0.0]), 1).item() == 1.0
    assert first_at_least(np.array([0, 0.2, 0.6, 1.0]), 0.5) == 2
    assert first_at_least(np.array([0, 0.2]), 0.5) is None


def test_allocate_water_filling():
    from run.patch import allocate

    a = allocate({"x": 5, "y": 100, "z": 100}, 90)
    assert a == {"x": 5, "y": 43, "z": 42} or sum(a.values()) == 90 and a["x"] == 5
    assert sum(allocate({"x": 3, "y": 4}, 50).values()) == 7

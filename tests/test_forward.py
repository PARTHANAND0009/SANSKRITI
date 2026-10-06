"""Stage 2 on tiny random models: batching, padding, shards, resume."""
import numpy as np
import pandas as pd
import pytest
import torch

import run.forward as fwd
from crystal.lens import lens_option_logits
from crystal.tokens import option_ids
from data.prep import build_prompt
from tests.fixtures import N_LAYERS, char_tokenizer, tiny_model

# prompt lengths differ a lot so batches carry real left padding
PROMPTS = [
    build_prompt("Which dish?", ["x", "y", "z", "w"]),
    build_prompt("A much longer question stem that pads the others in its batch?",
                 ["first option", "second", "third option here", "4"]),
    build_prompt("Q?", ["a", "b", "c", "d"]),
    build_prompt("Which festival is celebrated in this region every year?", ["p", "q", "r", "s"]),
    build_prompt("Mid-length stem here?", ["one", "two", "three", "four"]),
]


@pytest.fixture(scope="module")
def tok():
    return char_tokenizer()


@pytest.mark.parametrize("family", ["llama", "qwen2", "gemma2"])
def test_batched_matches_single(family, tok):
    model = tiny_model(family, tie=family == "gemma2")
    ids = option_ids(tok, "bare")
    a_batch, o_batch = fwd.run_shard(model, tok, PROMPTS, ids, batch_size=len(PROMPTS))
    _, mask = fwd.left_pad_batch(tok, PROMPTS)
    assert (mask == 0).any(), "fixture must exercise padding"
    a_single, o_single = fwd.run_shard(model, tok, PROMPTS, ids, batch_size=1)
    assert a_batch.shape == (len(PROMPTS), N_LAYERS, model.config.hidden_size)
    torch.testing.assert_close(a_batch, a_single)
    torch.testing.assert_close(o_batch, o_single)


@pytest.mark.parametrize("family", ["llama", "qwen2", "gemma2"])
def test_lens_on_collected_last_layer_matches_output(family, tok):
    model = tiny_model(family)
    ids = option_ids(tok, "bare")
    acts, opts = fwd.run_shard(model, tok, PROMPTS, ids, batch_size=3)
    with torch.no_grad():
        lens = lens_option_logits(model, acts[:, -1], N_LAYERS - 1, ids)
    torch.testing.assert_close(lens, opts)


def test_lens_option_logits_equals_full_lens():
    from crystal.lens import lens_logits

    model = tiny_model("gemma2", tie=True)
    r = torch.randn(3, model.config.hidden_size, generator=torch.Generator().manual_seed(3))
    ids = [10, 20, 30, 40]
    with torch.no_grad():
        torch.testing.assert_close(lens_option_logits(model, r, 1, ids), lens_logits(model, r, 1)[:, ids])


def test_right_padding_rejected(tok):
    model = tiny_model("llama")
    enc = tok(PROMPTS[:2], return_tensors="pt", padding=True, padding_side="right")
    with pytest.raises(ValueError, match="left-padded"):
        fwd.collect_resid(model, enc["input_ids"], enc["attention_mask"], option_ids(tok, "bare"))


def test_shards_and_resume(tmp_path, tok, monkeypatch):
    model = tiny_model("qwen2")
    ids = option_ids(tok, "bare")
    df = pd.DataFrame({"qid": [f"sk{i:05d}" for i in range(len(PROMPTS))], "prompt": PROMPTS})
    paths = fwd.run_forward(model, tok, df, tmp_path, ids, shard_size=2, batch_size=2, meta={"x": 1}, log=lambda *_: None)
    assert [p.name for p in paths] == ["shard_00000.npz", "shard_00001.npz", "shard_00002.npz"]
    z = np.load(paths[1])
    assert z["qids"].tolist() == ["sk00002", "sk00003"]
    assert z["acts"].dtype == np.float16 and z["acts"].shape == (2, N_LAYERS, model.config.hidden_size)
    assert z["out_opt_logits"].shape == (2, 4)

    # interrupted run: delete one shard; only it is recomputed
    paths[1].unlink()
    calls = []
    orig = fwd.run_shard
    monkeypatch.setattr(fwd, "run_shard", lambda *a, **k: calls.append(1) or orig(*a, **k))
    fwd.run_forward(model, tok, df, tmp_path, ids, shard_size=2, batch_size=2, log=lambda *_: None)
    assert len(calls) == 1
    np.testing.assert_array_equal(np.load(paths[1])["acts"], z["acts"])

    # a changed row selection must not silently mix with old shards
    with pytest.raises(RuntimeError, match="different qids"):
        fwd.run_forward(model, tok, df.iloc[1:], tmp_path, ids, shard_size=2, batch_size=2, log=lambda *_: None)


def test_smoke_pipeline_on_tiny_model(tmp_path, tok):
    """forward -> shards -> scripts/smoke.py checks, end to end on a tiny model."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("smoke", "scripts/smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    model = tiny_model("gemma2", tie=True)
    ids = option_ids(tok, "bare")
    df = pd.DataFrame({"qid": [f"sk{i:05d}" for i in range(len(PROMPTS))], "prompt": PROMPTS,
                       "gold_idx": [0, 1, 2, 3, 0]})
    fwd.run_forward(model, tok, df, tmp_path, ids, shard_size=2, batch_size=2, log=lambda *_: None)
    qids, acts, out = smoke.load_shards(tmp_path)
    lines = []
    res = smoke.smoke(model, tok, df, qids, acts, out, ids, n_examples=3, report=lines.append)
    assert res["ranks"].shape == (len(PROMPTS), N_LAYERS)
    assert 0.0 <= res["acc_out"] <= 1.0
    assert any("OK" in l for l in lines)

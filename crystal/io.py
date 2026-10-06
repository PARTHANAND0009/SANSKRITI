"""Config and data loaders shared by every stage."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_run_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "config" / "run.yaml") as f:
        return yaml.safe_load(f)


def load_models_config(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "config" / "models.yaml") as f:
        return yaml.safe_load(f)["models"]


def model_config(key: str, path: str | Path | None = None) -> dict:
    models = load_models_config(path)
    if key not in models:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(models)}")
    return {"key": key, **models[key]}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def load_prompts(path: str | Path | None = None) -> pd.DataFrame:
    """All rows written by stage 0, including ambiguous_gold rows."""
    path = path or load_run_config()["paths"]["prompts"]
    df = pd.read_parquet(resolve(path))
    if "option_token_ids" in df.columns:
        df["option_token_ids"] = df["option_token_ids"].map(json.loads)
    return df


def load_analysis_set(path: str | Path | None = None) -> pd.DataFrame:
    """Rows used in analysis: stage 0 output minus ambiguous_gold rows."""
    df = load_prompts(path)
    return df[~df["ambiguous_gold"]].reset_index(drop=True)


def select_rows(split: str = "analysis", limit: int | None = None, path=None) -> pd.DataFrame:
    """The rows stage 2 runs on, in a fixed order (qid order). Shared by
    run/forward.py and scripts/smoke.py so both see the same questions."""
    if split == "analysis":
        df = load_analysis_set(path)
    elif split == "all":
        df = load_prompts(path)
    else:
        raise ValueError(f"split must be 'analysis' or 'all', got {split!r}")
    df = df.sort_values("qid").reset_index(drop=True)
    return df.head(limit) if limit else df


def load_entities(path: str | Path | None = None) -> pd.DataFrame:
    path = path or load_run_config()["paths"]["entities"]
    return pd.read_parquet(resolve(path))


def load_freq(path: str | Path | None = None) -> pd.DataFrame:
    path = path or load_run_config()["paths"]["freq"]
    return pd.read_parquet(resolve(path))


def load_acts(model_key: str, variant: str = "prompt", acts_dir: str | Path | None = None):
    """Concatenate all stage 2 shards for a model.

    Returns (qids [n], acts [n, L, d] float16, out_opt_logits [n, 4] float32).
    """
    acts_dir = resolve(acts_dir or load_run_config()["paths"]["acts"])
    shards = sorted((acts_dir / model_key / variant).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {acts_dir / model_key / variant}")
    qids, acts, out = [], [], []
    for s in shards:
        z = np.load(s, allow_pickle=False)
        qids.append(z["qids"])
        acts.append(z["acts"])
        out.append(z["out_opt_logits"])
    return np.concatenate(qids), np.concatenate(acts), np.concatenate(out)

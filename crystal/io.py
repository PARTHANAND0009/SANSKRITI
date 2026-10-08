"""Config and data loaders shared by every stage."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


# Prompt variant -> (prompt column, gold-index column) in prompts.parquet. cyc0..cyc3 put
# the gold answer at A..D so letter preferences cancel over the four; perm is a seeded
# shuffle. "prompt"/"prompt_permuted" are the names the pilot activations were written under.
VARIANTS = {
    "cyc0": ("prompt_cyc0", "gold_idx_cyc0"),
    "cyc1": ("prompt_cyc1", "gold_idx_cyc1"),
    "cyc2": ("prompt_cyc2", "gold_idx_cyc2"),
    "cyc3": ("prompt_cyc3", "gold_idx_cyc3"),
    "perm": ("prompt_permuted", "gold_idx_permuted"),
    "prompt": ("prompt", "gold_idx"),
    "prompt_permuted": ("prompt_permuted", "gold_idx_permuted"),
}
CYCLIC = ("cyc0", "cyc1", "cyc2", "cyc3")


def variant_columns(variant: str) -> tuple[str, str]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {sorted(VARIANTS)}")
    return VARIANTS[variant]


def load_run_config(path: str | Path | None = None) -> dict:
    """config/run.yaml; CRYSTAL_ACTS_DIR overrides paths.acts (preflight keeps its test
    activations out of acts/)."""
    with open(path or ROOT / "config" / "run.yaml") as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("CRYSTAL_ACTS_DIR"):
        cfg["paths"]["acts"] = os.environ["CRYSTAL_ACTS_DIR"]
    return cfg


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


def stable_seed(key: str, base_seed: int) -> int:
    """Seed derived from a string key, identical on every machine and Python version."""
    return int(hashlib.sha256(f"{base_seed}:{key}".encode()).hexdigest()[:16], 16)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def load_analysis_set(path: str | Path | None = None, include_leaks: bool = False) -> pd.DataFrame:
    """Stage 0 rows minus ambiguous_gold and (unless include_leaks) leaks_answer."""
    df = load_prompts(path)
    keep = ~df["ambiguous_gold"]
    if not include_leaks:
        keep &= ~df["leaks_answer"]
    return df[keep].reset_index(drop=True)


def select_rows(
    split: str = "analysis",
    limit: int | None = None,
    path=None,
    include_leaks: bool = False,
    sample: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """The rows stage 2 runs on, in qid order (run/forward.py and scripts/smoke.py must agree)."""
    if split == "analysis":
        df = load_analysis_set(path, include_leaks=include_leaks)
    elif split == "all":
        df = load_prompts(path)
    else:
        raise ValueError(f"split must be 'analysis' or 'all', got {split!r}")
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=seed)
    df = df.sort_values("qid").reset_index(drop=True)
    return df.head(limit) if limit else df


def load_entities(path: str | Path | None = None) -> pd.DataFrame:
    path = path or load_run_config()["paths"]["entities"]
    return pd.read_parquet(resolve(path))


def load_freq(path: str | Path | None = None) -> pd.DataFrame:
    path = path or load_run_config()["paths"]["freq"]
    return pd.read_parquet(resolve(path))


def read_shards(shard_dir: Path):
    """Concatenate the stage 2 shards in a directory.

    Returns:
        qids [n], acts [n, L+1, d] float16 (layer 0 = embeddings), out_opt_logits [n, 4] float32.
    """
    shards = sorted(Path(shard_dir).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    qids, acts, out = [], [], []
    for s in shards:
        z = np.load(s, allow_pickle=False)
        qids.append(z["qids"])
        acts.append(z["acts"])
        out.append(z["out_opt_logits"])
    return np.concatenate(qids), np.concatenate(acts), np.concatenate(out)


def load_acts(model_key: str, variant: str = "prompt", acts_dir: str | Path | None = None):
    """read_shards for acts/{model}/{variant}."""
    acts_dir = resolve(acts_dir or load_run_config()["paths"]["acts"])
    return read_shards(acts_dir / model_key / variant)

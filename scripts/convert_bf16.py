"""Make a bf16 local copy of a checkpoint published in fp32, one shard at a time.

For machines whose disk cannot hold the fp32 download (google/gemma-2-9b ships
8 fp32 shards, 36.97 GB; bf16 is ~18.5 GB). Each fp32 shard is downloaded,
cast with torch's round-to-nearest .to(bfloat16) - the same cast
from_pretrained(dtype=bfloat16) applies at load time - saved, and deleted
before the next one. Config, tokenizer and the shard index are copied as is.

  python scripts/convert_bf16.py google/gemma-2-9b models/gemma-2-9b-bf16
Then set `local_path: models/gemma-2-9b-bf16` for the model in config/models.yaml.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file, save_file


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("out_dir")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(args.repo_id)
    shards = sorted(f for f in files if f.endswith(".safetensors"))
    for f in files:
        if f.endswith((".json", ".model")) and not (out / f).exists():
            shutil.copy(hf_hub_download(args.repo_id, f), out / f)
    for s in shards:
        dst = out / s
        if dst.exists():
            print(f"{s}: exists, skipped")
            continue
        src = hf_hub_download(args.repo_id, s)
        sd = load_file(src)
        sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in sd.items()}
        tmp = dst.with_suffix(".tmp")
        save_file(sd, str(tmp), metadata={"format": "pt"})
        os.replace(tmp, dst)
        real = Path(src).resolve()
        Path(src).unlink()
        real.unlink(missing_ok=True)
        print(f"{s}: {len(sd)} tensors -> bf16, fp32 copy deleted")
    idx = out / "model.safetensors.index.json"
    if idx.exists():
        d = json.loads(idx.read_text())
        d.setdefault("metadata", {})["total_size"] = sum((out / s).stat().st_size for s in shards)
        idx.write_text(json.dumps(d, indent=2))
    cfg = out / "config.json"
    c = json.loads(cfg.read_text())
    c["torch_dtype"] = c["dtype"] = "bfloat16"
    cfg.write_text(json.dumps(c, indent=2))
    print(f"done: {out}")


if __name__ == "__main__":
    main()

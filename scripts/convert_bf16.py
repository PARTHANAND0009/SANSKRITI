"""Make a local bf16 copy of a checkpoint published in fp32, without an fp32 shard on disk.

For disks that cannot hold the fp32 download (google/gemma-2-9b: 36.97 GB fp32, ~18.5 GB
bf16). Each tensor is fetched with an HTTP range request and cast with .to(bfloat16), the
same round-to-nearest cast from_pretrained(dtype=bfloat16) applies at load time. Peak RAM is
one bf16 shard plus one fp32 tensor. Auth uses the token huggingface_hub already has.

  python scripts/convert_bf16.py google/gemma-2-9b models/gemma-2-9b-bf16
Then set `local_path: models/gemma-2-9b-bf16` for the model in config/models.yaml.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import get_token, hf_hub_download, hf_hub_url, list_repo_files
from safetensors.torch import save_file

log = logging.getLogger(__name__)

NP = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32}


def fetch(session, url: str, start: int, end: int, tries: int = 6) -> bytes:
    """Bytes [start, end) of url."""
    for k in range(tries):
        try:
            r = session.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=300)
            if r.status_code == 206 and len(r.content) == end - start:
                return r.content
            err = f"HTTP {r.status_code}, {len(r.content)} bytes"
        except requests.RequestException as e:
            err = str(e)
        time.sleep(min(2**k, 30))
    raise RuntimeError(f"range {start}-{end} of {url}: {err}")


def to_tensor(buf: bytes, dtype: str, shape) -> torch.Tensor:
    if dtype == "BF16":
        t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16)
    else:
        t = torch.from_numpy(np.frombuffer(buf, dtype=NP[dtype]).copy())
    return t.reshape(shape)


def convert_shard(session, url: str, dst: Path) -> int:
    n = struct.unpack("<Q", fetch(session, url, 0, 8))[0]
    header = json.loads(fetch(session, url, 8, 8 + n))
    base = 8 + n
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        s, e = info["data_offsets"]
        t = to_tensor(fetch(session, url, base + s, base + e), info["dtype"], info["shape"])
        out[name] = t.to(torch.bfloat16) if t.is_floating_point() else t
    tmp = dst.with_suffix(".tmp")
    save_file(out, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, dst)
    return len(out)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("out_dir")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(args.repo_id)
    for f in files:
        if f.endswith((".json", ".model")) and not (out / f).exists():
            shutil.copy(hf_hub_download(args.repo_id, f), out / f)
    session = requests.Session()
    tok = get_token()
    if tok:
        session.headers["Authorization"] = f"Bearer {tok}"
    shards = sorted(f for f in files if f.endswith(".safetensors"))
    for s in shards:
        dst = out / s
        if dst.exists():
            log.info(f"{s}: exists, skipped")
            continue
        t0 = time.time()
        k = convert_shard(session, hf_hub_url(args.repo_id, s), dst)
        log.info(f"{s}: {k} tensors -> bf16 ({dst.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s)")
    idx = out / "model.safetensors.index.json"
    if idx.exists():
        d = json.loads(idx.read_text())
        d.setdefault("metadata", {})["total_size"] = sum((out / s).stat().st_size for s in shards)
        idx.write_text(json.dumps(d, indent=2))
    cfg = out / "config.json"
    c = json.loads(cfg.read_text())
    c["torch_dtype"] = c["dtype"] = "bfloat16"
    cfg.write_text(json.dumps(c, indent=2))
    log.info(f"wrote {out}")


if __name__ == "__main__":
    main()

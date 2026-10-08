"""Results archive: package -> verify; tampering is caught."""
import importlib.util
import json
import tarfile

import pytest


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pkg = _load("scripts/package_results.py", "package_results")
ing = _load("results/ingest.py", "ingest")


def test_package_and_verify_roundtrip(tmp_path):
    out = tmp_path / "results_test.tar.gz"
    m = pkg.build(out)
    paths = {f["path"] for f in m["files"]}
    assert any(p.startswith("results/") for p in paths)
    assert not any(p.startswith("acts/") for p in paths)                 # activations excluded by default
    assert not any(p.endswith(".pt") for p in paths)                      # lens weights excluded by default
    work = tmp_path / "w"
    work.mkdir()
    m2 = ing.verify(out, work)
    assert len(m2["files"]) == len(m["files"])


def test_tampered_archive_is_rejected(tmp_path):
    out = tmp_path / "results_test.tar.gz"
    pkg.build(out)
    src = tmp_path / "src"
    with tarfile.open(out) as t:
        t.extractall(src, filter="data")
    manifest = json.loads((src / "MANIFEST.json").read_text())
    victim = src / manifest["files"][0]["path"]
    victim.write_bytes(victim.read_bytes() + b"x")
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as t:
        for p in src.rglob("*"):
            if p.is_file():
                t.add(p, arcname=str(p.relative_to(src)))
    (tmp_path / "w").mkdir()
    with pytest.raises(SystemExit, match="checksum mismatch"):
        ing.verify(bad, tmp_path / "w")


def test_unsafe_member_rejected():
    assert not ing.safe_member("../etc/passwd") and not ing.safe_member("/abs") and not ing.safe_member("src/x.py")
    assert ing.safe_member("results/a.csv") and ing.safe_member("MANIFEST.json")

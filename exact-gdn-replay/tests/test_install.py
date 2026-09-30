"""Exercise source-hash and exact-anchor guards without changing an image."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mbx_install", HERE / "install.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_once_rejects_missing_and_duplicate_anchors():
    assert module.once("abc", "b", "x") == "axc"
    for value in ["ac", "abbc"]:
        with pytest.raises(RuntimeError):
            module.once(value, "b", "x")


def test_hash_guard_fails_before_any_write(tmp_path):
    path = tmp_path / "target.py"
    path.write_text("original = True\n")
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="source mismatch"):
        module.integrate(
            tmp_path, {"integration_source_sha256": {"target.py": "0" * 64}}
        )
    assert path.read_bytes() == before
    assert not path.with_suffix(".py.before-mbx-replay").exists()


def test_exact_image_sources_apply_once(tmp_path):
    import vllm

    source = Path(vllm.__file__).parent
    manifest = json.loads((HERE / "source-manifest.json").read_text())
    for relative in manifest["integration_source_sha256"]:
        src = source / relative
        backup = src.with_suffix(src.suffix + ".before-mbx-replay")
        if backup.exists():
            src = backup
        dst = tmp_path / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    assert len(module.integrate(tmp_path, manifest)) == 2
    with pytest.raises(RuntimeError, match="source mismatch"):
        module.integrate(tmp_path, manifest)

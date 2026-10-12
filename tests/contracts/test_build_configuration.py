"""Standalone source snapshots and explicit CUDA build settings."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("risa_build_test", ROOT / "build.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.mark.parametrize("fail", [False, True])
def test_standalone_snapshot_preserves_sources(tmp_path, monkeypatch, fail):
    source = tmp_path / "renamed project 中文"
    (source / "src/risa_attention").mkdir(parents=True)
    (source / "VERSION").write_text("0.2.0rc1")
    (source / "pyproject.toml").write_text("project metadata")
    output = source / "custom-wheels"
    output.mkdir()
    (output / "old.whl").write_bytes(b"existing output")
    monkeypatch.setattr(builder, "ROOT", source)
    monkeypatch.setattr(builder, "cuda_compiler", lambda: Path("/toolkit/nvcc"))
    monkeypatch.setenv("RISA_CUDA_ARCHITECTURES", "80-real;120-real")
    monkeypatch.setenv("RISA_CUDA_TAG", "cu130")
    snapshots = []

    def run(command, *, env, cwd, check):
        assert command[1:4] == ["-I", "-m", "build"]
        snapshot = Path(command[-1])
        snapshots.append(snapshot)
        assert snapshot != source
        assert (snapshot / "VERSION").read_text().strip() == "0.2.0rc1+cu130"
        assert (snapshot / "src/risa_attention").is_dir()
        assert not (snapshot / "custom-wheels").exists()
        assert env["CMAKE_ARGS"] == "-DCMAKE_CUDA_ARCHITECTURES=80-real;120-real"
        if fail:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(builder.subprocess, "run", run)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            builder.build_wheel(output)
    else:
        builder.build_wheel(output)
    assert snapshots and not snapshots[0].exists()
    assert (source / "VERSION").read_text() == "0.2.0rc1"
    assert (output / "old.whl").read_bytes() == b"existing output"


@pytest.mark.parametrize("architectures", ["", "native", "75-real", "80-real;80-real", "120"])
def test_rejects_implicit_or_unsupported_architectures(tmp_path, monkeypatch, architectures):
    monkeypatch.setenv("RISA_CUDA_ARCHITECTURES", architectures)
    with pytest.raises(ValueError):
        builder.build_wheel(tmp_path / "wheels")


def test_command_does_not_require_torch_or_parent_project(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "build.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--wheel" in result.stdout


def test_version_never_uses_parent_repository(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (tmp_path / "VERSION").write_text("99.0.0")
    from scikit_build_core.metadata.regex import dynamic_metadata

    with pytest.raises(FileNotFoundError):
        dynamic_metadata(
            "version", {"input": str(project / "VERSION"), "regex": r"^(?P<value>\S+)\s*$"}
        )


def test_core_package_contents():
    assert (ROOT / "src/risa_attention/dense.py").is_file()
    assert (ROOT / "src/risa_attention/sparse.py").is_file()
    assert not (ROOT / "src/risa_attention/extras").exists()
    assert not (ROOT / "version_provider.py").exists()


def test_core_has_no_model_adapters():
    assert not (ROOT / "src/risa_attention/_external_mask.py").exists()
    assert not (ROOT / "csrc/kernels/block_mask.cu").exists()
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "flashvsr" not in metadata and "swiftvr" not in metadata
    for directory in ("src", "csrc", "ci", "comfyui-risa-attention"):
        for path in (ROOT / directory).rglob("*"):
            if path.suffix in (".py", ".cpp", ".cu", ".cuh", ".h", ".txt", ".sh"):
                text = path.read_text(encoding="utf-8").lower()
                assert not any(
                    name in text
                    for name in ("flashvsr", "swiftvr", "external_mask", "block_mask_csr")
                ), path

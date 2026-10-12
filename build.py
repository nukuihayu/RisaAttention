"""Build an independent RISA wheel with explicit CUDA targets; never install it."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD_IGNORE = shutil.ignore_patterns(
    ".git",
    "build",
    "dist",
    "__pycache__",
    "*.egg-info",
    ".venv",
    ".gitlab-results",
    ".gitlab-venv",
    ".uv-cache",
    ".pytest_cache",
    ".ruff_cache",
)


def cuda_compiler():
    explicit = os.environ.get("CUDACXX")
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    name = "nvcc.exe" if sys.platform == "win32" else "nvcc"
    if not cuda_home and sys.platform != "win32" and Path("/usr/local/cuda/bin/nvcc").is_file():
        cuda_home = "/usr/local/cuda"
    candidate = explicit or (str(Path(cuda_home) / "bin" / name) if cuda_home else name)
    compiler = Path(shutil.which(candidate) or candidate).resolve()
    if not compiler.is_file():
        raise RuntimeError("CUDA compiler not found; set CUDACXX or CUDA_HOME")
    return compiler


def build_wheel(output):
    architectures = os.environ.get("RISA_CUDA_ARCHITECTURES", "")
    if not re.fullmatch(r"[0-9]+-real(?:;[0-9]+-real)*", architectures):
        raise ValueError("Set explicit RISA_CUDA_ARCHITECTURES, e.g. 80-real;120-real")
    targets = architectures.split(";")
    if len(set(targets)) != len(targets) or any(int(sm[:-5]) < 80 for sm in targets):
        raise ValueError("Architectures must be unique SM80+ native targets")
    cuda = os.environ.get("RISA_CUDA_TAG", "")
    if not re.fullmatch(r"cu[0-9]+", cuda):
        raise ValueError("Set RISA_CUDA_TAG, e.g. cu130")
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+)?", version):
        raise ValueError("VERSION must be a base version without CUDA suffix")
    environment = {
        **os.environ,
        "CUDACXX": str(cuda_compiler()),
        "CMAKE_ARGS": f"-DCMAKE_CUDA_ARCHITECTURES={architectures}",
    }
    output = Path(output).resolve()
    # Keep output outside the copied source even for custom output directories.
    with tempfile.TemporaryDirectory(prefix="risa-wheel-") as temporary:
        source = Path(temporary) / "risa_attention"

        def ignore(directory, names):
            excluded = set(BUILD_IGNORE(directory, names))
            excluded.update(name for name in names if (Path(directory) / name).resolve() == output)
            return excluded

        shutil.copytree(ROOT, source, ignore=ignore)
        (source / "VERSION").write_text(f"{version}+{cuda}\n", encoding="utf-8")
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(output),
                str(source),
            ],
            cwd=temporary,
            env=environment,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path, metavar="OUTPUT_DIR")
    build_wheel(parser.parse_args().wheel)

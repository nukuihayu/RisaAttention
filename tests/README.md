# Tests

Build/install this checkout before running GPU tests. Install development tools with
`python -m pip install -r requirements-test.txt` after installing CUDA PyTorch.

```bash
python -m pytest tests/contracts -q
python -m pytest tests/cuda -q
ruff check .
ruff format --check .
clang-format --dry-run --Werror csrc/bindings.cpp csrc/launchers.h csrc/kernels/*.cu csrc/kernels/*.cuh csrc/kernels/*.h csrc/*.cuh
```

CPU contracts cover API behavior, torch.compile schemas, ComfyUI integration, build paths, and release
metadata. CUDA tests cover quantized output, masks, causal attention, LSE, permutations,
sparse support, probability mass, TMA fallbacks, CUDA Graphs, and value launch schedules.
Tests load the installed package; reinstall after source changes to avoid testing an old wheel.

Real large-offset tests require approximately 18 GiB free VRAM:

```bash
RISA_LARGE_GPU_TEST=1 python -m pytest tests/cuda -q
```

Benchmark commands and output schema are in [benchmarks](../benchmarks/README.md).

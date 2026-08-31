# Contributing

Changes should stay within the attention backend, Python API, benchmarks,
tests, or ComfyUI adapter. Model-specific cache logic and unrelated operators
belong in separate projects.

## Development build

Build a real architecture target for the GPU used by the tests:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc \
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=120-real" \
python -m pip install . --no-build-isolation --force-reinstall --no-deps
python -m pytest -q -p no:cacheprovider
```

Use another `NN-real` value when testing a different SM. Do not add PTX as an
implicit compatibility path.

## Kernel changes

A kernel patch should include a benchmark case that exercises the changed
path and a correctness test for its numerical contract. Report fused median
and p90 latency, not only kernel or prequantized timing. Also check output
error, temporary memory and cubin size.

Equivalent variants that do not improve repeatable end-to-end latency should
be removed rather than retained behind a runtime or architecture branch. An
approximate change needs model-level quality evidence in addition to tensor
error metrics.

Keep upstream copyright and license headers when modifying derived CUDA code.
Document any new source or paper in `docs/design.md` and state whether the
implementation is exact, adapted, or only inspired by it.

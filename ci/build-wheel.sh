#!/usr/bin/env bash
set -euo pipefail
target="${CUDA_TAG}-py${PYTHON_VERSION//./}"
if [[ "${BUILD_TARGET:-all}" != all && "$BUILD_TARGET" != "$target" && "$BUILD_TARGET" != *-"$target" ]]; then
    echo "Not selected: $target"
    exit 0
fi
export MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1
export SKBUILD_BUILD_TOOL_ARGS="-j1"
export NVCC_PREPEND_FLAGS="--threads=1 --split-compile=1"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl
curl --fail --silent --show-error --location https://astral.sh/uv/0.8.15/install.sh |
    env UV_INSTALL_DIR=/usr/local/bin sh
uv venv --python "$PYTHON_VERSION" .gitlab-venv
uv pip install --python .gitlab-venv/bin/python -r ci/build-requirements.txt
.gitlab-venv/bin/python ci/wheels.py build

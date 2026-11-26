#!/usr/bin/env bash
# Build gsplat's CUDA extension ahead of time. No prebuilt gsplat wheel exists for torch 2.7 / CUDA 12.8, which is the
# first stack with Blackwell (sm_120) kernels.
#   env/build_gsplat.sh PYTHON WHEEL_DIR
# PYTHON must already hold torch, setuptools and ninja from main.lock; nvcc must be on PATH (12.8+ for sm_100/sm_120).
# TORCH_CUDA_ARCH_LIST picks the GPUs; the default covers Ampere (A100, A6000, A4000), Ada, Hopper and Blackwell.
set -euo pipefail
py=$1; out=$2
ver=1.5.3
sha=343f080c470e891ba3c697c03bdf0952cb982d1d535fdc24adb23c2cb4fba906
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;10.0;12.0+PTX}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/gsplat-$ver.tar.gz" "https://files.pythonhosted.org/packages/source/g/gsplat/gsplat-$ver.tar.gz"
echo "$sha  $tmp/gsplat-$ver.tar.gz" | sha256sum -c --quiet -
echo "building gsplat $ver for $TORCH_CUDA_ARCH_LIST with $(nvcc --version | tail -1)"
uv build --wheel --no-build-isolation --python "$py" -o "$out" "$tmp/gsplat-$ver.tar.gz"

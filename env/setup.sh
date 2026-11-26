#!/usr/bin/env bash
# Colab: build pinned uv venvs from the same lock files as the Apptainer images, independent of Colab's own image.
set -euo pipefail
ROOT=/content/gs; cd $ROOT
export UV_LINK_MODE=copy
command -v uv >/dev/null || pip install -q uv
mkdir -p envs
BACKEND=${BACKEND:-pi4}
if [ ! -f envs/main/.ok ]; then
  uv venv -q --python 3.10 envs/main --clear
  uv pip install -q --python envs/main/bin/python --index-strategy unsafe-best-match -r env/main.lock
  # no prebuilt gsplat for this torch: compile it for this VM's GPU only (a few minutes)
  export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
  TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1) bash env/build_gsplat.sh envs/main/bin/python envs/wheels
  uv pip install -q --python envs/main/bin/python --no-deps envs/wheels/gsplat-*.whl
  touch envs/main/.ok
fi
if [ "$BACKEND" = colmap ] && [ ! -f envs/main/.loma ]; then
  # LoMa's declared deps clash with the lock and only its CLI needs the missing ones
  uv pip install -q --python envs/main/bin/python --no-deps "lomatch @ git+https://github.com/davnords/LoMa@ecc947d44ddc9d50387e887bbedc743d65ffac00"
  touch envs/main/.loma
fi
if [ "$BACKEND" = pi4 ] && [ ! -f envs/pi4/.ok ]; then  # Pi4 is non-commercial: never installed for colmap
  uv venv -q --python 3.11 envs/pi4 --clear
  uv pip install -q --python envs/pi4/bin/python --index-strategy unsafe-best-match -r env/pi4.lock
  touch envs/pi4/.ok
fi
envs/main/bin/python -c "import torch,gsplat;from gsplat import csrc;print('main',torch.__version__,gsplat.__version__,torch.cuda.is_available())"
[ "$BACKEND" != pi4 ] || envs/pi4/bin/python -c "import torch;from pi4 import Pi4;print('pi4',torch.__version__,torch.cuda.is_available())"

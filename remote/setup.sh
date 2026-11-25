#!/usr/bin/env bash
# Builds two pinned uv venvs, independent of whatever Colab's image ships today.
set -euo pipefail
ROOT=/content/gs; cd $ROOT
export UV_LINK_MODE=copy
command -v uv >/dev/null || pip install -q uv
PI4_REV=32dfbea9ce6268ea17446208a804710df9475c32
clone() { # url rev dir
  [ -d "$3/.git" ] || git clone -q "$1" "$3"; git -C "$3" checkout -q "$2"; }
mkdir -p third_party envs

if [ ! -f envs/main/.ok ]; then
  uv venv -q --python 3.10 envs/main --clear
  uv pip install -q --python envs/main/bin/python --index-strategy unsafe-best-match -r remote/requirements-main.txt
  touch envs/main/.ok
fi
if [ "${BACKEND:-pi4}" = pi4 ] && [ ! -f envs/pi4/.ok ]; then  # Pi4 is non-commercial: not installed for colmap
  clone https://github.com/yyfz/Pi4 $PI4_REV third_party/pi4
  uv venv -q --python 3.11 envs/pi4 --clear
  (cd third_party/pi4 && uv pip install -q --python ../../envs/pi4/bin/python --index-strategy unsafe-best-match -r requirements-cu126.txt scipy==1.16.3)
  touch envs/pi4/.ok
fi
LOMA_REV=ecc947d44ddc9d50387e887bbedc743d65ffac00
if [ "${BACKEND:-pi4}" = colmap ] && [ ! -f envs/main/.loma ]; then
  clone https://github.com/davnords/LoMa $LOMA_REV third_party/loma
  uv pip install -q --python envs/main/bin/python --no-deps -e third_party/loma  # its deps are already in requirements-main
  uv pip install -q --python envs/main/bin/python tyro==0.9.17
  touch envs/main/.loma
fi
envs/main/bin/python -c "import torch,gsplat;print('main',torch.__version__,gsplat.__version__,torch.cuda.is_available())"
[ "${BACKEND:-pi4}" != pi4 ] || envs/pi4/bin/python -c "import torch;from pi4 import Pi4;print('pi4',torch.__version__,torch.cuda.is_available())"

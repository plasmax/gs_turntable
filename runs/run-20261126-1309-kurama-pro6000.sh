#!/usr/bin/env bash
# Kurama turnaround (1920x1080, 250 frames) -> gaussian splat on one RTX Pro 6000, highest-quality settings.
# Run from any shell, tcsh included:
#   /job/nocturne/dev/sandbox/sandbox_mlast/work/mlast/git/gs_turntable/runs/run-20261126-1309-kurama-pro6000.sh [colmap|pi4] [OUT_DIR] [03_train.py flags, e.g. --iters 4000]
# Without OUT_DIR a new out/kurama_<backend>_<timestamp>/ is made. Extra flags override the defaults below; a resume
# must repeat them.
# Pause:  touch OUT_DIR/STOP   (training checkpoints within ~100 iterations, then the job exits 75)
# Resume: run this script again with the same backend and that OUT_DIR; finished stages are skipped.
# colmap (default) is commercially usable; pi4 is CC BY-NC 4.0, non-commercial.
# Writes out/kurama_<backend>_<YYYYmmdd-HHMMSS>/ (splat.ply, cameras.nk, splat_yup.ply, metrics.json, run.log).
# Estimate 2.5-4 h, almost all training; the log prints elapsed seconds every 1000 of the 60000 iterations.
set -euo pipefail
backend="${1:-colmap}"
[[ $# -gt 0 ]] && shift
resume=""
if [[ $# -gt 0 && $1 != -* ]]; then resume=$1; shift; fi
case "$backend" in colmap|pi4) ;; *) echo "backend must be colmap or pi4" >&2; exit 2;; esac

repo=/job/nocturne/dev/sandbox/sandbox_mlast/work/mlast/git/gs_turntable
sifs=/job/mlearn/dev/sandbox/sandbox_mlast/work/mlast/git/sifs/pro6000
seq=/job/nocturne/vault/vfx_image_sequence/film/gen000/gen000_1000/reviewcomp_visdev/KuramaTurnaround_default_1920x1080_ldn.nocturne.asset.661301/v002/main/gen000_1000_visdev_KuramaTurnaround_v002_main.%04d.png
export GS_TURNTABLE_MAIN_IMAGE=$sifs/gs-turntable-main.sif
export GS_TURNTABLE_PI4_IMAGE=$sifs/gs-turntable-pi4.sif
export GS_TURNTABLE_MODELS=/job/mlearn/dev/sandbox/sandbox_mlast/work/mlast/git/models

out="${resume:-$repo/out/kurama_${backend}_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$out"
out=$(cd "$out" && pwd)  # a relative OUT_DIR still works after the cd below
cd "$repo"
exec > >(tee -a "$out/run.log") 2>&1
[[ -n $resume ]] && echo "=== resuming $(date) ==="

gpu=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)
echo "host $(hostname)  gpu $gpu  commit $(git rev-parse --short HEAD 2>/dev/null || echo ?)  backend $backend"
echo "out $out"
[[ $gpu == *"PRO 6000"* ]] || echo "warning: not an RTX Pro 6000; --cap 3000000 may not fit in memory" >&2

# fail in seconds rather than hours if a model file is missing
if [[ $backend == colmap ]]; then
    ./apptainer-python-main scripts/download_models.py /models --check --only main
else
    ./apptainer-python-main scripts/download_models.py /models --check
fi

time ./run_local.sh "$seq" "$out" "$backend" --cap 3000000 --iters 60000 --sh 3 --lam_lpips 0.2 --holdout 8 "$@"

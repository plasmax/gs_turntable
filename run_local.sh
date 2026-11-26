#!/usr/bin/env bash
# Run the turntable pipeline on a Linux GPU box through the Apptainer images. No network needed.
#   ./run_local.sh VIDEO|SEQ.####.png OUT_DIR [colmap|pi4] [03_train.py flags...]
# Pause: touch OUT_DIR/STOP. Matting stops after the current frame, training checkpoints, and the run exits 75.
# Resume: run the same command again; finished stages are skipped and training continues from its checkpoint.
set -euo pipefail
if [[ $# -lt 2 ]]; then sed -n '2,5p' "${BASH_SOURCE[0]}"; exit 2; fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
video="$(realpath "$1")"
mkdir -p "$2"; out="$(realpath "$2")"
backend="${3:-colmap}"
shift 2; [[ $# -gt 0 ]] && shift
case "$backend" in colmap|pi4) ;; *) echo "backend must be colmap or pi4" >&2; exit 2;; esac
# make both locations visible in the container
export GS_TURNTABLE_BIND="$(dirname "$video"),$out${GS_TURNTABLE_BIND:+,$GS_TURNTABLE_BIND}"
p=pipelines/turntable
stop="$out/STOP"
if [[ -e $stop ]]; then echo "removing $stop: resuming"; rm -f "$stop"; fi

paused() { echo "PAUSED $1. Resume by rerunning the same command (OUT_DIR $out)."; exit 75; }
stage() {  # run one stage; exit 75 means it saw STOP and saved its progress
    local rc=0
    "$@" || rc=$?
    if [[ $rc == 75 ]]; then paused "inside $(basename "$2")"; fi
    if [[ $rc != 0 ]]; then exit $rc; fi
}
between() { if [[ -e $stop ]]; then paused "before $1"; fi; }

if [[ -f $out/work/meta.json ]]; then
    echo "matte: done, skipping"
else
    stage "$here/apptainer-python-main" $p/01_matte.py "$video" "$out/work" --stop "$stop"
fi
between pose
if [[ -f $out/work/pose_$backend.npz ]]; then
    echo "pose: done, skipping"
elif [[ $backend == pi4 ]]; then
    stage "$here/apptainer-python-pi4" $p/02_pose_pi4.py "$out/work"
else
    stage "$here/apptainer-python-main" $p/02_pose_colmap.py "$out/work"
fi
between training
if [[ -f $out/.train_done ]]; then
    echo "train: done, skipping"
else
    stage "$here/apptainer-python-main" $p/03_train.py "$out/work" "$out" --pose "pose_$backend.npz" --stop "$stop" "$@"
fi
"$here/apptainer-python-main" scripts/export_nuke.py "$out"
echo "done: $out/splat.ply, Nuke camera: $out/cameras.nk"

#!/usr/bin/env bash
# Run the turntable pipeline on a Linux GPU box through the Apptainer images. No network needed.
#   ./run_local.sh VIDEO|SEQ.####.png OUT_DIR [colmap|pi4] [03_train.py flags...]
set -euo pipefail
if [[ $# -lt 2 ]]; then sed -n '2,3p' "${BASH_SOURCE[0]}"; exit 2; fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
video="$(realpath "$1")"
mkdir -p "$2"; out="$(realpath "$2")"
backend="${3:-colmap}"
shift 2; [[ $# -gt 0 ]] && shift
case "$backend" in colmap|pi4) ;; *) echo "backend must be colmap or pi4" >&2; exit 2;; esac
# make both locations visible in the container
export GS_TURNTABLE_BIND="$(dirname "$video"),$out${GS_TURNTABLE_BIND:+,$GS_TURNTABLE_BIND}"
p=pipelines/turntable
"$here/apptainer-python-main" $p/01_matte.py "$video" "$out/work"
if [[ $backend == pi4 ]]; then
    "$here/apptainer-python-pi4" $p/02_pose_pi4.py "$out/work"
else
    "$here/apptainer-python-main" $p/02_pose_colmap.py "$out/work"
fi
"$here/apptainer-python-main" $p/03_train.py "$out/work" "$out" --pose "pose_$backend.npz" "$@"
echo "done: $out/splat.ply"

#!/usr/bin/env bash
# Run the turntable pipeline on a Linux GPU box through the Apptainer images. No network needed.
#   ./run_local.sh VIDEO|SEQ.####.png OUT_DIR [colmap|pi4] [03_train.py flags...]
# Pause: touch OUT_DIR/STOP. Matting stops after the current frame, training checkpoints, and the run exits 75.
# Resume: run the same command again; finished stages are skipped and training continues from its checkpoint.
# Extend: rerun into the same OUT_DIR with a larger --iters (other training flags must match).
# Stages: GS_TURNTABLE_STAGES=matte,match,map,train (the default: all, in order). Only "map" (COLMAP mapping, focal
# scan, visual hull) needs no GPU, so it can run on a CPU-only machine into the same OUT_DIR; pi4 does its poses in "match".
set -euo pipefail

# The whole script sits in a function that bash parses before running it, so updating the checkout (e.g. git pull
# on NFS) while a job runs cannot cut the job short.
main() {
    if [[ $# -lt 2 ]]; then sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 2; fi
    local here video out backend p stop stages
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    video="$(realpath "$1")"
    mkdir -p "$2"; out="$(realpath "$2")"
    backend="${3:-colmap}"
    shift 2; [[ $# -gt 0 ]] && shift
    case "$backend" in colmap|pi4) ;; *) echo "backend must be colmap or pi4" >&2; exit 2;; esac
    stages=",${GS_TURNTABLE_STAGES:-matte,match,map,train},"
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
    want() { [[ $stages == *",$1,"* ]] && { [[ ! -e $stop ]] || paused "before $1"; }; }

    if want matte; then
        if [[ -f $out/work/meta.json ]]; then echo "matte: done, skipping"
        else stage "$here/apptainer-python-main" $p/01_matte.py "$video" "$out/work" --stop "$stop"; fi
    fi
    if want match; then
        if [[ -f $out/work/pose_$backend.npz ]]; then echo "match: poses done, skipping"
        elif [[ $backend == pi4 ]]; then stage "$here/apptainer-python-pi4" $p/02_pose_pi4.py "$out/work"
        else stage "$here/apptainer-python-main" $p/02_pose_colmap.py "$out/work" --stage match; fi
    fi
    if want map && [[ $backend == colmap ]]; then
        if [[ -f $out/work/pose_$backend.npz ]]; then echo "map: poses done, skipping"
        else stage "$here/apptainer-python-main" $p/02_pose_colmap.py "$out/work" --stage map; fi
    fi
    if want train; then
        # always runs: continues from its checkpoint; a finished run only re-exports (or extends, given more --iters)
        stage "$here/apptainer-python-main" $p/03_train.py "$out/work" "$out" --pose "pose_$backend.npz" --stop "$stop" "$@"
        "$here/apptainer-python-main" scripts/export_nuke.py "$out"
        echo "done: $out/splat.ply, Nuke camera: $out/cameras.nk"
    fi
}
main "$@"
exit

#!/usr/bin/env bash
# Preview a run's training checkpoint without touching the run: splat.ply, compare.mp4, orbit.mp4, metrics.json and the
# Nuke files go to OUT_DIR/preview_it<N>_<time>/. Safe while the run is paused or training elsewhere (it reads a copy).
#   scripts/preview_checkpoint.sh OUT_DIR
# Uses the same Apptainer images and GS_TURNTABLE_* variables as run_local.sh. Any GPU with ~16 GB will do.
set -euo pipefail

main() {
    if [[ $# -ne 1 ]]; then sed -n '2,5p' "${BASH_SOURCE[0]}"; exit 2; fi
    local here out prev
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    out="$(realpath "$1")"
    [[ -f $out/train_ckpt.pt ]] || { echo "no checkpoint in $out" >&2; exit 1; }
    prev="$out/preview_$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$prev"
    echo "copying the checkpoint to $prev"
    cp "$out/train_ckpt.pt" "$prev/train_ckpt.pt"  # a copy: the run may replace its own at any time
    export GS_TURNTABLE_BIND="$out${GS_TURNTABLE_BIND:+,$GS_TURNTABLE_BIND}"
    "$here/apptainer-python-main" pipelines/turntable/03_train.py "$out/work" "$prev" --export_only
    "$here/apptainer-python-main" scripts/export_nuke.py "$prev"
    rm -f "$prev/train_ckpt.pt"
    local it
    it=$(grep -o '"iters": [0-9]*' "$prev/metrics.json" | grep -o '[0-9]*$' || echo x)
    mv "$prev" "$out/preview_it${it}_${prev##*preview_}"
    echo "preview: $out/preview_it${it}_${prev##*preview_}"
}
main "$@"
exit

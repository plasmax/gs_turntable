# Apptainer images

| Image | Definition | Contents |
|---|---|---|
| `gs-turntable-main.sif` | `containers/main.def` | matting, COLMAP + LoMa poses, gsplat training. No Pi4 |
| `gs-turntable-pi4.sif` | `containers/pi4.def` | Pi4 poses only. **CC BY-NC 4.0: non-commercial** |

The Pi4 path uses both images; the COLMAP path uses only `main`. Neither contains model weights or this repository's code:
the wrappers bind the checkout, so editing a script needs no rebuild.

## Build

Actions > Build Apptainer images > Run workflow (manual). Each image is uploaded as an artifact with its SHA-256 and the
list of installed packages. Private repositories on a free GitHub plan have far less artifact storage than these images need.

Locally, from the repository root: `sudo apptainer build gs-turntable-main.sif containers/main.def`

## Move to an offline network

On a machine with internet (standard-library Python, no GPU or container needed):

    python3 scripts/download_models.py models                # about 5 GB
    python3 scripts/download_models.py models --only main    # without Pi4

Copy the checkout, the `.sif` files and `models/` across, then verify: `python3 scripts/download_models.py models --check`

## Run

    ./run_local.sh turntable.mp4 out/turntable_colmap colmap
    ./run_local.sh turntable.mp4 out/turntable_pi4 pi4
    ./run_local.sh /path/shot.####.png out/shot_colmap colmap    # image sequence, any start frame
    ./apptainer-python-main pipelines/turntable/03_train.py --help

Defaults, overridable by environment variable:

| Variable | Default |
|---|---|
| `GS_TURNTABLE_MAIN_IMAGE`, `GS_TURNTABLE_PI4_IMAGE` | `containers/gs-turntable-{main,pi4}.sif` |
| `GS_TURNTABLE_MODELS` | `models/` (bound read-only) |
| `GS_TURNTABLE_CACHE` | `$TMPDIR/gs-turntable-$UID` (kernel and JIT caches) |
| `GS_TURNTABLE_BIND` | extra comma-separated binds; `/net` and `/job` are bound when present |

One pair of images runs on every Ampere to Blackwell GPU (sm_80 to sm_120: A4000, A6000, A100, L4, RTX 4090/6000 Ada,
H100, B200, RTX 50xx, RTX Pro 6000). The host needs an NVIDIA driver for CUDA 12.8 (main) and 12.9 (pi4), i.e. R575 or
newer; Blackwell needs that anyway. gsplat has no prebuilt wheel for this torch, so `main.def` compiles it in a build stage
(`env/build_gsplat.sh`, about 30 min on a GitHub runner) and fails if any of those architectures is missing.

Pi4 holds every frame in one pass when it fits and otherwise splits into passes that share anchor frames (about 125
frames at the default 672 px fit 16 GB); `--chunk N` forces a pass size. Training prints its peak GPU memory, which is the
number to watch when raising `--cap`.
`splat.sog` and the HTML viewer are made by `run.py` with Node and are not produced offline; `splat.ply` loads in SuperSplat.

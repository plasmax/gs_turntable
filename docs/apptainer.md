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
    ./apptainer-python-main pipelines/turntable/03_train.py --help

Defaults, overridable by environment variable:

| Variable | Default |
|---|---|
| `GS_TURNTABLE_MAIN_IMAGE`, `GS_TURNTABLE_PI4_IMAGE` | `containers/gs-turntable-{main,pi4}.sif` |
| `GS_TURNTABLE_MODELS` | `models/` (bound read-only) |
| `GS_TURNTABLE_CACHE` | `$TMPDIR/gs-turntable-$UID` (kernel and JIT caches) |
| `GS_TURNTABLE_BIND` | extra comma-separated binds; `/net` and `/job` are bound when present |

Needs an NVIDIA GPU with bfloat16 (Ampere or newer) for Pi4, and a driver for CUDA 12.4 (main) / 12.6 (pi4).
`splat.sog` and the HTML viewer are made by `run.py` with Node and are not produced offline; `splat.ply` loads in SuperSplat.

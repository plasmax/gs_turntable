# Turntable video -> gaussian splat

    python run.py turntable.mp4                     # Pi4 poses (non-commercial)
    python run.py turntable.mp4 --backend colmap    # permissively licensed path, Pi4 is never installed

Needs `colab` (google-colab-cli, logged in), Python 3.9+ and Node. All GPU work runs on a Colab VM, which is released at the end.
Outputs land in `out/<video name>_<backend>/`: `splat.ply`, `splat.sog`, `splat.html` (viewer), `compare.mp4` (source | splat),
`orbit.mp4` (free orbit), `metrics.json`, `cameras.npz`.

Offline, through Apptainer instead of Colab: see [docs/apptainer.md](docs/apptainer.md).

| Path | What |
|---|---|
| `run.py` | Colab driver: VM up, stages, download, splat export, VM down |
| `run_local.sh`, `apptainer-python-{main,pi4}` | the same stages on a local GPU through the images |
| `pipelines/turntable/01_matte.py` | BiRefNet HR matting + foreground decontamination; drops the repeated last frame of a loop |
| `pipelines/turntable/02_pose_pi4.py` | Pi4 on grey-backed crops: poses, focal, depth, seed points |
| `pipelines/turntable/02_pose_colmap.py` | SIFT + LoMa matches, COLMAP mapping at a fixed focal, focal scan, visual-hull seeds |
| `pipelines/turntable/03_train.py` | gsplat MCMC, pose/focal refinement, per-frame deformation and colour (training only) |
| `env/` | lock files shared by Colab and the images |
| `containers/`, `.github/workflows/` | Apptainer definitions and the manual build |
| `scripts/download_models.py` | pinned, checksummed model download |

Options: `--keep` leaves the VM up (it bills until `colab stop -s gstt`); `--stage train` restarts there on a kept VM;
`--train-args "--iters 12000 --holdout 8"` passes trainer flags (`--holdout 8` adds held-out metrics).

Notes
- The splat holds one consistent pose. Where the video's pose drifts, per-pixel error against the source is higher than a
  rigid fit, which instead ghosts limbs. `--train-args "--deform 0"` gives the rigid fit.
- One elevation was filmed, so the crown and undersides are extrapolated.
- Runs are seeded, but CUDA rasterization is not bit-deterministic.
- Known issue: on the sample clip the colmap poses jump 11.9 degrees at frame 98 where the video is smooth (Pi4: 3.3). Held-out
  PSNR is 21.7 dB against 24.3 for Pi4 (12k iterations, same settings), although the two splats look alike.
- Orbit focal is only weakly observable; the colmap backend scans it and takes the minimum of a shallow curve.

Licenses (code and weights as published; not legal advice)
- pi4 backend: Pi4 code and weights are CC BY-NC 4.0, so not for commercial use.
- colmap backend: COLMAP (BSD), LoMa (MIT, matcher Apache-2.0; `commercial_loma_B` weights, per their README trained only on
  commercially licensed data), gsplat (Apache-2.0), BiRefNet (MIT), LPIPS (BSD), pymatting and splat-transform (MIT).
- Not checked: the training data behind the BiRefNet and VGG (LPIPS) weights, and third-party dependencies of COLMAP builds.
  LoMa's bundled DaD detector is identical to the ordinary DaD weights, so this pipeline uses SIFT keypoints and never loads it.

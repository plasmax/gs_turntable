# Handoff: Kurama 1080p turntable splat quality (updated 2026-11-27)

## Goal
Get a sharp gaussian splat of a 250-frame 1920x1080 CG turntable (Kurama) on the farm RTX Pro 6000, at higher quality
than an overnight run that "looked promising but low res". Every high-quality attempt since has come out blurry or
garbage. The overnight settings now reproduce on the new image; the work is to raise quality from that baseline.

## User constraints (from ~/.claude/CLAUDE.md)
- Replies 1-2 sentences; the user runs several agent sessions at once.
- No Co-Authored-By / Claude-Session / "Generated with Claude Code" lines in commits or PRs.
- Never push to gitlab1.framestore.com. This repo's remote is GitHub (git@github.com:plasmax/gs_turntable.git), and
  the user asked for pushes there after changes.
- Farm commands must be tcsh. Pattern: `go <dir> && module unload fortress && ...` (`go` sets env and cds; required).

## Repo / environment
- Local checkout: I:\dev\gs_turntable (Windows, no farm access from here). Branch `gpu-blackwell-nuke-export`.
  Last commits: 0ad5aeb (--focal_lr), 0757c08 (--canon_frac), 805e1a4 (preview script, pause exits 0 for the farm).
- Farm checkout (NFS): /job/nocturne/dev/sandbox/sandbox_mlast/work/mlast/git/gs_turntable (user runs `git pull` there).
- SIFs: /job/mlearn/dev/sandbox/sandbox_mlast/work/mlast/git/sifs/pro6000/gs-turntable-{main,pi4}.sif
  (main: torch 2.7.1+cu128, gsplat 1.5.3 built from sdist for sm_80..sm_120; pi4: torch 2.12.1+cu129).
  The overnight run used the OLDER SIFs in .../sifs/ (older torch/gsplat build); exact versions not recorded.
- Models: /job/mlearn/dev/sandbox/sandbox_mlast/work/mlast/git/models
- Input: /job/nocturne/vault/vfx_image_sequence/film/gen000/gen000_1000/reviewcomp_visdev/KuramaTurnaround_default_1920x1080_ldn.nocturne.asset.661301/v002/main/gen000_1000_visdev_KuramaTurnaround_v002_main.%04d.png (frames 1001-1250)
- Run script: runs/run-20261126-1309-kurama-pro6000.sh `[colmap|pi4] [OUT_DIR] [--stages matte,match,map,train] [03_train flags]`.
  It hardcodes `--cap 3000000 --iters 60000 --sh 3 --lam_lpips 0.2 --holdout 8`; extra flags come later and override.
- Preview a checkpoint on the user's A4000 without touching the run (copies train_ckpt.pt, writes OUT/preview_it<N>_<time>/):
  `setenv GS_TURNTABLE_MAIN_IMAGE .../pro6000/gs-turntable-main.sif && setenv GS_TURNTABLE_MODELS .../models && scripts/preview_checkpoint.sh OUT_DIR`
- Pause: `touch OUT_DIR/STOP` (exit 75, mapped to 0 by the run script). Resume: rerun the same command; flags must
  match the checkpoint except a larger --iters.
- To reuse matting + cameras: `mkdir -p out/NEW && ln -s ../OLD/work out/NEW/work` and run with `--stages train`.

## Pipeline (colmap backend)
01_matte.py (BiRefNet HR + pymatting) -> 02_pose_colmap.py (LoMa matches, fixed-FOV COLMAP mapping, focal scan,
visual-hull seeds) -> 03_train.py (gsplat MCMC; per-frame pose deltas, shared focal (log-scale `lf`), smooth per-frame
deformation MLP + colour affine used only in training) -> export splat.ply (canonical = undeformed) -> scripts/export_nuke.py.
Exported splat is the CANONICAL one; `train_canonical` vs `train_deformed` gap in metrics.json shows how much training leaned on the deformation.

## Run history (all 1080p, crop 1920x916, N=249)
| run | settings | it14000 focal / rot / trn / |dx| | result |
|---|---|---|---|---|
| out/kurama_colmap (overnight, old SIF, old code) | cap 350k, sh 2, lpips 0.1, 30k iters, holdout 0, lam_def 0.001 | 0.9635 / 0.965deg / 0.0072 / 0.038 | canonical 21.37 dB, deformed 32.3 dB; user: "promising, looks low res" |
| out/kurama_colmap_20261126-140857 | HQ defaults (3M, sh3, lpips .2, 60k, holdout 8) | 0.6998 / 1.41 / 0.0235 / 0.224 | garbage (stopped ~21k+) |
| out/kurama_colmap_canon (work symlinked to 140857) | HQ + canon_frac 0.5 | 0.6964 / ? / ? / 0.0995 | "very blurry, slightly better" |
| out/kurama_colmap_f0 (work symlinked to 140857) | HQ + canon_frac 0.5 + focal_lr 0 | 1.0000 / 1.662 / 0.0285 / 0.1306 | "still terrible" |
| out/kurama_colmap_repro (work -> kurama_colmap, new SIF, commit 0ad5aeb, A6000) | overnight settings + canon_frac 0 | 0.9606 / 0.959 / 0.0071 / 0.041 | REPRODUCES overnight: canonical 20.73 dB, deformed 32.56 dB, 24 min. Log: out/log_261126_2314.md |
| out/kurama_pi4 (own matting from 02:38; pi4 pose stage + train) | overnight settings + canon_frac 0 | ~0.999 / ? / ? / ~0.008 | user: "not really any better than colmap_repro" despite near-1 focal and small dx |
| out/kr_c05 (work -> kurama_colmap) | repro + canon_frac 0.5 | ? | user: no better than colmap_repro |
| out/kr_d0 (work -> kurama_colmap) | repro + deform 0 | ? | user: no better than colmap_repro |
| out/kr_1m (work -> kurama_colmap) | repro + cap 1M | ? | user: only one MARGINALLY better than colmap_repro |

Overnight log for reference: loss ~0.01-0.05 throughout; rot 0.35deg at 2k -> 1.10deg at 29k; trn 0.004 -> 0.008; |dx| 0.02-0.05.
f0 log: n hits 3M by it 8000; loss 0.07-0.31 from it 8000 (spikes 0.28/0.31) - far above overnight; rot/trn climb
steadily (1.66deg, 0.0285 at 14k = ~4x overnight trn); |dx| grows 0.01 -> 0.13. Peak 11.3 GB.

Both pose stages chose fov 17.89 (focal ~3431 px); the focal scan has a diverged entry (huge value at fov 32), now treated as inf.

## What this rules in / out
- The build (new SIF, torch 2.7.1 / gsplat 1.5.3) is fine: kurama_colmap_repro matches the overnight log and result.
- Pi4 cameras are not the fix: kurama_pi4 looks about the same as the colmap repro.
- Focal drift is not the (only) cause: fixed focal still fails; pose deltas (rot/trn) and deformation still run away.
- canon_frac 0.5 did not fix it either.
- Remaining suspects, unresolved:
  1. HQ settings: 3M splats (densifies to 3M by 8k), sh 3, lpips 0.2 (x2 weight in code), 60k schedule (pose/deform
     LRs decay 2x slower at a given iteration), holdout 8.
  2. New camera solve in out/kurama_colmap_20261126-140857/work vs overnight's out/kurama_colmap/work.
  3. New build (torch 2.7.1 / gsplat 1.5.3 sdist build) vs old SIF.
  4. Possibly the overnight work dir was later overwritten: the user's pasted overnight log shows a new matte/pose run
     starting after the training finished. CHECKED: pose_colmap.npz 01:46, splat.ply 02:29, so the overnight cameras are
  intact (the later log lines were another run into a different folder or an aborted start).

## Current diagnosis
The exported (canonical, undeformed) splat is ~12 dB worse than the deformed training render (20.7 vs 32.6 dB), so the
deformation MLP soaks up detail and the canonical splat averages to blur. That gap, not splat count, is the first target.
Log fields: `focal xF` = training's multiplier on the pose stage focal; `|dx|` = mean per-frame deformation offset
(object ~1.8 units tall); `rot`/`trn` = mean per-frame pose corrections.

## Next (commands ready in runs/kurama-next-20261126.md)
Three single-change variants from the repro baseline, each into its own out/ dir with work -> ../kurama_colmap/work:
`kr_c05` (--canon_frac 0.5), `kr_d0` (--deform 0), `kr_1m` (--cap 1000000, 30k iters). Compare `train_canonical`
fgPSNR in metrics.json and compare.mp4. Then stack --cap 1M and --sh 3 on the winner, staying at 30k iterations.
All three ran overnight 2026-11-26/27: only kr_1m was marginally better than colmap_repro; kr_c05 and kr_d0 were not.

Seed clouds: `scripts/export_seeds.py WORK...` (commit 3d3771a) writes seeds_<b>_all/_used.ply and cams_<b>.ply in the
splat_yup.ply world; command in the same runs file. Only tested on synthetic data locally.

## Ideas not yet tried
- Stronger pose prior (code: `.01 * (rot^2 + trn^2)` per frame) or lower pose LR (2e-4) for long runs.
- Stronger --lam_def, or make the deformation LR schedule independent of --iters.
- Keep the 30k LR schedule when raising cap (iters sets the decay of all LRs and MCMC refine_stop = 0.8*iters).
- canon_frac defaults to 0.5 now; pass --canon_frac 0 for the overnight behaviour. --focal_lr default 1e-4 = old.

## Repo state
Commit 3d3771a is local only: the agent's `git push` was blocked by the permission classifier, so the user must push
(`git push origin HEAD`) before the farm can `git pull` export_seeds.py and the runs file.

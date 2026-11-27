# Kurama: next runs (2026-11-26)

Baseline: out/kurama_colmap_repro (overnight settings), canonical 20.73 dB vs deformed 32.56 dB.
Compare `train_canonical` fgPSNR in each OUT/metrics.json, and look at OUT/compare.mp4.

## Quality variants (each ~25 min on an A6000, ~6 GB; can run side by side)

Half the iterations undeformed:
```tcsh
go /job/nocturne/dev/sandbox/sandbox_mlast/work/mlast/git/gs_turntable && module unload fortress && git pull && mkdir -p out/kr_c05 && ln -s ../kurama_colmap/work out/kr_c05/work && ./runs/run-20261126-1309-kurama-pro6000.sh colmap out/kr_c05 --stages train --cap 350000 --sh 2 --lam_lpips 0.1 --iters 30000 --holdout 0 --canon_frac 0.5
```

No deformation:
```tcsh
go /job/nocturne/dev/sandbox/sandbox_mlast/work/mlast/git/gs_turntable && module unload fortress && git pull && mkdir -p out/kr_d0 && ln -s ../kurama_colmap/work out/kr_d0/work && ./runs/run-20261126-1309-kurama-pro6000.sh colmap out/kr_d0 --stages train --cap 350000 --sh 2 --lam_lpips 0.1 --iters 30000 --holdout 0 --canon_frac 0 --deform 0
```

1M splats, same 30k schedule:
```tcsh
go /job/nocturne/dev/sandbox/sandbox_mlast/work/mlast/git/gs_turntable && module unload fortress && git pull && mkdir -p out/kr_1m && ln -s ../kurama_colmap/work out/kr_1m/work && ./runs/run-20261126-1309-kurama-pro6000.sh colmap out/kr_1m --stages train --cap 1000000 --sh 2 --lam_lpips 0.1 --iters 30000 --holdout 0 --canon_frac 0
```

## Seed point clouds (colmap vs pi4)

Writes seeds_<backend>_all.ply, seeds_<backend>_used.ply and cams_<backend>.ply into each work dir,
Y-up with feet on y=0 like splat_yup.ply. No GPU needed.
```tcsh
go /job/nocturne/dev/sandbox/sandbox_mlast/work/mlast/git/gs_turntable && module unload fortress && git pull && setenv GS_TURNTABLE_MAIN_IMAGE /job/mlearn/dev/sandbox/sandbox_mlast/work/mlast/git/sifs/pro6000/gs-turntable-main.sif && setenv GS_TURNTABLE_MODELS /job/mlearn/dev/sandbox/sandbox_mlast/work/mlast/git/models && ./apptainer-python-main scripts/export_seeds.py out/kurama_colmap/work out/kurama_pi4/work
```

## Results
- out/kurama_pi4 (pi4 cameras, overnight settings, --canon_frac 0): not really any better than kurama_colmap_repro,
  although its focal stayed ~x0.999 and |dx| ~0.008. Camera solve is not the quality bottleneck; keep colmap.

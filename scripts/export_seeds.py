"""Export the point clouds 03_train.py starts from, as PLY files you can open next to splat_yup.ply.

  scripts/export_seeds.py WORK_DIR [WORK_DIR ...]

For every pose_*.npz in each WORK_DIR, writes into that WORK_DIR:
  seeds_<backend>_all.ply   every seed point from the pose stage
  seeds_<backend>_used.ply  after training's filters (radius, sparse-voxel outliers, 120k sample)
  cams_<backend>.ply        camera centres, coloured red (first frame) to blue (last)
All in the training world turned Y-up with the feet on y=0, like splat_yup.ply, so both backends line up roughly.
"""
import argparse, glob, json, math, os
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

ap = argparse.ArgumentParser()
ap.add_argument("work", nargs="+")
a = ap.parse_args()


def write(path, xyz, rgb):
    v = np.empty(len(xyz), dtype=[(k, "f4") for k in "xyz"] + [(k, "u1") for k in ("red", "green", "blue")])
    for k, c in zip("xyz", xyz.T):
        v[k] = c
    for k, c in zip(("red", "green", "blue"), np.clip(rgb, 0, 255).astype(np.uint8).T):
        v[k] = c
    PlyData([PlyElement.describe(v, "vertex")]).write(path)
    print(f"wrote {path} ({len(xyz)} points)")


for work in a.work:
    meta = json.load(open(f"{work}/meta.json"))
    files = sorted(glob.glob(f"{work}/rgba/*.png"))
    for npz in sorted(glob.glob(f"{work}/pose_*.npz")):
        b = os.path.basename(npz)[5:-4]
        z = np.load(npz)
        idx, c2w = z["idx"], z["c2w"].astype(np.float64)
        N = len(idx)
        # crop height as in 03_train.py: alpha bounding box over all used frames
        any_rows = np.zeros(meta["H"], bool)
        for i in idx:
            any_rows |= (np.asarray(Image.open(files[i]).getchannel("A")) > 8).any(1)
        ys = np.where(any_rows)[0]
        y0, y1 = max(ys[0] - 12, 0), min(ys[-1] + 13, meta["H"])
        h = (y1 - y0) // 2 * 2
        # canonical frame, copied from 03_train.py
        C = c2w[:, :3, 3]
        m0 = C.mean(0)
        Vt = np.linalg.svd(C - m0)[2]
        up = Vt[2]
        if (-c2w[:, :3, 1].mean(0)) @ up < 0:
            up = -up
        e1 = Vt[0]
        Rw = np.stack([e1, -up, np.cross(e1, -up)])
        xy = (C - m0) @ Vt[:2].T
        s_ = np.linalg.lstsq(np.c_[2 * xy, np.ones(N)], (xy ** 2).sum(1), rcond=None)[0]
        ctr = m0 + s_[:2] @ Vt[:2]
        radius = math.sqrt(s_[2] + s_[:2] @ s_[:2])
        scale = 1.8 / (h / float(z["K"][0, 0]) * radius)
        pts = (z["pts"] - ctr) @ Rw.T * scale
        rgb = z["rgb"].astype(np.float32)
        cam = (C - ctr) @ Rw.T * scale
        allp, allc = pts, rgb
        ok = np.linalg.norm(pts[:, [0, 2]], axis=1) < 1.5
        pts, rgb = pts[ok], rgb[ok]
        _, inv, cnt = np.unique(np.floor(pts / 0.03).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
        ok = cnt[inv.ravel()] > 8
        pts, rgb = pts[ok], rgb[ok]
        sel = np.random.default_rng(0).permutation(len(pts))[:120000]
        pts, rgb = pts[sel], rgb[sel]
        # Y-up (180 deg about X) with the feet on y=0, as splat_yup.ply
        foot = float(np.percentile(pts[:, 1], 99.8))
        yup = lambda p: (p - [0, foot, 0]) * [1, -1, -1]
        print(f"{b}: {len(allp)} seeds, {len(pts)} used, {N} cameras, orbit radius {radius * scale:.2f}")
        write(f"{work}/seeds_{b}_all.ply", yup(allp), allc)
        write(f"{work}/seeds_{b}_used.ply", yup(pts), rgb)
        t = np.linspace(0, 1, N)[:, None]
        write(f"{work}/cams_{b}.ply", yup(cam), (1 - t) * [255, 0, 0] + t * [0, 0, 255])

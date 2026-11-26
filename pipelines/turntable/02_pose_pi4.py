"""Feed-forward camera poses + pointmaps for the matted frames from Pi4. NON-COMMERCIAL: Pi4 is CC BY-NC 4.0.

The background is replaced by flat grey, so "spinning object" and "orbiting camera" are indistinguishable
to the network and it returns an orbit. Frames are cropped to a centred square for resolution.
Writes pose_pi4.npz: c2w (N,4,4) OpenCV, K (3,3) full-frame px, pts/rgb seed cloud, depth maps.
"""
import argparse, glob, json, os
import numpy as np, cv2, torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--size", type=int, default=672)
ap.add_argument("--stride", type=int, default=1); ap.add_argument("--suffix", default="")
a = ap.parse_args()
meta = json.load(open(f"{a.out}/meta.json")); W, H = meta["W"], meta["H"]
n = meta["n"] - (1 if meta["loop"] else 0)
idx = np.arange(0, n, a.stride); S = min(W, H)
files = sorted(glob.glob(f"{a.out}/grey/*.png")); rg = sorted(glob.glob(f"{a.out}/rgba/*.png"))
x0, y0 = (W - S) // 2, (H - S) // 2
sz = a.size // 112 * 112  # multiple of 16
rgb = np.stack([np.asarray(Image.open(files[i]).convert("RGB").crop((x0, y0, x0 + S, y0 + S)).resize((sz, sz), Image.LANCZOS)) for i in idx])
al = np.stack([np.asarray(Image.open(rg[i]).crop((x0, y0, x0 + S, y0 + S)).resize((sz, sz), Image.LANCZOS))[..., 3] for i in idx]) / 255.

from pi4 import Pi4
local = os.environ.get("GS_MODELS") and os.path.join(os.environ["GS_MODELS"], "pi4")  # offline copy from scripts/download_models.py
net = (Pi4.from_pretrained(local) if local else Pi4.from_pretrained("yyfz233/Pi4", revision="59f307317fb630c9c5a5aa635038d98ed106aaf5")).cuda().eval()
x = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float().div(255)[None].cuda()
with torch.inference_mode():  # no autograd graph: 125+ frames fit a 16 GB card
    p = net(x)
lp = p["local_points"][0].float().cpu().numpy(); c2w = p["camera_poses"][0].float().cpu().numpy(); conf = p["confidence"][0].float().cpu().numpy()
# shared focal (px @ sz) by robust fit of u = f X/Z + c on the object
v, u = np.mgrid[0:sz, 0:sz].astype(np.float32); c = (sz - 1) / 2
m = (al > 0.9) & (lp[..., 2] > 1e-4)
ux = (u - c)[None].repeat(len(idx), 0)[m]; vy = (v - c)[None].repeat(len(idx), 0)[m]
X = lp[..., 0][m] / lp[..., 2][m]; Y = lp[..., 1][m] / lp[..., 2][m]
r = (ux * X + vy * Y) / np.maximum(X * X + Y * Y, 1e-9); f = float(np.median(r[(X * X + Y * Y) > 0.0025]))
per = [float(np.median(((ux_ * X_ + vy_ * Y_) / np.maximum(X_ * X_ + Y_ * Y_, 1e-9)))) for ux_, vy_, X_, Y_ in
       [((u - c)[mm], (v - c)[mm], l[..., 0][mm] / l[..., 2][mm], l[..., 1][mm] / l[..., 2][mm]) for mm, l in zip(m, lp)]]
print("per-frame focal spread", np.percentile(per, [0, 25, 50, 75, 100]))
depth = lp[..., 2]
K = np.array([[f * S / sz, 0, W / 2], [0, f * S / sz, H / 2], [0, 0, 1]], np.float64)
print("focal px (full frame)", K[0, 0], "vfov", np.degrees(2 * np.arctan(H / 2 / K[0, 0])))

# seed cloud
pts, cols = [], []
v, u = np.mgrid[0:sz, 0:sz].astype(np.float32); ray = np.stack([(u + .5 - sz / 2) / f, (v + .5 - sz / 2) / f, np.ones_like(u)], -1)
for k in range(0, len(idx), max(1, len(idx) // 60)):
    e = cv2.erode((al[k] > 0.9).astype(np.uint8), np.ones((5, 5))) > 0
    g = e & (conf[k] > np.percentile(conf[k][e], 20)); q = ray[g] * depth[k][g][:, None]
    s = np.random.default_rng(k).permutation(len(q))[:8000]
    pts.append(q[s] @ c2w[k, :3, :3].T + c2w[k, :3, 3]); cols.append(rgb[k][g][s])
np.savez_compressed(f"{a.out}/pose_pi4{a.suffix}.npz", c2w=c2w, K=K, idx=idx, pts=np.concatenate(pts).astype(np.float32), rgb=np.concatenate(cols),
                    depth=depth.astype(np.float16)[:, ::2, ::2], f_crop=f, sz=sz)

# orbit diagnostics
C = c2w[:, :3, 3]; ctr = C.mean(0); U, D, Vt = np.linalg.svd(C - ctr); nrm = Vt[2]
xy = (C - ctr) @ Vt[:2].T; A = np.c_[2 * xy, np.ones(len(xy))]; s = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)[0]
R = np.sqrt(s[2] + s[:2] @ s[:2]); th = np.unwrap(np.arctan2(*(xy - s[:2]).T[::-1])); rad = np.linalg.norm(xy - s[:2], axis=1)
print(f"orbit: radius {R:.4f}  radial rms {np.std(rad)/R*100:.2f}%  off-plane rms {np.std((C-ctr)@nrm)/R*100:.2f}%  sweep {np.degrees(th[-1]-th[0]):.1f} deg")
print("angle every 12:", np.round(np.degrees(th - th[0])[::max(1, 12 // a.stride)]).astype(int).tolist())

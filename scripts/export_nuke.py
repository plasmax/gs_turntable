"""Solved cameras -> Nuke: an animated Camera2 node (cameras.nk, paste into the node graph) and a .chan file.

    python scripts/export_nuke.py OUT_DIR [--first 1001] [--native]

Reads OUT_DIR/cameras.npz from 03_train.py. By default the scene is turned 180 degrees about X into Nuke's Y-up
world, and OUT_DIR/splat_yup.ply is written in that same world (positions, rotations and view-dependent colour), so
the camera and the splat line up with no extra transform. --native keeps the training world (-Y up) and writes no ply.
Keys sit on source frame numbers: frame 1 is the first video frame; --first overrides (e.g. for an image sequence
from before cameras.npz recorded its start frame). The principal point is the image centre, so no window offset.
"""
import argparse, math, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--first", type=int, help="frame number of the first source frame (default: recorded, else 1)")
ap.add_argument("--native", action="store_true", help="keep the training world (-Y up); no splat_yup.ply")
ap.add_argument("--haperture", type=float, default=36.0, help="mm; focal is written in the same units")
ap.add_argument("--name", default="gs_turntable_cam")
a = ap.parse_args()

z = np.load(os.path.join(a.out, "cameras.npz"))
c2w, W, H, fpx = z["c2w"].astype(np.float64), int(z["W"]), int(z["H"]), float(z["focal"])
first = a.first if a.first is not None else int(z["first"]) if "first" in z else 1
frames = first + z["idx"].astype(int)

F = np.eye(3) if a.native else np.diag([1., -1., -1.])  # world: 180 deg about X, -Y up -> Y up
CV2GL = np.diag([1., -1., -1.])  # camera: OpenCV (y down, looks +Z) -> Nuke (y up, looks -Z)
R = F @ c2w[:, :3, :3] @ CV2GL
T = c2w[:, :3, 3] @ F.T

# Nuke rot_order ZXY: R = Ry @ Rx @ Rz. Orbit cameras spin about Y, so X and Z stay small and nothing gimbal-locks.
rx = np.degrees(np.arcsin(np.clip(-R[:, 1, 2], -1, 1)))
ry = np.degrees(np.unwrap(np.arctan2(R[:, 0, 2], R[:, 2, 2])))
rz = np.degrees(np.unwrap(np.arctan2(R[:, 1, 0], R[:, 1, 1])))
focal = fpx * a.haperture / W
vaperture = a.haperture * H / W
vfov = math.degrees(2 * math.atan(H / 2 / fpx))


def curve(v):
    return "{curve " + " ".join(f"x{f} {x:.6f}" for f, x in zip(frames, v)) + "}"


nk = f"""Camera2 {{
 inputs 0
 rot_order ZXY
 translate {{{curve(T[:, 0])} {curve(T[:, 1])} {curve(T[:, 2])}}}
 rotate {{{curve(rx)} {curve(ry)} {curve(rz)}}}
 focal {focal:.6f}
 haperture {a.haperture:.6f}
 vaperture {vaperture:.6f}
 label "{W}x{H}, frames {frames[0]}-{frames[-1]}"
 name {a.name}
}}
"""
open(os.path.join(a.out, "cameras.nk"), "w", newline="\n").write(nk)
with open(os.path.join(a.out, "cameras.chan"), "w", newline="\n") as f:  # Camera > Import chan file (ZXY)
    for row in zip(frames, *T.T, rx, ry, rz):
        f.write(f"{row[0]} " + " ".join(f"{v:.6f}" for v in row[1:]) + f" {vfov:.6f}\n")
print(f"{len(frames)} cameras, frames {frames[0]}-{frames[-1]}, focal {focal:.3f} mm on a {a.haperture:g} mm back "
      f"({fpx:.1f} px, vfov {vfov:.2f} deg), {'training' if a.native else 'Y-up'} world")

if not a.native:
    from plyfile import PlyData, PlyElement
    ply = PlyData.read(os.path.join(a.out, "splat.ply"))
    v = ply["vertex"].data.copy()
    for k in ("y", "z", "ny", "nz"):
        v[k] = -v[k]
    # quaternion (w,x,y,z): q' = (0,1,0,0) * q
    w, x, y, q = (v[f"rot_{k}"].copy() for k in range(4))
    v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"] = -x, w, -q, y
    # SH coefficients: f'(d) = f(F d) with F = diag(1,-1,-1); per basis function (3DGS order) the sign under y,z -> -y,-z
    sign = [-1, -1, 1,  -1, 1, 1, -1, 1,  -1, 1, -1, -1, 1, -1, 1]
    for c in range(3):
        for k, s in enumerate(sign):
            if s < 0 and f"f_rest_{c * 15 + k}" in v.dtype.names:
                v[f"f_rest_{c * 15 + k}"] = -v[f"f_rest_{c * 15 + k}"]
    PlyData([PlyElement.describe(v, "vertex")]).write(os.path.join(a.out, "splat_yup.ply"))
    print("wrote splat_yup.ply (same world as the camera)")

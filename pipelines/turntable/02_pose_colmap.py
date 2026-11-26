"""Camera poses from COLMAP with LoMa (default) or DISK+LightGlue matches. Permissive-license alternative to 02_pose.py.

Uses nothing from Pi4. Focal is weakly observable on a single-axis turntable and COLMAP's own focal refinement diverges,
so: map once at a fixed field of view, then slide the cameras along their axes to several focals, bundle adjust each,
and pick the minimum of the (shallow) error curve. Training refines it further.
Seeds are a tolerant visual hull plus the sparse points. Writes pose_colmap.npz (no depth: the depth loss is skipped).
"""
import argparse, glob, json, math, os, shutil, sqlite3
import numpy as np, cv2, torch, pycolmap
from PIL import Image
from scipy.ndimage import binary_erosion
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--matcher", choices=["loma", "disk"], default="loma")
ap.add_argument("--up", type=float, default=1.5, help="DISK: upscale before feature extraction")
ap.add_argument("--kpts", type=int, default=4096)
ap.add_argument("--fovs", default="16,12,22,30", help="vertical FOV guesses for mapping, tried until one registers the clip")
ap.add_argument("--stage", choices=["all", "match", "map"], default="all",
                help="match: features and matches only (GPU); map: COLMAP mapping, focal scan and seeds from cached matches (CPU only)")
ap.add_argument("--hull", type=float, default=0.9, help="fraction of views that must agree; below 1 tolerates pose drift")
a = ap.parse_args()
torch.manual_seed(0)
np.random.seed(0)
meta = json.load(open(f"{a.out}/meta.json"))
W, H = meta["W"], meta["H"]
n = meta["n"] - (1 if meta["loop"] else 0)
grey = sorted(glob.glob(f"{a.out}/grey/*.png"))[:n]
rg = sorted(glob.glob(f"{a.out}/rgba/*.png"))[:n]
names = [os.path.basename(f) for f in grey]  # COLMAP image names are the real file names
alpha = np.stack([np.asarray(Image.open(f))[..., 3] for f in rg])
CACHE = f"{a.out}/colmap_feats_{a.matcher}.pt"
pairs = set()
for k in range(n):
    for s in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        j = (k + s) % n if meta["loop"] else k + s
        if j < n and j != k:
            pairs.add((min(k, j), max(k, j)))
pairs = sorted(pairs)


def on_subject(k, kp):
    m = cv2.erode((alpha[k] > 200).astype(np.uint8), np.ones((5, 5)))
    return m[np.clip(kp[:, 1].astype(int), 0, H - 1), np.clip(kp[:, 0].astype(int), 0, W - 1)] > 0


def match_disk():
    import kornia.feature as KF
    disk = KF.DISK.from_pretrained("depth").cuda().eval()
    lg = KF.LightGlue("disk").cuda().eval()
    size = torch.tensor([[W, H]]).cuda().float()
    kps, ds = [], []
    for k in tqdm(range(n), mininterval=10):
        im = np.asarray(Image.open(grey[k]).convert("RGB"))
        hh, ww = int(H * a.up) // 16 * 16, int(W * a.up) // 16 * 16
        x = torch.from_numpy(cv2.resize(im, (ww, hh), interpolation=cv2.INTER_CUBIC)).cuda().permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            f = disk(x, a.kpts * 2, pad_if_not_divisible=True)[0]
        kp = (f.keypoints.cpu().numpy() + .5) * [W / ww, H / hh]
        ok = np.where(on_subject(k, kp))[0][:a.kpts]
        kps.append(kp[ok].astype(np.float32))
        ds.append(f.descriptors[torch.tensor(ok).cuda()])
    out = {}
    for i, j in tqdm(pairs, mininterval=10):
        with torch.no_grad():
            r = lg({"image0": {"keypoints": torch.tensor(kps[i])[None].cuda(), "descriptors": ds[i][None], "image_size": size},
                    "image1": {"keypoints": torch.tensor(kps[j])[None].cuda(), "descriptors": ds[j][None], "image_size": size}})
        out[(i, j)] = r["matches"][0].cpu().numpy()
    return kps, out


def match_loma():
    """SIFT keypoints + LoMa-B descriptor/matcher, weights released as trained on commercially licensed data.

    LoMa's bundled detector is byte-identical to ordinary DaD, which that claim may not cover, so it is never loaded.
    """
    import loma.loma as L

    class NoDetector(torch.nn.Module):
        Cfg = staticmethod(lambda **kw: None)

        def __init__(self, cfg=None):
            super().__init__()

    L.DaD = NoDetector
    net = L.LoMa(L.LoMaB(weights_url=None))
    w = torch.hub.load_state_dict_from_url(
        "https://github.com/davnords/storage/releases/download/loma/commercial_loma_B.pt", map_location="cpu")
    w = w.get("model", w)
    r = net.load_state_dict({k: v for k, v in w.items() if not k.startswith("_detector")}, strict=False)
    assert not r.missing_keys, r.missing_keys[:5]
    net = net.cuda().eval()
    S = min(W, H)
    x0, y0 = (W - S) // 2, (H - S) // 2
    sift = cv2.SIFT_create(contrastThreshold=0.005)
    kps, ds = [], []
    for k in tqdm(range(n), mininterval=10):
        im = cv2.imread(grey[k])
        g = cv2.cvtColor(cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
        kp = np.array([(q.pt[0], q.pt[1], q.response) for q in sift.detect(g)])
        kp = kp[np.argsort(-kp[:, 2])][:, :2] / 2 + .25  # 2x image, integer pixel centres -> centres at .5
        kp = kp[on_subject(k, kp)]
        kp = kp[((kp >= [x0, y0]) & (kp < [x0 + S, y0 + S])).all(1)][:a.kpts]
        kp = np.unique(np.round(kp, 2), axis=0).astype(np.float32)
        crop = cv2.cvtColor(im[y0:y0 + S, x0:x0 + S], cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(cv2.resize(crop, (784, 784), interpolation=cv2.INTER_CUBIC)).permute(2, 0, 1)[None].float().cuda() / 255
        nk = torch.tensor((kp - [x0, y0]) / S * 2 - 1, dtype=torch.float32).cuda()[None]
        with torch.inference_mode():
            d = net._descriptor.describe_keypoints(t, nk)["descriptions"]
        kps.append(kp)
        ds.append((nk, d.float()))
    out = {}
    for i, j in tqdm(pairs, mininterval=10):
        with torch.inference_mode():
            m0 = L.filter_matches(net(ds[i][0], ds[j][0], ds[i][1], ds[j][1])["scores"], net.cfg.filter_threshold)[0][0]
        v = (m0 > -1).nonzero()[:, 0]
        out[(i, j)] = torch.stack([v, m0[v]], 1).cpu().numpy()
    return kps, out


if os.path.exists(CACHE):
    kps, matches = torch.load(CACHE, weights_only=False)
elif a.stage == "map":
    raise SystemExit(f"{CACHE} missing: run --stage match on a GPU machine first")
else:
    kps, matches = (match_loma if a.matcher == "loma" else match_disk)()
    torch.save((kps, matches), CACHE + ".part")
    os.replace(CACHE + ".part", CACHE)
if a.stage == "match":
    print(f"matches cached in {CACHE}; mapping (CPU only) runs with --stage map")
    raise SystemExit(0)
torch.cuda.empty_cache()
matches = {k: v.astype(np.uint32) for k, v in matches.items() if len(v) >= 20}
print(f"matcher {a.matcher}: median {int(np.median([len(k) for k in kps]))} keypoints, {len(matches)}/{len(pairs)} pairs, "
      f"median {int(np.median([len(v) for v in matches.values()]))} matches", flush=True)

# ---------- one mapping, calibrated, at a fixed focal ----------
db_path = f"{a.out}/colmap.db"
open(f"{a.out}/pairs.txt", "w").write("\n".join(f"{names[i]} {names[j]}" for i, j in matches))


def fov2f(v):
    return H / 2 / math.tan(math.radians(v) / 2)


def cw(im):
    c = im.cam_from_world
    return c() if callable(c) else c


def err(r):
    r.update_point_3d_errors()  # otherwise compute_mean_reprojection_error returns stale per-point values
    return r.compute_mean_reprojection_error()


rec = None
for fov in [float(v) for v in a.fovs.split(",")]:
    if os.path.exists(db_path):
        os.remove(db_path)
    db = pycolmap.Database(db_path)
    cam = pycolmap.Camera(model="SIMPLE_PINHOLE", width=W, height=H, params=[fov2f(fov), W / 2, H / 2])
    cam.camera_id = 1
    cam.has_prior_focal_length = True  # otherwise two-view geometry is uncalibrated and this focal is ignored
    db.write_camera(cam, use_camera_id=True)
    for k in range(n):
        db.write_image(pycolmap.Image(name=names[k], camera_id=1, image_id=k + 1), use_image_id=True)
        db.write_keypoints(k + 1, kps[k])
    for (i, j), q in matches.items():
        db.write_matches(i + 1, j + 1, q)
    db.close()
    pycolmap.verify_matches(db_path, f"{a.out}/pairs.txt")
    con = sqlite3.connect(db_path)
    ratio = {}
    for pid, rows in con.execute("select pair_id, rows from two_view_geometries"):
        jj = pid % 2147483647
        ii = (pid - jj) // 2147483647
        gap = min((jj - ii) % n, (ii - jj) % n)
        ratio.setdefault(gap, []).append(rows / len(matches[(ii - 1, jj - 1)]))
    con.close()
    print(f"fov {fov:g} inlier ratio by frame gap:", {g: round(float(np.median(v)), 2) for g, v in sorted(ratio.items())}, flush=True)
    sp = f"{a.out}/sparse_{a.matcher}_{int(fov)}"
    shutil.rmtree(sp, ignore_errors=True)
    os.makedirs(sp)
    o = pycolmap.IncrementalPipelineOptions()
    o.min_num_matches = 15
    o.multiple_models = True
    o.max_num_models = 4
    o.ba_refine_focal_length = False
    o.ba_refine_extra_params = False
    o.mapper.init_min_tri_angle = 8.0
    o.mapper.abs_pose_min_inlier_ratio = 0.15
    for r in pycolmap.incremental_mapping(db_path, f"{a.out}/grey", sp, options=o).values():
        print(f"start fov {fov:g}: registered {r.num_reg_images()}/{n}, {r.num_points3D()} pts, reproj {err(r):.3f}px", flush=True)
        if rec is None or r.num_reg_images() > rec.num_reg_images():
            rec = r
    if rec is not None and rec.num_reg_images() >= 0.95 * n:
        break
if rec is None or rec.num_reg_images() < 0.9 * n:
    raise SystemExit(f"COLMAP registered only {0 if rec is None else rec.num_reg_images()}/{n} frames; not usable for training")

# ---------- focal scan ----------
base = f"{a.out}/mapped_{a.matcher}"
os.makedirs(base, exist_ok=True)
rec.write(base)
Cs = np.array([cw(im).inverse().translation for im in rec.images.values()])
Ds = np.array([cw(im).inverse().rotation.matrix()[:, 2] for im in rec.images.values()])
ctr = np.linalg.solve(sum(np.eye(3) - np.outer(v, v) for v in Ds),
                      sum((np.eye(3) - np.outer(v, v)) @ p for v, p in zip(Ds, Cs)))


def at_fov(v):
    r = pycolmap.Reconstruction(base)
    c = list(r.cameras.values())[0]
    s_ = fov2f(v) / c.params[0]
    c.params = [fov2f(v), W / 2, H / 2]
    for im in r.images.values():
        T = cw(im)
        pos = ctr + (T.inverse().translation - ctr) * s_
        im.cam_from_world = pycolmap.Rigid3d(T.rotation, -T.rotation.matrix() @ pos)
    bo = pycolmap.BundleAdjustmentOptions()
    bo.refine_focal_length = False
    bo.refine_principal_point = False
    bo.refine_extra_params = False
    bo.solver_options.max_num_iterations = 60
    pycolmap.bundle_adjustment(r, bo)
    e = err(r)
    return r, (e if math.isfinite(e) and e < 100 else math.inf)  # a diverged solve counts as no fit


scan = {}
for v in (8, 12, 16, 20, 26, 32):
    scan[v] = at_fov(v)[1]
    print(f"focal scan fov {v}: {scan[v]:.4f}px" if math.isfinite(scan[v]) else f"focal scan fov {v}: diverged", flush=True)
vs = sorted(scan)
q = min(max(vs.index(min(scan, key=scan.get)), 1), len(vs) - 2)
near = [scan[v] for v in vs[q - 1:q + 2]]
co = np.polyfit(np.log(vs[q - 1:q + 2]), near, 2) if all(map(math.isfinite, near)) else [0, 0, 0]
best = math.exp(-co[1] / (2 * co[0])) if co[0] > 0 else min(scan, key=scan.get)
best = float(np.clip(best, vs[0], vs[-1]))
rec, e = at_fov(best)
print(f"chosen fov {best:.2f}: {e:.4f}px (scan spans {min(scan.values()):.3f}-{max(v for v in scan.values() if math.isfinite(v)):.3f}px, so focal is approximate)", flush=True)

# ---------- export ----------
items = sorted(((names.index(im.name), im) for im in rec.images.values()), key=lambda t: t[0])
idx = np.array([k for k, _ in items])
c2w = np.tile(np.eye(4), (len(items), 1, 1))
for q, (_, im) in enumerate(items):
    c2w[q, :3] = cw(im).inverse().matrix()
f = list(rec.cameras.values())[0].params[0]
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)

C = c2w[:, :3, 3]
D = c2w[:, :3, 2]
ctr = np.linalg.solve(sum(np.eye(3) - np.outer(v, v) for v in D), sum((np.eye(3) - np.outer(v, v)) @ p for v, p in zip(D, C)))
R = np.median(np.linalg.norm(C - ctr, axis=1))
half = R * H / (2 * f) * 1.1
G = 160
g = np.linspace(-half, half, G)
grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) + ctr
vote, colsum, colw = np.zeros(len(grid)), np.zeros((len(grid), 3)), np.zeros(len(grid))
use = np.unique(np.linspace(0, len(idx) - 1, 120).astype(int))
for q in use:
    p = (grid - c2w[q, :3, 3]) @ c2w[q, :3, :3]
    u = np.clip((p[:, 0] / p[:, 2] * f + W / 2).astype(int), 0, W - 1)
    v = np.clip((p[:, 1] / p[:, 2] * f + H / 2).astype(int), 0, H - 1)
    inside = alpha[idx[q]][v, u] > 127
    vote += inside
    im = np.asarray(Image.open(rg[idx[q]]))[..., :3]
    colsum[inside] += im[v[inside], u[inside]]
    colw += inside
occ = (vote / len(use) >= a.hull).reshape(G, G, G)
shell = (occ & ~binary_erosion(occ, iterations=3)).reshape(-1)
hp, hc = grid[shell], (colsum / np.maximum(colw, 1)[:, None])[shell]
if len(hp) < 1000:
    raise SystemExit(f"visual hull nearly empty ({len(hp)} voxels): poses or mattes are inconsistent; try a lower --hull")
rep = int(np.ceil(120000 / len(hp)))
hp, hc = np.repeat(hp, rep, 0), np.repeat(hc, rep, 0)
hp = hp + np.random.uniform(-.5, .5, hp.shape) * (g[1] - g[0])
sp = np.array([p.xyz for p in rec.points3D.values()])
sc = np.array([p.color for p in rec.points3D.values()])
print(f"hull seeds {len(hp)} ({shell.sum()} voxels), sparse points {len(sp)}")
np.savez_compressed(f"{a.out}/pose_colmap.npz", c2w=c2w, K=K, idx=idx, pts=np.concatenate([hp, sp]).astype(np.float32),
                    rgb=np.concatenate([hc, sc]).clip(0, 255).astype(np.uint8))

m0 = C.mean(0)
Vt = np.linalg.svd(C - m0)[2]
xy = (C - m0) @ Vt[:2].T
s = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy ** 2).sum(1), rcond=None)[0]
Rr = np.sqrt(s[2] + s[:2] @ s[:2])
th = np.unwrap(np.arctan2(xy[:, 1] - s[1], xy[:, 0] - s[0]))
rad = np.linalg.norm(xy - s[:2], axis=1)
print(f"orbit: radial rms {np.std(rad)/Rr*100:.2f}%  off-plane rms {np.std((C-m0)@Vt[2])/Rr*100:.2f}%  sweep {np.degrees(th[-1]-th[0]):.1f} deg")

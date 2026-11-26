"""Gaussian splat training tuned for generated / imperfect character turntables.

* poses + shared focal refined jointly; angle-balanced frame sampling (uneven spin speed)
* smooth per-frame deformation + colour affine, used only during training: absorbs pose drift, breathing and
  flicker, so the exported canonical splat stays sharp instead of averaging
* SH evaluated with jittered elevation: studio light is fixed to the camera, so shading is a function of orbit
  azimuth; jitter stops SH exploding at elevations the video never shows
* soft-alpha supervision on random backgrounds: no floaters, clean translucent edges
"""
import argparse, glob, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import imageio, lpips
from PIL import Image
from scipy.spatial import cKDTree
from gsplat import rasterization, MCMCStrategy
from plyfile import PlyData, PlyElement

ap = argparse.ArgumentParser()
ap.add_argument("work")
ap.add_argument("out")
ap.add_argument("--pose", default="pose_pi4.npz")
ap.add_argument("--iters", type=int, default=30000)
ap.add_argument("--cap", type=int, default=350000)
ap.add_argument("--sh", type=int, default=2)
ap.add_argument("--el_jitter", type=float, default=50., help="deg; 2-sigma of elevation jitter for SH lookup")
ap.add_argument("--deform", type=int, default=1)
ap.add_argument("--lam_def", type=float, default=0.001)
ap.add_argument("--lam_alpha", type=float, default=0.3)
ap.add_argument("--lam_lpips", type=float, default=0.1)
ap.add_argument("--lam_depth", type=float, default=0.1)
ap.add_argument("--lam_flat", type=float, default=0.001)
ap.add_argument("--holdout", type=int, default=0, help="hold out every Nth frame for evaluation")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--stop", help="pause file: when it appears, checkpoint and exit 75; rerun to resume (default OUT/STOP)")
ap.add_argument("--ckpt_every", type=int, default=10000, help="also checkpoint every N iterations (0: only on pause)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
rng = np.random.default_rng(a.seed)
dev = "cuda"

# ---------- data ----------
meta = json.load(open(f"{a.work}/meta.json"))
W, H = meta["W"], meta["H"]
z = np.load(f"{a.work}/{a.pose}")
idx = z["idx"]
N = len(idx)
c2w = z["c2w"].astype(np.float64)
files = sorted(glob.glob(f"{a.work}/rgba/*.png"))
rgba = np.stack([np.asarray(Image.open(files[i])) for i in idx])
xs = np.where((rgba[..., 3] > 8).any((0, 1)))[0]
ys = np.where((rgba[..., 3] > 8).any((0, 2)))[0]
bx0, bx1 = max(xs[0] - 12, 0), min(xs[-1] + 13, W)
by0, by1 = max(ys[0] - 12, 0), min(ys[-1] + 13, H)
bx1, by1 = bx0 + (bx1 - bx0) // 2 * 2, by0 + (by1 - by0) // 2 * 2
rgba = torch.from_numpy(rgba[:, by0:by1, bx0:bx1].copy()).to(dev)
w, h = int(bx1 - bx0), int(by1 - by0)
S = min(W, H)
cx0, cy0 = (W - S) // 2, (H - S) // 2

# canonical frame: orbit axis -> -Y up (standard 3DGS/COLMAP), origin on the axis, object ~1.8 units tall
C = c2w[:, :3, 3]
m0 = C.mean(0)
Vt = np.linalg.svd(C - m0)[2]
up = Vt[2]
if (-c2w[:, :3, 1].mean(0)) @ up < 0:
    up = -up
e1 = Vt[0]
Rw = np.stack([e1, -up, np.cross(e1, -up)])  # rows: new x, y(down), z; right-handed
xy = (C - m0) @ Vt[:2].T
s_ = np.linalg.lstsq(np.c_[2 * xy, np.ones(N)], (xy ** 2).sum(1), rcond=None)[0]
ctr = m0 + s_[:2] @ Vt[:2]
radius = math.sqrt(s_[2] + s_[:2] @ s_[:2])
f0 = float(z["K"][0, 0])
scale = 1.8 / (h / f0 * radius)
c2w[:, :3, 3] = (c2w[:, :3, 3] - ctr) @ Rw.T * scale
c2w[:, :3, :3] = Rw @ c2w[:, :3, :3]
pts = (z["pts"] - ctr) @ Rw.T * scale
prgb = z["rgb"] / 255.
R0 = radius * scale
ok = np.linalg.norm(pts[:, [0, 2]], axis=1) < 1.5
pts, prgb = pts[ok], prgb[ok]
_, inv, cnt = np.unique(np.floor(pts / 0.03).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
ok = cnt[inv.ravel()] > 8  # sparse-voxel outlier rejection
pts, prgb = pts[ok], prgb[ok]
sel = rng.permutation(len(pts))[:120000]
pts, prgb = pts[sel], prgb[sel]

az = np.unwrap(np.arctan2(c2w[:, 2, 3], c2w[:, 0, 3]))
turn = 2 * np.pi * np.sign(az[-1] - az[0])
gap = np.abs(np.gradient(np.r_[az[-1] - turn, az, az[0] + turn]))[1:-1]
test = np.zeros(N, bool)
if a.holdout:
    test[a.holdout // 2::a.holdout] = True
prob = np.where(test, 0, np.maximum(gap, np.radians(0.3)) ** 0.6)
prob /= prob.sum()
tr = np.where(~test)[0]
print(f"N={N} crop={w}x{h} focal={f0:.0f} radius={R0:.2f} seeds={len(pts)} max step={np.degrees(gap.max()):.1f}deg", flush=True)

if "depth" in z:  # N,s,s on the centred square crop; the COLMAP backend has none
    depth = torch.from_numpy(z["depth"].astype(np.float32) * scale)
else:
    a.lam_depth = 0.
gu = (torch.arange(w, device=dev) + bx0 + .5 - cx0) / S * 2 - 1
gv = (torch.arange(h, device=dev) + by0 + .5 - cy0) / S * 2 - 1
dgrid = torch.stack(torch.meshgrid(gv, gu, indexing="ij")[::-1], -1)[None].float()

# ---------- model ----------
SH0 = 0.28209479177387814
KS = (a.sh + 1) ** 2
P = torch.tensor(pts, dtype=torch.float32, device=dev)
d = np.clip(cKDTree(pts).query(pts, 4)[0][:, 1:].mean(1), 1e-4, 0.02)
d = torch.tensor(d, dtype=torch.float32, device=dev)
par = dict(
    means=P,
    scales=d.log()[:, None].repeat(1, 3),
    quats=F.normalize(torch.randn(len(P), 4, device=dev), dim=-1),
    opacities=torch.logit(torch.full((len(P),), 0.5, device=dev)),
    sh0=((torch.tensor(prgb, dtype=torch.float32, device=dev) - .5) / SH0)[:, None],
    shN=torch.zeros(len(P), KS - 1, 3, device=dev),
)
par = nn.ParameterDict({k: nn.Parameter(v) for k, v in par.items()})
lr = dict(means=1.6e-4, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
opt = {k: torch.optim.Adam([par[k]], lr=lr[k], eps=1e-15) for k in par.keys()}
sched = torch.optim.lr_scheduler.ExponentialLR(opt["means"], 0.01 ** (1 / a.iters))
strat = MCMCStrategy(cap_max=a.cap, refine_start_iter=500, refine_stop_iter=int(a.iters * .8), noise_lr=1e5)
strat.check_sanity(par, opt)
st = strat.initialize_state()


class Frames(nn.Module):
    """Pose deltas, shared focal, deformation + colour. Deformation codes sit on a circular knot grid (smooth in time)."""

    def __init__(s, n, knots=60, dim=24):
        super().__init__()
        s.n, s.k, s.L = n, knots, 4
        s.rot = nn.Parameter(torch.zeros(n, 3))
        s.trn = nn.Parameter(torch.zeros(n, 3))
        s.lf = nn.Parameter(torch.zeros(()))
        s.code = nn.Parameter(torch.randn(knots, dim) * .05)
        s.col = nn.Parameter(torch.zeros(n, 12))
        s.mlp = nn.Sequential(nn.Linear(3 * (1 + 2 * s.L) + dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 6))
        nn.init.zeros_(s.mlp[-1].weight)
        nn.init.zeros_(s.mlp[-1].bias)

    def deform(s, x, i):
        t = float(i) / s.n * s.k
        k0, u = int(t) % s.k, t - int(t)
        c = s.code[k0] * (1 - u) + s.code[(k0 + 1) % s.k] * u
        q = (x[:, None] * (2. ** torch.arange(s.L, device=x.device) * 2.2)[:, None]).flatten(1)
        o = s.mlp(torch.cat([x, q.sin(), q.cos(), c[None].expand(len(x), -1)], -1)) * .05
        return o[:, :3], o[:, 3:]


fr = Frames(N).to(dev)
fopt = torch.optim.Adam([
    dict(params=[fr.rot, fr.trn], lr=2e-4), dict(params=[fr.lf], lr=1e-4), dict(params=[fr.code], lr=2e-3),
    dict(params=fr.mlp.parameters(), lr=5e-4), dict(params=[fr.col], lr=2e-3)])
fsched = torch.optim.lr_scheduler.ExponentialLR(fopt, 0.05 ** (1 / a.iters))
c2w_t = torch.tensor(c2w, dtype=torch.float32, device=dev)


def expm(r):
    th = (r * r).sum(-1).add(1e-12).sqrt()
    k = r / th
    o = torch.zeros((), device=r.device)
    X = torch.stack([o, -k[2], k[1], k[2], o, -k[0], -k[1], k[0], o]).view(3, 3)
    return torch.eye(3, device=r.device) + th.sin() * X + (1 - th.cos()) * X @ X


def pose(i, interp=False):
    if interp:  # held-out frame: borrow the neighbouring corrections
        lo = max([t for t in tr if t < i], default=tr[-1])
        hi = min([t for t in tr if t > i], default=tr[0])
        r, t = (fr.rot[lo] + fr.rot[hi]) / 2, (fr.trn[lo] + fr.trn[hi]) / 2
    else:
        r, t = fr.rot[i], fr.trn[i]
    E = expm(r)
    return E @ c2w_t[i, :3, :3], E @ c2w_t[i, :3, 3] + t


def qmul(p, q):
    a1, b1, c1, d1 = p.unbind(-1)
    a2, b2, c2, d2 = q.unbind(-1)
    return torch.stack([a1*a2 - b1*b2 - c1*c2 - d1*d2, a1*b2 + b1*a2 + c1*d2 - d1*c2,
                        a1*c2 - b1*d2 + c1*a2 + d1*b2, a1*d2 + b1*c2 - c1*b2 + d1*a2], -1)


def sh_basis(v, deg):
    x, y, q = v.unbind(-1)
    b = [torch.full_like(x, SH0)]
    if deg > 0:
        b += [-.4886025119 * y, .4886025119 * q, -.4886025119 * x]
    if deg > 1:
        b += [1.0925484306 * x * y, -1.0925484306 * y * q, .3153915653 * (3 * q * q - 1),
              -1.0925484306 * x * q, .5462742153 * (x * x - y * y)]
    if deg > 2:
        b += [-.5900435899 * y * (3 * x * x - y * y), 2.8906114426 * x * y * q, -.4570457995 * y * (5 * q * q - 1),
              .3731763326 * q * (5 * q * q - 3), -.4570457995 * x * (5 * q * q - 1),
              1.4453057213 * q * (x * x - y * y), -.5900435899 * x * (x * x - 3 * y * y)]
    return torch.stack(b, -1)


def render(R, t, i=None, deg=None, jitter=0., bg=None, focal=None, size=None, pp=None, want_depth=False):
    deg = a.sh if deg is None else deg
    x, q, dx = par["means"], par["quats"], None
    if i is not None and a.deform:
        dx, dr = fr.deform(x.detach(), i)
        x = x + dx
        q = qmul(torch.cat([torch.ones_like(dr[:, :1]), dr], -1), F.normalize(q, dim=-1))
    cam = t.detach()
    if jitter:
        cam = expm(R[:, 0].detach() * math.radians(jitter)) @ cam
    dirs = F.normalize(x.detach() - cam, dim=-1)
    sh = torch.cat([par["sh0"], par["shN"]], 1)[:, :(deg + 1) ** 2]
    col = ((sh_basis(dirs, deg)[:, :, None] * sh).sum(1) + .5).clamp_min(0)
    if i is not None:
        A = (fr.col[i] - fr.col.mean(0)).view(3, 4) * .1  # zero-mean, so it cannot shift the exported colours
        col = col + col @ A[:, :3].T + A[:, 3]
    last = torch.tensor([[0., 0., 0., 1.]], device=dev)
    w2c = torch.cat([torch.cat([R.T, (-R.T @ t)[:, None]], 1), last], 0)
    # gsplat has no gradient w.r.t. K, so the focal correction is applied as an x/y scale on the view matrix
    f = torch.tensor(float(f0 if focal is None else focal), device=dev)
    if focal is None:
        g = fr.lf.exp()
        w2c = torch.stack([g, g, torch.ones_like(g), torch.ones_like(g)])[:, None] * w2c
    ww, hh = size or (w, h)
    px, py = pp or (W / 2 - bx0, H / 2 - by0)
    o = torch.zeros((), device=dev)
    K = torch.stack([f, o, o + float(px), o, f, o + float(py), o, o, o + 1]).view(3, 3)
    img, al, info = rasterization(x, q, par["scales"].exp(), par["opacities"].sigmoid(), col, w2c[None], K[None], ww, hh,
                                  render_mode="RGB+ED" if want_depth else "RGB", packed=False, near_plane=.01)
    rgb, al = img[0, ..., :3], al[0]
    if bg is not None:
        rgb = rgb + (1 - al) * bg
    return rgb, al, (img[0, ..., 3:] if want_depth else None), info, dx


g1 = torch.exp(-(torch.arange(11, device=dev) - 5.) ** 2 / 4.5)
g1 = (g1 / g1.sum())[None]
KER = (g1.T @ g1).expand(3, 1, 11, 11)


def nchw(v):
    return v.permute(2, 0, 1)[None]


def ssim(x, y):
    x, y = nchw(x), nchw(y)
    f_ = lambda v: F.conv2d(v, KER, padding=5, groups=3)
    mx, my = f_(x), f_(y)
    sx, sy, sxy = f_(x * x) - mx * mx, f_(y * y) - my * my, f_(x * y) - mx * my
    return (((2 * mx * my + 1e-4) * (2 * sxy + 9e-4)) / ((mx * mx + my * my + 1e-4) * (sx + sy + 9e-4))).mean()


lp = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
for p_ in lp.parameters():
    p_.requires_grad_(False)


def gt(i, bg):
    g = rgba[i].float() / 255
    al = g[..., 3:]
    return g[..., :3] * al + (1 - al) * bg, al


# ---------- checkpoint / resume ----------
CKPT, STOP = f"{a.out}/train_ckpt.pt", a.stop or f"{a.out}/STOP"
SAME = [k for k in vars(a) if k not in ("work", "out", "stop", "ckpt_every", "iters")]  # must match to resume


def save_ckpt(it):
    torch.save(dict(it=it, args={k: getattr(a, k) for k in SAME}, par={k: v.data for k, v in par.items()},
                    opt={k: o.state_dict() for k, o in opt.items()}, sched=sched.state_dict(), st=st,
                    fr=fr.state_dict(), fopt=fopt.state_dict(), fsched=fsched.state_dict(), rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), CKPT + ".part")
    os.replace(CKPT + ".part", CKPT)  # an interrupted save never replaces a good checkpoint


start = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)  # optimizers place their state; steps stay on CPU
    diff = {k: (ck["args"].get(k), getattr(a, k)) for k in SAME if ck["args"].get(k) != getattr(a, k)}
    if diff:
        raise SystemExit(f"{CKPT} was made with other settings {diff} (saved, now); delete it to start over")
    if ck["it"] > a.iters:
        raise SystemExit(f"{CKPT} is already at iteration {ck['it']}, past --iters {a.iters}")
    par = nn.ParameterDict({k: nn.Parameter(v.to(dev)) for k, v in ck["par"].items()})
    opt = {k: torch.optim.Adam([par[k]], lr=lr[k], eps=1e-15) for k in par.keys()}
    for k, o in opt.items():
        o.load_state_dict(ck["opt"][k])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt["means"], 0.01 ** (1 / a.iters))
    strat.check_sanity(par, opt)
    st = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in ck["st"].items()}
    fr.load_state_dict(ck["fr"])
    fopt.load_state_dict(ck["fopt"])
    rng.bit_generator.state = ck["rng"]
    torch.set_rng_state(ck["torch_rng"])
    torch.cuda.set_rng_state(ck["cuda_rng"])
    start = ck["it"]
    del ck
    # learning rates where a run of this --iters would be now: equal to the saved ones when --iters is unchanged,
    # and a longer run (more --iters than before) carries on as if it had been planned that long
    for o, decay, sc in ((opt["means"], 0.01, sched), (fopt, 0.05, fsched)):
        for g in o.param_groups:
            g["lr"] = g["initial_lr"] * decay ** (start / a.iters)
        sc.last_epoch, sc._last_lr = start, [g["lr"] for g in o.param_groups]
    print(f"resumed at iteration {start} of {a.iters} with {len(par['means'])} gaussians", flush=True)

# ---------- train ----------
t0 = time.time()
depth_end = a.iters * .6
for it in range(start, a.iters):
    if it > start and it % 100 == 0 and os.path.exists(STOP):
        save_ckpt(it)
        print(f"STOP found ({STOP}): checkpointed at iteration {it}; rerun the same command to resume", flush=True)
        raise SystemExit(75)
    if a.ckpt_every and it > start and it % a.ckpt_every == 0:
        save_ckpt(it)
    i = int(rng.choice(N, p=prob))
    deg = min(a.sh, it // 1500)
    bg = torch.rand(3, device=dev) if rng.random() < .8 else torch.full((3,), .5, device=dev)
    jit = float(np.clip(rng.normal(0, a.el_jitter / 2), -75, 75)) if a.el_jitter and rng.random() < .7 else 0.
    use_d = a.lam_depth > 0 and it < depth_end
    R, t = pose(i)
    rgb, al, dep, info, dx = render(R, t, i if it > 1500 else None, deg, jit, bg, want_depth=use_d)
    y, ya = gt(i, bg)
    strat.step_pre_backward(par, opt, st, it, info)
    loss = .8 * (rgb - y).abs().mean() + .2 * (1 - ssim(rgb, y)) + a.lam_alpha * (al - ya).abs().mean()
    if a.lam_lpips and it > 4000 and it % 2 == 0:
        s = 256
        yy, xx = int(rng.integers(0, max(h - s, 1))), int(rng.integers(0, max(w - s, 1)))
        c_ = lambda v: nchw(v[yy:yy + s, xx:xx + s]) * 2 - 1
        loss = loss + 2 * a.lam_lpips * lp(c_(rgb), c_(y)).mean()
    if use_d:
        gd = F.grid_sample(depth[i][None, None].to(dev), dgrid, mode="nearest", align_corners=False)[0, 0, ..., None]
        m = (ya > .95) & (al > .5) & (gd > 0)
        loss = loss + a.lam_depth * (1 - it / depth_end) * F.smooth_l1_loss(dep[m], gd[m], beta=.01)
    sc = par["scales"].exp()
    loss = loss + .01 * par["opacities"].sigmoid().mean() + .01 * sc.mean()
    loss = loss + a.lam_flat * F.relu(sc.max(-1).values / sc.median(-1).values.clamp_min(1e-6) - 4).clamp_max(20).mean()
    loss = loss + .01 * (fr.rot[i].square().sum() + fr.trn[i].square().sum()) + 1e-3 * fr.col[i].square().mean()
    if dx is not None:
        loss = loss + a.lam_def * (dx / .05).square().sum(-1).mean()
    loss.backward()
    if it == 1200:  # a camera parameter without gradient would otherwise just silently stay put
        for nme, q_ in (("rot", fr.rot), ("trn", fr.trn), ("focal", fr.lf)):
            if q_.grad is None or not q_.grad.abs().sum() > 0:
                raise RuntimeError(f"no gradient reaches camera parameter '{nme}'")
    for o in opt.values():
        o.step()
        o.zero_grad(set_to_none=True)
    if it > 1000:
        fopt.step()
    fopt.zero_grad(set_to_none=True)
    strat.step_post_backward(par, opt, st, it, info, lr=sched.get_last_lr()[0])
    sched.step()
    fsched.step()
    if it % 1000 == 0:
        mdx = 0 if dx is None else dx.norm(dim=-1).mean().item()
        print(f"it {it} loss {loss.item():.4f} n={len(par['means'])} focal x{fr.lf.exp().item():.4f} rot={np.degrees(fr.rot.norm(dim=-1).mean().item()):.3f}deg trn={fr.trn.norm(dim=-1).mean().item():.4f} |dx|={mdx:.4f} peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB {time.time()-t0:.0f}s", flush=True)


if start < a.iters:  # kept (GBs at millions of gaussians): a rerun with a larger --iters continues from here
    save_ckpt(a.iters)
elif start == a.iters:
    print(f"training already at {a.iters} iterations; re-exporting", flush=True)


# ---------- evaluate ----------
@torch.no_grad()
def evaluate(ids, held, deformed):
    ps, ss, ls = [], [], []
    bg = torch.full((3,), .5, device=dev)
    for i in ids:
        R, t = pose(i, interp=held)
        rgb = render(R, t, None, bg=bg)[0] if not deformed else render(R, t, int(i), bg=bg)[0]
        rgb = rgb.clamp(0, 1)
        y, ya = gt(i, bg)
        m = (ya > .05).expand_as(y)
        ps.append(-10 * math.log10(((rgb - y) ** 2)[m].mean().item()))
        ss.append(ssim(rgb, y).item())
        ls.append(lp(nchw(rgb) * 2 - 1, nchw(y) * 2 - 1).item())
    return dict(fgPSNR=round(float(np.mean(ps)), 2), SSIM=round(float(np.mean(ss)), 4), LPIPS=round(float(np.mean(ls)), 4))


ev = {"train_canonical": evaluate(tr[::6], False, False), "train_deformed": evaluate(tr[::6], False, True)}
if a.holdout:
    ev["heldout_canonical"] = evaluate(np.where(test)[0], True, False)
    ev["heldout_deformed"] = evaluate(np.where(test)[0], True, True)
ev["n_gaussians"] = len(par["means"])
ev["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
ev["gpu"] = torch.cuda.get_device_name()
print(json.dumps(ev, indent=1))
ev["args"] = vars(a)
json.dump(ev, open(f"{a.out}/metrics.json", "w"), indent=1)

# ---------- export ----------
with torch.no_grad():
    fpx = float(f0 * fr.lf.exp())
    cams = []
    for i in range(N):
        R, t = pose(i, interp=bool(test[i]))
        cams.append(torch.cat([R, t[:, None]], 1).cpu().numpy())
    cams = np.stack(cams)
    keep = (par["opacities"].sigmoid() > .02) & (par["scales"].exp().max(-1).values < .3)
    x = par["means"][keep].cpu().numpy()
    foot = float(np.percentile(x[:, 1], 99.8))
    off = np.array([0, -foot, 0], np.float32)  # feet on y=0, -Y up
    rest = torch.zeros(len(x), 15, 3)
    rest[:, :KS - 1] = par["shN"][keep].cpu()
    rest = rest.permute(0, 2, 1).reshape(len(x), 45).numpy()
    arr = np.concatenate([x + off, np.zeros_like(x), par["sh0"][keep][:, 0].cpu().numpy(), rest,
                          par["opacities"][keep][:, None].cpu().numpy(), par["scales"][keep].cpu().numpy(),
                          F.normalize(par["quats"][keep], dim=-1).cpu().numpy()], 1).astype(np.float32)
    nm = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{k}" for k in range(45)]
    nm += ["opacity"] + [f"scale_{k}" for k in range(3)] + [f"rot_{k}" for k in range(4)]
    rec = np.empty(len(arr), dtype=[(q, "f4") for q in nm])
    for k, q in enumerate(nm):
        rec[q] = arr[:, k]
    PlyData([PlyElement.describe(rec, "vertex")]).write(f"{a.out}/splat.ply")
    cams2 = cams.copy()
    cams2[:, :3, 3] += off
    np.savez(f"{a.out}/cameras.npz", c2w=cams2, focal=fpx, W=W, H=H, idx=idx, first=meta.get("start", 1))

    bg = torch.full((3,), .5, device=dev)
    kw = dict(quality=9, macro_block_size=2)
    wr = imageio.get_writer(f"{a.out}/compare.mp4", fps=24, **kw)  # source | canonical splat
    for i in range(N):
        R, t = pose(i, interp=bool(test[i]))
        r = render(R, t, None, bg=bg)[0].clamp(0, 1)
        wr.append_data((torch.cat([gt(i, bg)[0], r], 1).cpu().numpy() * 255).astype(np.uint8))
    wr.close()
    wr = imageio.get_writer(f"{a.out}/orbit.mp4", fps=30, **kw)  # free orbit, +-35 deg elevation
    o = torch.tensor([0., (float(x[:, 1].min()) + foot) / 2, 0.], device=dev)
    for k in range(360):
        th, el = math.radians(k * 2), math.radians(35 * math.sin(math.radians(k)))
        t = o + R0 * 1.05 * torch.tensor([math.cos(th) * math.cos(el), -math.sin(el), math.sin(th) * math.cos(el)], device=dev)
        fw = F.normalize(o - t, dim=0)
        rt = F.normalize(torch.linalg.cross(torch.tensor([0., 1., 0.], device=dev), fw), dim=0)
        R = torch.stack([rt, torch.linalg.cross(fw, rt), fw], 1)
        r = render(R, t, None, bg=bg, focal=fpx, size=(720, 720), pp=(360, 360))[0].clamp(0, 1)
        wr.append_data((r.cpu().numpy() * 255).astype(np.uint8))
    wr.close()
print("saved", a.out, len(arr))

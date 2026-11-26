"""Video or image sequence (path with #### or %04d, e.g. shot.####.png) -> frames -> soft alpha (BiRefNet HR matting) -> decontaminated RGBA.

out/rgba/%04d.png  straight-alpha foreground   out/grey/%04d.png  fg on flat grey (pose input)
"""
import argparse, subprocess, glob, os, json, re
import numpy as np, cv2, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation
from pymatting import estimate_foreground_ml

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("out")
ap.add_argument("--res", type=int, default=2048)
ap.add_argument("--loop_thresh", type=float, default=4.0, help="mean abs diff under which the last frame is a repeat of the first")
a = ap.parse_args()
a.video = re.sub(r"%0?(\d*)d", lambda m: "#" * int(m.group(1) or 1), a.video, count=1)  # shot.%04d.png == shot.####.png
for d in ("raw", "rgba", "grey"):
    os.makedirs(f"{a.out}/{d}", exist_ok=True)
src = ["-i", a.video]
start = 1  # frame number of the first frame, as a Nuke Read numbers it
if "#" in a.video:  # image sequence: #### -> %04d, starting at the first frame on disk
    seq = sorted(glob.glob(re.sub("#+", lambda m: "[0-9]" * len(m.group()), a.video, count=1)))
    if not seq:
        raise SystemExit("no frames match " + a.video)
    pre, digits, post = re.match(r"(.*?)(#+)(.*)$", a.video).groups()  # first run of #s
    start = int(seq[0][len(pre):len(seq[0]) - len(post)])
    src = ["-start_number", str(start), "-i", pre + "%%0%dd" % len(digits) + post]
    print("sequence", seq[0], "...", os.path.basename(seq[-1]), len(seq), "frames")
subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-fps_mode", "passthrough", "-pix_fmt", "rgb24", f"{a.out}/raw/%04d.png"], check=True)
files = sorted(glob.glob(f"{a.out}/raw/*.png"))
f0, f1 = (cv2.imread(files[i]).astype(np.float32) for i in (0, -1))
loop = float(np.abs(f0 - f1).mean()) < a.loop_thresh
print("frames", len(files), "loop closed:", loop, "diff", float(np.abs(f0 - f1).mean()))

REV = "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f"
local = os.path.join(os.environ.get("GS_MODELS", ""), "birefnet_hr_matting")  # offline copy from scripts/download_models.py
src = dict(pretrained_model_name_or_path=local) if os.environ.get("GS_MODELS") else dict(pretrained_model_name_or_path="ZhengPeng7/BiRefNet_HR-matting", revision=REV)
net = AutoModelForImageSegmentation.from_pretrained(**src, trust_remote_code=True).cuda().eval().half()
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda(); std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()
for f in tqdm(files, mininterval=10):
    im = np.array(Image.open(f).convert("RGB")); H, W = im.shape[:2]
    x = torch.from_numpy(cv2.resize(im, (a.res, a.res), interpolation=cv2.INTER_CUBIC)).cuda().permute(2, 0, 1).float() / 255
    with torch.no_grad():
        al = net(((x - mean) / std)[None].half())[-1].float().sigmoid()
    al = torch.nn.functional.interpolate(al, (H, W), mode="area")[0, 0].clamp(0, 1).cpu().numpy()
    # keep the main connected blob (drops stray background hits) without hard-cutting soft edges
    n, lab, st, _ = cv2.connectedComponentsWithStats((al > 0.05).astype(np.uint8), connectivity=8)
    if n > 1:
        k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); al = al * (lab == k)
    al[al < 0.02] = 0
    fg = estimate_foreground_ml(im.astype(np.float64) / 255, al.astype(np.float64))
    rgba = np.dstack([np.clip(fg, 0, 1), al]); b = os.path.basename(f)
    Image.fromarray((rgba * 255 + .5).astype(np.uint8)).save(f"{a.out}/rgba/{b}")
    g = fg * al[..., None] + 0.5 * (1 - al[..., None])
    Image.fromarray((np.clip(g, 0, 1) * 255 + .5).astype(np.uint8)).save(f"{a.out}/grey/{b}")
json.dump(dict(n=len(files), loop=loop, W=W, H=H, start=start), open(f"{a.out}/meta.json", "w"))

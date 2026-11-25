#!/usr/bin/env python3
"""Download every model the offline pipeline needs into one plain directory (no symlinks, no cache layout).

Standard library only, so it runs on any internet-connected machine, with or without the container:

  python3 scripts/download_models.py models                 # everything
  python3 scripts/download_models.py models --only main     # commercial-friendly set: no Pi4
  python3 scripts/download_models.py models --check         # verify a copied directory, needs no network

Copy the whole directory anywhere and point GS_TURNTABLE_MODELS at it. Interrupted downloads resume.
"""
import argparse
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request

HF = "https://huggingface.co/{}/resolve/{}/{}"
BIR = ("ZhengPeng7/BiRefNet_HR-matting", "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f")
PI4 = ("yyfz233/Pi4", "59f307317fb630c9c5a5aa635038d98ed106aaf5")
CKPT = "torch/hub/checkpoints/"
# group, path in the model directory, url, sha256 ("" = not pinned), size in bytes
FILES = [
    ("main", "birefnet_hr_matting/model.safetensors", HF.format(*BIR, "model.safetensors"),
     "a5a4de698739ea5e0e8bbab28e1b293dde95092b87a442d566cbc585c53cef55", 444473596),
    ("main", "birefnet_hr_matting/birefnet.py", HF.format(*BIR, "birefnet.py"), "2a45b4e0ece72d7c4212bca1a988e7d7e52bfe9f98ec59c58b8809c8a8b7a831", 92134),
    ("main", "birefnet_hr_matting/BiRefNet_config.py", HF.format(*BIR, "BiRefNet_config.py"), "e7b8c2a74f6cea6a59553d517f71d47f2c1d90e670a13416af17c25fe2f3dc52", 298),
    ("main", "birefnet_hr_matting/config.json", HF.format(*BIR, "config.json"), "d0b779ee9a76bf079673c5852c346f439591f016ca0d778a87bbd1dcbe24398a", 416),
    ("main", CKPT + "vgg16-397923af.pth", "https://download.pytorch.org/models/vgg16-397923af.pth", "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0", 553433881),
    ("main", CKPT + "commercial_loma_B.pt",
     "https://github.com/davnords/storage/releases/download/loma/commercial_loma_B.pt",
     "aa7a11127734de4d52334edb3a9daf697441fcb572f83132be859f1d9cd1dc8a", 757887713),
    ("main", CKPT + "dinov2_vitl14_pretrain.pth",
     "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
     "d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428", 1217586395),
    ("pi4", "pi4/model.safetensors", HF.format(*PI4, "model.safetensors"),
     "fefacac1fb3d23d33c628c41f40d3ce3be98ca064c271948371805db7b5a0e69", 1995767812),
    ("pi4", "pi4/LICENSE_MODEL", HF.format(*PI4, "LICENSE_MODEL"), "", 20024),
    ("pi4", "pi4/NOTICE", HF.format(*PI4, "NOTICE"), "", 2290),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def problem(path, digest, size):
    if not os.path.isfile(path):
        return "missing"
    if size and os.path.getsize(path) != size:
        return "wrong size (%d, expected %d)" % (os.path.getsize(path), size)
    if digest and sha256(path) != digest:
        return "checksum mismatch"
    return None


def fetch(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    for attempt in range(6):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
        if have:
            req.add_header("Range", "bytes=%d-" % have)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:  # server ignored the range: start again
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                done, shown = have, 0.0
                with open(part, "ab" if have else "wb") as f:
                    for b in iter(lambda: r.read(1 << 20), b""):
                        f.write(b)
                        done += len(b)
                        if time.time() - shown > 5:
                            shown = time.time()
                            print("    %.0f / %.0f MB" % (done / 1e6, total / 1e6), flush=True)
            if total and done < total:
                raise OSError("connection closed early")
            os.replace(part, path)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:  # the partial file does not fit the remote one
                os.remove(part)
            elif e.code < 500 and e.code != 429:
                raise
            err = e
        except OSError as e:
            err = e
        print("    retry after: %s" % err, flush=True)
        time.sleep(3 * (attempt + 1))
    raise OSError("giving up on " + url)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--only", choices=["main", "pi4"], help="default: both. 'main' holds nothing from Pi4 (CC BY-NC 4.0)")
    ap.add_argument("--check", action="store_true", help="only verify local files")
    a = ap.parse_args()
    files = [f for f in FILES if a.only in (None, f[0])]
    bad = []
    for _, rel, url, digest, size in files:
        path = os.path.join(a.directory, *rel.split("/"))
        why = problem(path, digest, size)
        if why and not a.check:
            print("get  %s" % rel, flush=True)
            fetch(url, path)
            why = problem(path, digest, size)
        print("%-5s%s%s" % ("BAD" if why else "ok", rel, "  <- " + why if why else ""), flush=True)
        if why:
            bad.append(rel)
    if bad:
        sys.exit("%d of %d model files missing or wrong" % (len(bad), len(files)))
    print("all %d model files present and verified in %s" % (len(files), os.path.abspath(a.directory)))


if __name__ == "__main__":
    main()
